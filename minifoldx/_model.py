"""MiniFold core model in MLX: projections, FoldingTrunk, and the top-level MiniFoldMLX.

ESM2 is NOT included — the caller provides pre-computed ESM2 representations and
attention maps (from the existing MLX ESM2 model).  All MiniFold-specific layers run
in MLX, including rigid-body geometry (via mlx_rigid.py).  No PyTorch is required
for inference.

Weight key conventions
----------------------
PyTorch Sequential layers with non-consecutive indices (ReLU at idx 1 or 2 has no params)
are flattened to named attributes to avoid MLX tree_unflatten key-type issues:
  HF key: ``fc_s.0.weight``    → MLX attr: ``fc_s_0.weight``
  HF key: ``fc_s.2.weight``    → MLX attr: ``fc_s_2.weight``
  (similarly for fc_z, fold.fc_out, sz_project.s_z_mlp_{0,1,3}, sz_project.combiner_0,
   structure_module.transitions.N.mlp_{0,1,3}, angle_resnet.layers.N.mlp_{1,3})
The convert_hf_to_mlx.py converter applies the corresponding key renaming.
"""

from __future__ import annotations

import re
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from minifoldx._miniformer import MiniFormer
from minifoldx._structure import StructureModuleMLX
from minifoldx._heads import AuxiliaryHeadsMLX


# ── Sequential-key flattening ─────────────────────────────────────────────────
# PyTorch nn.Sequential uses integer indices as submodule names.  MLX's
# tree_unflatten converts numeric-string keys to Python ints and then fails with
# KeyError when it tries to look up int 0 in a Python dict with string key "0".
# We rename  ``parent.N.field``  →  ``parent_N.field``  for every affected
# Sequential attribute so the resulting attribute name is a plain Python identifier.
# This remapping is applied at load time so both freshly-converted and legacy NPZ
# files are handled transparently.
_SEQ_REMAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(fc_s)\.(0|2)\."),          r"\1_\2."),
    (re.compile(r"^(fc_z)\.(0|2)\."),          r"\1_\2."),
    (re.compile(r"^(fold\.fc_out)\.(0|2)\."),  r"\1_\2."),
    (re.compile(r"^(sz_project\.s_z_mlp)\.(0|1|3)\."),   r"\1_\2."),
    (re.compile(r"^(sz_project\.combiner)\.(0)\."),       r"\1_\2."),
    (re.compile(r"^(structure_module\.transitions\.\d+\.mlp)\.(0|1|3)\."),        r"\1_\2."),
    (re.compile(r"^(structure_module\.angle_resnet\.layers\.\d+\.mlp)\.(1|3)\.")  , r"\1_\2."),
]


def _remap_weight_key(key: str) -> str:
    for pat, repl in _SEQ_REMAP:
        key = pat.sub(repl, key)
    return key


# ── Helper modules ────────────────────────────────────────────────────────────

class SequenceToPair(nn.Module):
    """Project sequence repr to an outer-product pairwise repr."""

    def __init__(self, sequence_state_dim: int, inner_dim: int, pairwise_state_dim: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim)

    def __call__(self, sequence_state: mx.array) -> mx.array:
        s = self.layernorm(sequence_state)
        s = self.proj(s)
        half = s.shape[-1] // 2
        q, k = s[..., :half], s[..., half:]
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        return self.o_proj(mx.concatenate([prod, diff], axis=-1))


class RelativePosition(nn.Module):
    """Relative position embedding for pairwise distances."""

    def __init__(self, bins: int, pairwise_state_dim: int):
        super().__init__()
        self.bins      = bins
        self.embedding = nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def __call__(self, residue_index: mx.array, mask: mx.array) -> mx.array:
        """
        residue_index : (1, L) or (B, L) int array
        mask          : (B, L, L) pair mask  — 0 → use padding embedding (idx 0)
        """
        diff = residue_index[:, None, :] - residue_index[:, :, None]  # (B, L, L) or (1, L, L)
        diff = mx.clip(diff, -self.bins, self.bins)
        diff = diff + self.bins + 1
        diff = mx.where(mask != 0, diff, mx.zeros_like(diff))
        return self.embedding(diff)


class PairToSequence(nn.Module):
    """Decode pair representation back to sequence representation.

    Flat attribute naming (see module docstring):
      s_z_mlp_0 = LayerNorm, s_z_mlp_1 = Linear, s_z_mlp_3 = Linear  (idx 2 is ReLU)
      combiner_0 = Linear
    """

    def __init__(self, c_z: int = 128, c_s: int = 1024, c_s_out: int | None = None):
        super().__init__()
        if c_s_out is None:
            c_s_out = c_s
        self.s_z_mlp_0 = nn.LayerNorm(c_z)
        self.s_z_mlp_1 = nn.Linear(c_z, c_z)
        self.s_z_mlp_3 = nn.Linear(c_z, c_z)
        self.combiner_0 = nn.Linear(2 * c_z + c_s, c_s_out)

    def __call__(self, s_z: mx.array, s_s_in: mx.array, pair_mask: mx.array) -> mx.array:
        x = self.s_z_mlp_0(s_z)
        x = self.s_z_mlp_1(x)
        x = mx.maximum(x, 0)    # ReLU
        x = self.s_z_mlp_3(x)

        x = x * pair_mask[..., None]

        norm_c = mx.maximum(pair_mask.sum(axis=2), 1.0)[..., None]
        s_s_c  = x.sum(axis=2) / norm_c

        norm_r = mx.maximum(pair_mask.sum(axis=1), 1.0)[..., None]
        s_s_r  = x.sum(axis=1) / norm_r

        return self.combiner_0(mx.concatenate([s_s_c, s_s_r, s_s_in], axis=-1))


class FoldingTrunk(nn.Module):
    """FoldingTrunk: combine seq/pair features + run MiniFormer recycling.

    fc_out_0 = first Linear, fc_out_2 = second Linear
    (idx 1 in PyTorch Sequential is a no-param ReLU — flattened, see module docstring).
    """

    def __init__(self, c_s: int, c_z: int, bins: int,
                 disto_bins: int = 64, num_layers: int = 48):
        super().__init__()
        self.disto_bins = disto_bins

        self.positional_embedding = RelativePosition(bins, c_z)
        self.seq_to_pair          = SequenceToPair(c_s, c_z // 2, c_z)
        self.projection           = nn.Linear(c_z * 3, c_z)
        self.recycle              = nn.Linear(disto_bins, c_z)
        self.miniformer           = MiniFormer(c_z, blocks=num_layers)
        self.fc_out_0 = nn.Linear(c_z, c_z)
        self.fc_out_2 = nn.Linear(c_z, disto_bins)

    def __call__(self, s_s: mx.array, s_z: mx.array, mask: mx.array,
                 num_recycling: int = 0):
        """
        s_s  : (B, T, c_s)
        s_z  : (B, T, T, c_z)
        mask : (B, T)  float or bool
        """
        B, L = s_s.shape[:2]
        _timing = getattr(self, '_timing', True)

        pair_mask = mask[:, None, :] * mask[:, :, None]  # (B, L, L)

        # Positional embeddings — broadcast over batch
        residx = mx.arange(L, dtype=mx.int32)[None, :]  # (1, L)

        if _timing: t0 = time.perf_counter()
        s_z = mx.concatenate([
            s_z,
            self.seq_to_pair(s_s),
            self.positional_embedding(residx, pair_mask),
        ], axis=-1)
        s_z = self.projection(s_z)
        if _timing:
            mx.eval(s_z)
            print(f"    [timing]   seq_to_pair + pos_emb + projection: {time.perf_counter()-t0:.3f}s")

        # Recycling loop
        dists = mx.zeros((*s_z.shape[:3], self.disto_bins), dtype=s_z.dtype)
        for i in range(num_recycling + 1):
            if _timing: t_r = time.perf_counter()

            s_z_c = s_z + self.recycle(dists)
            try:
                _mf = object.__getattribute__(self, '_compiled_miniformer')
            except AttributeError:
                _mf = self.miniformer
            s_z_c = _mf(s_z_c, pair_mask)

            # Symmetrise and pass through output MLP
            x = self.fc_out_0(s_z_c + s_z_c.transpose(0, 2, 1, 3))
            x = mx.maximum(x, 0)   # ReLU
            preds = self.fc_out_2(x)

            # Soft one-hot for next recycling iteration (stop gradient)
            dists_idx = mx.argmax(mx.stop_gradient(preds), axis=-1)
            dists = (mx.arange(self.disto_bins) == dists_idx[..., None]).astype(s_z.dtype)

            if _timing:
                mx.eval(preds)
                print(f"    [timing]   MiniFormer recycle {i}/{num_recycling}: {time.perf_counter()-t_r:.3f}s")

        return preds, s_z_c


# ── Top-level model ───────────────────────────────────────────────────────────

class MiniFoldMLX(nn.Module):
    """MiniFold model in MLX, accepting ESM2 output from the external MLX ESM2 model.

    The ESM2 backbone (``vincentamato/mlx-esm-2``) must be loaded separately and
    passed as ``esm_model`` to the constructor.  Its weights are NOT stored inside
    this module.

    Usage::

        from mlx_esm import ESM2  # vincentamato/mlx-esm-2
        tokenizer, esm = ESM2.from_pretrained("./minifold_cache/checkpoints")

        model = MiniFoldMLX.from_pretrained("./minifoldx_48L", esm_model=esm)

        tokens_mx = tokenizer(sequences)
        output    = model(tokens_mx)
    """

    # ── Config defaults (48L model trained on ESM2-3B) ────────────────────────
    _DEFAULTS = dict(
        esm_embed_dim       = 2560,
        esm_num_layers      = 36,
        esm_attention_heads = 40,
        c_s                 = 1024,
        c_z                 = 128,
        num_blocks          = 48,
        no_bins             = 64,
        plddt_no_bins       = 50,
        plddt_c_hidden      = 128,
        # structure module
        sm_c_resnet         = 128,
        sm_head_dim         = 64,
        sm_no_heads         = 16,
        sm_no_blocks        = 8,
        sm_no_resnet_blocks = 2,
        sm_no_angles        = 7,
        sm_epsilon          = 1e-5,
    )

    def __init__(self, config: dict | None = None, esm_model=None):
        super().__init__()
        cfg = {**self._DEFAULTS, **(config or {})}

        esm_embed_dim       = cfg["esm_embed_dim"]
        esm_num_layers      = cfg["esm_num_layers"]
        esm_attention_heads = cfg["esm_attention_heads"]
        attn_dim            = esm_num_layers * esm_attention_heads
        c_s                 = cfg["c_s"]
        c_z                 = cfg["c_z"]
        num_blocks          = cfg["num_blocks"]
        no_bins             = cfg["no_bins"]

        # Projection layers (flattened from PyTorch Sequential, see module docstring)
        self.fc_s_0 = nn.Linear(esm_embed_dim, c_s)
        self.fc_s_2 = nn.Linear(c_s, c_s)
        self.fc_z_0 = nn.Linear(attn_dim, c_z)
        self.fc_z_2 = nn.Linear(c_z, c_z)

        # Folding trunk
        self.fold = FoldingTrunk(c_s=c_s, c_z=c_z, bins=32,
                                  disto_bins=no_bins, num_layers=num_blocks)

        # Pair-to-sequence bridge
        self.sz_project = PairToSequence(c_z=c_z, c_s=c_s)

        # Structure module (neural-network layers only)
        self.structure_module = StructureModuleMLX(
            c_s             = c_s,
            c_z             = c_z,
            c_resnet        = cfg["sm_c_resnet"],
            head_dim        = cfg["sm_head_dim"],
            no_heads        = cfg["sm_no_heads"],
            no_blocks       = cfg["sm_no_blocks"],
            no_resnet_blocks= cfg["sm_no_resnet_blocks"],
            no_angles       = cfg["sm_no_angles"],
            epsilon         = cfg["sm_epsilon"],
        )

        # pLDDT head
        self.aux_heads = AuxiliaryHeadsMLX(
            no_bins  = cfg["plddt_no_bins"],
            c_in     = c_s,
            c_hidden = cfg["plddt_c_hidden"],
        )

        # Store ESM model as a plain attribute (bypasses MLX parameter tracking).
        # Its weights are not included in this module's load/save.
        object.__setattr__(self, "_esm_model", esm_model)
        object.__setattr__(self, "_esm_num_layers", esm_num_layers)

    # ── Float16 ───────────────────────────────────────────────────────────────

    def _convert_dtype(self, dtype) -> None:
        """Cast all model parameters to the given MLX dtype."""
        def _cast(tree):
            if isinstance(tree, mx.array):
                return tree.astype(dtype)
            elif isinstance(tree, dict):
                return {k: _cast(v) for k, v in tree.items()}
            elif isinstance(tree, list):
                return [_cast(v) for v in tree]
            return tree

        self.update(_cast(self.parameters()))
        object.__setattr__(self, '_infer_dtype', dtype)
        print(f"All model parameters cast to {dtype}.")

    def convert_to_fp16(self) -> None:
        """Cast all model parameters to float16."""
        self._convert_dtype(mx.float16)

    def convert_trunk_to_fp16(self) -> None:
        """Cast only the FoldingTrunk (MiniFormer) to float16; leave StructureModule,
        AuxHeads, and PairToSequence in float32 to avoid IPA overflow.

        This gives most of the memory/speed benefit of fp16 (MiniFormer is the
        bulk of compute) while keeping numerically sensitive layers in fp32.
        """
        def _cast_fp16(tree):
            if isinstance(tree, mx.array):
                return tree.astype(mx.float16)
            elif isinstance(tree, dict):
                return {k: _cast_fp16(v) for k, v in tree.items()}
            elif isinstance(tree, list):
                return [_cast_fp16(v) for v in tree]
            return tree

        self.fold.update(_cast_fp16(self.fold.parameters()))
        mx.eval(self.fold.parameters())
        object.__setattr__(self, '_infer_dtype', mx.float16)
        print("FoldingTrunk cast to float16; StructureModule/AuxHeads remain float32.")

    def convert_to_bf16(self) -> None:
        """Cast all model parameters to bfloat16 and apply norm_out RMSNorm patch.

        bfloat16 has the same exponent range as float32 (no overflow risk) but
        half the memory footprint.  MLX has native Metal bfloat16 matmul kernels
        on Apple Silicon, so this can be faster than float32 without the overflow
        hazard of float16.

        Also replaces the TriangularUpdate norm_out LayerNorm with RMSNorm,
        absorbing the LayerNorm bias into the downstream linear biases.  This is
        safe (empirically <2 pLDDT loss) and gives ~1.12x MiniFormer speedup.
        norm_in is left as LayerNorm — its mean subtraction is load-bearing.
        """
        self._convert_dtype(mx.bfloat16)
        self.fold.miniformer.patch_norms_rms_absorb_bias(
            triangular=True, tri_norm_in=False, tri_norm_out=True)

    # ── Compilation ───────────────────────────────────────────────────────────

    def enable_compile(self) -> None:
        """Wrap FoldingTrunk with mx.compile for fused Metal kernel dispatch.

        After calling this, subsequent forward passes will use the compiled fold.
        The first call pays the compilation cost; later calls reuse the cached graph.
        Per-recycling timing prints are disabled inside the compiled function
        (side effects cannot run inside a compiled graph).
        """
        self.fold._timing = False   # suppress prints inside the compiled fn
        object.__setattr__(self, '_compiled_fold', mx.compile(self.fold))
        print("FoldingTrunk compiled with mx.compile — first call will pay compilation cost.")

    def enable_bf16_esm2(self) -> None:
        """Dequantize ESM2 weights from int8 to bfloat16 in-place.

        If the model was loaded from an int8-quantized checkpoint (ESM2_MiniFold_int8),
        this replaces every QuantizedLinear in ESM2 with a regular bf16 Linear.

        NOTE: For single-sequence protein folding inference this is typically SLOWER
        than int8, not faster.  ESM2 inference at batch_size=1 is memory-bandwidth-bound:
        the full 3B-parameter weight read per forward pass costs more in bf16 (~5.6GB)
        than in int8 (~3.5GB), outweighing the AMX compute advantage of bf16 matmul.
        Benchmarks on Apple Silicon (M-series) show 0.71–0.90× for L=100–400.

        Use this only if you need higher numerical precision, are running very large
        batch sizes (where compute becomes the bottleneck), or are diagnosing
        quantization quality. For normal AlphaTracer inference, keep the default int8.

        Memory cost: +~2GB (int8+scales ~3.5GB → bf16 ~5.6GB for ESM2-3B).
        Must be called AFTER load_model(). Safe to call on a non-quantized model
        (no-op if no QuantizedLinear layers are found).
        """
        import mlx.nn as _nn

        def _dequant_linear(q: "_nn.QuantizedLinear") -> "_nn.Linear":
            w = mx.dequantize(
                q.weight, q.scales, q.biases, q.group_size, q.bits
            ).astype(mx.bfloat16)
            lin = _nn.Linear(w.shape[1], w.shape[0], bias=q.bias is not None)
            lin.weight = w
            if q.bias is not None:
                lin.bias = q.bias.astype(mx.bfloat16)
            return lin

        esm = self._esm_model
        n_replaced = 0
        for i in range(esm.num_layers):
            layer = getattr(esm, f'layer_{i}')
            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                old = getattr(layer.self_attn, proj)
                if isinstance(old, _nn.QuantizedLinear):
                    setattr(layer.self_attn, proj, _dequant_linear(old))
                    n_replaced += 1
            for fc in ['fc1', 'fc2']:
                old = getattr(layer, fc)
                if isinstance(old, _nn.QuantizedLinear):
                    setattr(layer, fc, _dequant_linear(old))
                    n_replaced += 1
        mx.eval(esm.parameters())
        if n_replaced:
            print(f"ESM2 dequantized: {n_replaced} QuantizedLinear → bf16 Linear.")
        else:
            print("enable_bf16_esm2: no QuantizedLinear layers found (already bf16?).")

    def enable_sgmm_gate(self) -> None:
        """Enable the fused SGMM LayerNorm+gating Metal kernel for TriangularUpdate.

        Replaces the norm_in + proj_in*sigmoid(gate_in) sequence in each of the 48
        TriangularUpdate blocks with a single Metal kernel that uses simdgroup matrix
        multiply to keep x_norm on-chip.  Benchmarked ~2× speedup on the gating step.

        Must be called AFTER convert_to_bf16() since the kernel requires bfloat16 weights.
        """
        self.fold.miniformer.patch_sgmm_gate()

    def enable_sgmm_out_gate(self) -> None:
        """Enable the fused SGMM LayerNorm+gating Metal kernel for the output gating
        step of TriangularUpdate (K=64 → N=128).

        Must be called AFTER convert_to_bf16() since the kernel requires bfloat16 weights.
        """
        self.fold.miniformer.patch_sgmm_out_gate()

    def enable_compile_miniformer(self) -> None:
        """Wrap only MiniFormer with mx.compile, leaving ESM2 and StructureModule uncompiled.

        More targeted than enable_compile(): only the L×L pair ops are compiled,
        avoiding the pathological slowdowns seen when ESM2 / StructureModule shapes
        also get compiled.  First call per unique sequence length pays compilation cost.
        """
        compiled = mx.compile(self.fold.miniformer)
        object.__setattr__(self.fold, '_compiled_miniformer', compiled)
        print("MiniFormer compiled with mx.compile — first call per shape pays compilation cost.")

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_dir: str, esm_model=None):
        """Load from a directory created by ``convert_hf_to_mlx.py``.

        Parameters
        ----------
        model_dir  : path to directory containing ``minifold.npz`` and ``config.json``
        esm_model  : pre-loaded MLX ESM2 model (from ``vincentamato/mlx-esm-2``)
        """
        import json, os
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}

        model = cls(config=cfg, esm_model=esm_model)

        weights_path = os.path.join(model_dir, "minifold.npz")
        if os.path.exists(weights_path):
            weights = mx.load(weights_path)
            # Remap Sequential dict-index keys (e.g. fc_s.0.weight → fc_s_0.weight)
            # so they match the flat named attributes used in this module.
            remapped = [((_remap_weight_key(k)), v) for k, v in weights.items()]
            model.load_weights(remapped)
            print(f"Loaded {len(remapped)} tensors from {weights_path}")
        else:
            print(f"[WARN] No weights file found at {weights_path} — using random init")

        return model

    # ── Forward ───────────────────────────────────────────────────────────────

    def __call__(
        self,
        tokens_mx: mx.array,
        seq_mask: mx.array | None = None,
        aatype_mx: mx.array | None = None,
        num_recycling: int = 3,
    ) -> dict:
        """Run structure prediction.

        Parameters
        ----------
        tokens_mx    : (B, T) MLX int — ESM2 token IDs
        seq_mask     : (B, T) MLX array — 1 for valid residues (default: all 1s)
        aatype_mx    : (B, T) MLX int array — residue type indices
                       (required for full atom-position output)
        num_recycling : number of recycling iterations

        Returns
        -------
        dict with keys:
          ``plddt``               (np.ndarray, B×T) — per-residue confidence
          ``preds``               (np.ndarray, B×T×T×bins) — distogram logits
          ``final_atom_positions``(np.ndarray, B×T×37×3) — only when aatype_mx given
          ``final_atom_mask``     (np.ndarray, B×T×37)   — only when aatype_mx given
        """
        from minifoldx._rigid import build_all_atom_positions

        esm_model = object.__getattribute__(self, "_esm_model")
        num_layers = object.__getattribute__(self, "_esm_num_layers")

        if esm_model is None:
            raise RuntimeError("No ESM2 model provided. Pass esm_model= to from_pretrained().")

        B, N = tokens_mx.shape  # N = sequence length (no BOS/EOS — matches fair-ESM2 behaviour)

        # ── ESM2 forward ──────────────────────────────────────────────────────
        # Tokens are passed WITHOUT BOS/EOS, exactly matching the PyTorch fair-ESM2
        # path in predict.py (alphabet.encode() returns residue tokens only).
        # MLX ESM2 called without BOS/EOS produces representations identical to
        # PyTorch fair-ESM2 to numerical precision (r=1.0, max diff ~1e-5).
        t0 = time.perf_counter()
        esm_out = esm_model(tokens_mx, repr_layers=[num_layers], need_head_weights=True)
        esm_repr = esm_out["representations"][num_layers]   # (B, N, embed_dim)
        esm_attn = esm_out["attentions"]                    # (B, num_layers, H, N, N)
        mx.eval(esm_repr, esm_attn)
        print(f"  [timing] ESM2 forward:       {time.perf_counter()-t0:.3f}s")

        if seq_mask is None:
            seq_mask = mx.ones((B, N), dtype=mx.float32)
        else:
            seq_mask = seq_mask.astype(mx.float32)

        # ── Cast ESM2 outputs to match model weight dtype (fp16 / bf16 / fp32) ─
        _infer_dtype = getattr(self, '_infer_dtype', None)
        if _infer_dtype is not None:
            esm_repr = esm_repr.astype(_infer_dtype)
            esm_attn = esm_attn.astype(_infer_dtype)
            seq_mask = seq_mask.astype(_infer_dtype)

        # ── Reshape attentions ────────────────────────────────────────────────
        # PyTorch fair-ESM2 returns attentions as (B, T, T, num_layers, H).
        # PyTorch MiniFold then does view(B, N, N, -1) → (B, N, N, num_layers*H).
        # MLX ESM2 returns attentions as (B, num_layers, H, T, T), so we must
        # transpose to (B, T, T, num_layers, H) first to match the PT memory layout
        # before flattening.  Without the transpose, the reshape produces r≈0 with
        # PyTorch's fc_z input, explaining the ~9 pLDDT gap.
        t0 = time.perf_counter()
        s_z = esm_attn.transpose(0, 3, 4, 1, 2).reshape(B, N, N, -1)  # (B, N, N, num_layers*H)

        # ── Projections ───────────────────────────────────────────────────────
        s_s = mx.maximum(self.fc_s_0(esm_repr), 0)
        s_s = self.fc_s_2(s_s)
        s_z = mx.maximum(self.fc_z_0(s_z), 0)
        s_z = self.fc_z_2(s_z)
        mx.eval(s_s, s_z)
        print(f"  [timing] fc_s + fc_z:        {time.perf_counter()-t0:.3f}s")

        # ── FoldingTrunk ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        print(f"  [timing] FoldingTrunk ({num_recycling+1} recycling passes):")
        try:
            _fold = object.__getattribute__(self, '_compiled_fold')
        except AttributeError:
            _fold = self.fold
        preds, s_z = _fold(s_s, s_z, seq_mask, num_recycling=num_recycling)
        # fp16: clamp FoldingTrunk outputs to safe range before downstream layers
        if s_z.dtype == mx.float16:
            preds = mx.clip(preds, -65000, 65000)
            s_z   = mx.clip(s_z,   -65000, 65000)
        mx.eval(preds, s_z)
        print(f"  [timing] FoldingTrunk total: {time.perf_counter()-t0:.3f}s")

        # ── PairToSequence ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        # fp16 s_s can reach ~6500, causing combiner_0 (1280-input Linear) to overflow
        # fp16 max (~65504).  Cast s_s to fp32 here; FoldingTrunk already normalised it
        # internally via SequenceToPair's LayerNorm so fp16 there was fine.
        s_s_stable = s_s.astype(mx.float32) if s_s.dtype == mx.float16 else s_s
        single = self.sz_project(s_z, s_s_stable, pair_mask)
        mx.eval(single)
        print(f"  [timing] PairToSequence:     {time.perf_counter()-t0:.3f}s")

        # ── Structure module (MLX NN part) ────────────────────────────────────
        # fp16: IPA dot-products overflow — cast single and s_z to fp32 first
        t0 = time.perf_counter()
        if single.dtype == mx.float16:
            single = single.astype(mx.float32)
            s_z    = s_z.astype(mx.float32)
        sm_out = self.structure_module(single, s_z, seq_mask)
        mx.eval(sm_out["s"], sm_out["angles"], sm_out["n_ca_c"])
        print(f"  [timing] StructureModule:    {time.perf_counter()-t0:.3f}s")

        # ── pLDDT (stays in MLX) ──────────────────────────────────────────────
        t0 = time.perf_counter()
        plddt = self.aux_heads(sm_out["s"])
        mx.eval(plddt)
        print(f"  [timing] AuxHeads (pLDDT):   {time.perf_counter()-t0:.3f}s")

        result = {
            "preds": np.array(preds.astype(mx.float32)),
            "plddt": np.array(plddt.astype(mx.float32)),
        }

        if aatype_mx is None:
            return result

        # ── Pure-MLX rigid-body geometry ──────────────────────────────────────
        t0 = time.perf_counter()
        atom37_pos, atom37_mask = build_all_atom_positions(
            sm_out["n_ca_c"], sm_out["angles"], aatype_mx,
        )
        mx.eval(atom37_pos, atom37_mask)
        print(f"  [timing] RigidBody (MLX):    {time.perf_counter()-t0:.3f}s")

        result["final_atom_positions"] = np.array(atom37_pos)
        result["final_atom_mask"]      = np.array(atom37_mask)
        return result
