"""minifoldx — MLX MiniFold structure prediction package.

Public API
----------
load_model(mlx_esm_path, mlx_minifold_path, *, bf16, fp16, compile, clamp_val)
    Load MLX ESM2 + MiniFoldMLX and return (tokenizer, model).

predict_sequence(seq_id, seq, model, tokenizer, *, num_recycling, max_seq_len)
    Predict a single sequence; returns PDB string or None if skipped.

predict_batch(sequences, model, tokenizer, *, num_recycling, batch_tokens, max_seq_len)
    Predict a list of (seq_id, seq) pairs; returns {seq_id: pdb_str}.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from minifoldx._model import MiniFoldMLX
from minifoldx._data import (
    prepare_input,
    pad_tokens,
    pad_mask,
    pad_aatype,
    output_to_pdb,
    create_batches,
)

__version__ = "0.1.0"

__all__ = [
    "load_model",
    "predict_sequence",
    "predict_batch",
    "MiniFoldMLX",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    mlx_esm_path: str,
    mlx_minifold_path: str,
    *,
    bf16: bool = True,
    fp16: bool = False,
    compile: bool = False,
    clamp_val: float = 0.0,
    ckpt_path: Optional[str] = None,
):
    """Load MLX ESM2 + MiniFoldMLX.

    Parameters
    ----------
    mlx_esm_path      : path to the MLX ESM2 checkpoint directory
                        (mlx-esm2_t36_3B_UR50D_finetuned)
    mlx_minifold_path : path to the MiniFold MLX weights directory
                        (output of convert_hf_to_mlx.py)
    bf16              : cast parameters to bfloat16 (default True)
    fp16              : cast parameters to float16 (mutually exclusive with bf16)
    compile           : wrap FoldingTrunk with mx.compile for fused Metal dispatch
    clamp_val         : clamp MiniFormer activations to [-v, v] (0 = disabled)
    ckpt_path         : optional path to original MiniFold .ckpt for fine-tuned
                        ESM2 layer patching (not needed with the finetuned ESM2 dir)

    Returns
    -------
    tokenizer, model : (ESM2 tokenizer, MiniFoldMLX)
    """
    if fp16 and bf16:
        raise ValueError("bf16 and fp16 are mutually exclusive")

    from minifoldx.esm2 import ESM2

    print(f"Loading MLX ESM2 from {mlx_esm_path} …")
    tokenizer, esm_model = ESM2.from_pretrained(mlx_esm_path)

    print(f"Loading MiniFold MLX weights from {mlx_minifold_path} …")
    model = MiniFoldMLX.from_pretrained(mlx_minifold_path, esm_model=esm_model)

    if ckpt_path is not None:
        import torch
        from minifold.utils.esm import patch_mlx_esm_finetuned
        print(f"Patching fine-tuned ESM2 layers from {ckpt_path} …")
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", "").replace("model.", ""): v
              for k, v in raw_ckpt["state_dict"].items()}
        esm_raw = object.__getattribute__(model, "_esm_model")
        patch_mlx_esm_finetuned(esm_raw, sd)

    if fp16:
        model.convert_to_fp16()
    elif bf16:
        model.convert_to_bf16()

    if clamp_val > 0:
        for block in model.fold.miniformer.blocks:
            block.clamp_val = clamp_val

    if compile:
        model.enable_compile()

    return tokenizer, model


# ── Single-sequence prediction ────────────────────────────────────────────────

def predict_sequence(
    seq_id: str,
    seq: str,
    model: MiniFoldMLX,
    tokenizer,
    *,
    num_recycling: int = 0,
    max_seq_len: int = 800,
    early_stop: Optional[float] = None,
) -> Optional[str]:
    """Predict a single sequence and return a PDB string (or None if skipped).

    Parameters
    ----------
    seq_id        : sequence identifier
    seq           : amino-acid string
    model         : MiniFoldMLX model (from load_model)
    tokenizer     : ESM2 tokenizer (from load_model)
    num_recycling : recycling iterations (default 0 → 1 forward pass)
    max_seq_len   : skip sequences longer than this (0 = no limit)
    """
    if max_seq_len > 0 and len(seq) > max_seq_len:
        print(f"  [{seq_id}] SKIP: len={len(seq)} > max_seq_len={max_seq_len}")
        return None

    pad_id = int(getattr(tokenizer, "padding_idx", 1))

    tokens, mask_np, aatype = prepare_input(seq, tokenizer)

    tokens_mx = mx.array(tokens)[None]          # (1, L)
    mask_mx   = mx.array(mask_np)[None]         # (1, L)
    aatype_mx = mx.array(aatype)[None]          # (1, L)

    mx.eval(tokens_mx, mask_mx, aatype_mx)
    mx.metal.clear_cache()

    output = model(
        tokens_mx,
        seq_mask      = mask_mx,
        aatype_mx     = aatype_mx,
        num_recycling = num_recycling,
        early_stop    = early_stop,
    )

    L = len(seq)
    return output_to_pdb(
        seq,
        output["final_atom_positions"][0, :L],
        output["final_atom_mask"][0, :L],
        output["plddt"][0, :L],
    )


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(
    sequences: list[tuple[str, str]],
    model: MiniFoldMLX,
    tokenizer,
    *,
    num_recycling: int = 0,
    batch_tokens: int = 2048,
    max_seq_len: int = 800,
    early_stop: Optional[float] = None,
) -> dict[str, Optional[str]]:
    """Predict a list of sequences and return {seq_id: pdb_str}.

    Sequences that exceed max_seq_len map to None.

    Parameters
    ----------
    sequences     : list of (seq_id, seq_str)
    model         : MiniFoldMLX model (from load_model)
    tokenizer     : ESM2 tokenizer (from load_model)
    num_recycling : recycling iterations (default 0)
    batch_tokens  : quadratic memory budget (max_len² per batch)
    max_seq_len   : skip sequences longer than this (0 = no limit)

    Returns
    -------
    dict mapping seq_id → PDB string (or None for skipped sequences)
    """
    pad_id  = int(getattr(tokenizer, "padding_idx", 1))
    results: dict[str, Optional[str]] = {}

    runnable = []
    for seq_id, seq in sequences:
        if max_seq_len > 0 and len(seq) > max_seq_len:
            print(f"  [{seq_id}] SKIP: len={len(seq)} > max_seq_len={max_seq_len}")
            results[seq_id] = None
        else:
            runnable.append((seq_id, seq))

    for batch in create_batches(runnable, batch_tokens):
        seq_ids  = [s[0] for s in batch]
        seq_strs = [s[1] for s in batch]
        print(f"\n=== Batch: {seq_ids}  lengths={[len(s) for s in seq_strs]} ===")

        feats     = [prepare_input(seq, tokenizer) for seq in seq_strs]
        max_len   = max(tok.shape[0] for tok, _, _ in feats)

        tokens_mx = pad_tokens([tok for tok, _, _ in feats], max_len, pad_id)
        mask_mx   = pad_mask([m   for _, m,   _ in feats], max_len)
        aatype_mx = pad_aatype([a for _, _,   a in feats], max_len)

        mx.eval(tokens_mx, mask_mx, aatype_mx)
        mx.metal.clear_cache()

        try:
            output = model(
                tokens_mx,
                seq_mask      = mask_mx,
                aatype_mx     = aatype_mx,
                num_recycling = num_recycling,
                early_stop    = early_stop,
            )

            out_pos  = output["final_atom_positions"]
            out_mask = output["final_atom_mask"]
            plddt    = output["plddt"]

            for idx, (seq_id, seq) in enumerate(zip(seq_ids, seq_strs)):
                L = len(seq)
                results[seq_id] = output_to_pdb(
                    seq,
                    out_pos[idx, :L],
                    out_mask[idx, :L],
                    plddt[idx, :L],
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Batch failed: {e}")
            for seq_id in seq_ids:
                results[seq_id] = None

    return results
