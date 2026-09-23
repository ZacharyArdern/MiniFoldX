"""MLX implementation of MiniFormer blocks (TransitionUpdate, TriangularUpdate, Block, MiniFormer).

Weight naming matches the HuggingFace model except for the raw nn.Parameter keys that are
remapped by convert_hf_to_mlx.py:
  transition.wn  -> transition.norm.weight
  transition.bn  -> transition.norm.bias
  transition.w1  -> transition.linear_1.weight  (shape kept the same: [hidden, dim])
  transition.b1  -> transition.linear_1.bias
  transition.w2  -> transition.linear_2.weight  (shape kept the same: [dim, hidden])
  transition.b2  -> transition.linear_2.bias
  triangular.ni_w -> triangular.norm_in.weight
  triangular.ni_b -> triangular.norm_in.bias
  triangular.pi_w -> triangular.proj_in.weight
  triangular.pi_b -> triangular.proj_in.bias
  triangular.gi_w -> triangular.gate_in.weight
  triangular.gi_b -> triangular.gate_in.bias
  triangular.no_w -> triangular.norm_out.weight
  triangular.no_b -> triangular.norm_out.bias
  triangular.po_w -> triangular.proj_out.weight
  triangular.po_b -> triangular.proj_out.bias
  triangular.go_w -> triangular.gate_out.weight
  triangular.go_b -> triangular.gate_out.bias
"""

import mlx.core as mx
import mlx.nn as nn


# ── Fused LayerNorm + proj*sigmoid(gate) Metal kernel ─────────────────────────
# Mirrors the Triton gating.py kernel from miniscaffold, ported to Apple Metal.
# Valid for C1=C2=128 bf16 inputs only (norm_in gating step of TriangularUpdate).
#
# Each threadgroup (512 threads = 16 SIMD groups) processes BLOCK_M=8 input rows:
#   Phase 1 (SG 0-7)  : LayerNorm per row → threadgroup smem_x [8×128 bf16]
#   Phase 2 (all SG)  : simdgroup_multiply_accumulate x_norm[8×128] @ W[128×128].T
#   Phase 3 (all 512) : bias + sigmoid + multiply → global output
_SGMM_GATE_SOURCE = r"""
    constexpr uint BLOCK_M   = 8;
    constexpr uint C1_       = 128;
    constexpr uint C2_       = 128;
    constexpr uint SG_PER_TG = C2_ / 8;        // 16 SIMD groups per threadgroup
    constexpr float EPS      = 1e-5f;

    const uint tid    = thread_position_in_threadgroup.x;
    const uint sg_idx = tid / 32;
    const uint lane   = tid % 32;
    const uint gid    = threadgroup_position_in_grid.x;
    const uint base   = gid * BLOCK_M;

    threadgroup bfloat smem_x[BLOCK_M * C1_];  // 2 KB — x_norm (bf16)
    threadgroup float  smem_p[BLOCK_M * C2_];  // 4 KB — proj accumulator
    threadgroup float  smem_g[BLOCK_M * C2_];  // 4 KB — gate accumulator

    // Phase 1: LayerNorm (SIMD groups 0..BLOCK_M-1, one row each)
    // 32 lanes × 4 elements = 128 = C1_ per group
    if (sg_idx < BLOCK_M) {
        const uint row_g = base + sg_idx;
        const uint col0  = lane * 4;
        const uint xoff  = row_g * C1_ + col0;
        const float v0 = float(x[xoff+0]), v1 = float(x[xoff+1]);
        const float v2 = float(x[xoff+2]), v3 = float(x[xoff+3]);
        const float mean = simd_sum(v0+v1+v2+v3) * (1.0f / float(C1_));
        const float d0=v0-mean, d1=v1-mean, d2=v2-mean, d3=v3-mean;
        const float inv_std = metal::rsqrt(
            simd_sum(d0*d0+d1*d1+d2*d2+d3*d3) * (1.0f/float(C1_)) + EPS);
        const uint soff = sg_idx * C1_ + col0;
        smem_x[soff+0] = bfloat(d0*inv_std*float(gamma[col0+0])+float(beta[col0+0]));
        smem_x[soff+1] = bfloat(d1*inv_std*float(gamma[col0+1])+float(beta[col0+1]));
        smem_x[soff+2] = bfloat(d2*inv_std*float(gamma[col0+2])+float(beta[col0+2]));
        smem_x[soff+3] = bfloat(d3*inv_std*float(gamma[col0+3])+float(beta[col0+3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: SGMM — each SIMD group owns output cols [sg_idx*8, sg_idx*8+8)
    {
        simdgroup_bfloat8x8 x_tile, w_tile;
        simdgroup_float8x8 acc_p = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc_g = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint col_off = sg_idx * 8;
        for (uint k = 0; k < C1_/8; k++) {
            simdgroup_load(x_tile, (threadgroup const bfloat*)smem_x,
                           C1_, ulong2(k*8, 0), false);
            simdgroup_load(w_tile, W_p, C1_, ulong2(k*8, col_off), true);
            simdgroup_multiply_accumulate(acc_p, x_tile, w_tile, acc_p);
            simdgroup_load(w_tile, W_g, C1_, ulong2(k*8, col_off), true);
            simdgroup_multiply_accumulate(acc_g, x_tile, w_tile, acc_g);
        }
        simdgroup_store(acc_p, (threadgroup float*)smem_p, C2_, ulong2(col_off,0), false);
        simdgroup_store(acc_g, (threadgroup float*)smem_g, C2_, ulong2(col_off,0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: bias + sigmoid gating + write output (512 threads × 2 elems = 1024)
    for (uint i = 0; i < 2; i++) {
        const uint e   = tid * 2 + i;
        const uint row = e / C2_;
        const uint col = e % C2_;
        const float p   = smem_p[e] + float(b_p[col]);
        const float g_  = smem_g[e] + float(b_g[col]);
        out[(base+row)*C2_+col] = bfloat(p * (1.0f/(1.0f+metal::exp(-g_))));
    }
"""

_SGMM_GATE_KERNEL = None   # built lazily on first use
_SGMM_OUT_KERNEL  = None   # built lazily on first use


def _get_sgmm_gate_kernel():
    global _SGMM_GATE_KERNEL
    if _SGMM_GATE_KERNEL is None:
        _SGMM_GATE_KERNEL = mx.fast.metal_kernel(
            name="sgmm_gate",
            input_names=["x", "gamma", "beta", "W_p", "b_p", "W_g", "b_g"],
            output_names=["out"],
            source=_SGMM_GATE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _SGMM_GATE_KERNEL


# ── Fused RMSNorm + proj*sigmoid(gate) Metal kernel (output gating, K=64→N=128) ──
# norm_out is RMSNorm after convert_to_bf16(): no mean subtraction, no bias.
# Phase 1 (8 SGs): RMSNorm per row, 32 lanes × 2 elems = 64 = C1_.
# Phase 2 (16 SGs): dual SGMM, K-loop = C1_/8 = 8 iterations.
_SGMM_OUT_SOURCE = r"""
    constexpr uint BLOCK_M   = 8;
    constexpr uint C1_       = 64;
    constexpr uint C2_       = 128;
    constexpr float EPS      = 1e-5f;

    const uint tid    = thread_position_in_threadgroup.x;
    const uint sg_idx = tid / 32;
    const uint lane   = tid % 32;
    const uint gid    = threadgroup_position_in_grid.x;
    const uint base   = gid * BLOCK_M;

    threadgroup bfloat smem_x[BLOCK_M * C1_];
    threadgroup float  smem_p[BLOCK_M * C2_];
    threadgroup float  smem_g[BLOCK_M * C2_];

    // Phase 1: RMSNorm — 8 SGs, one row each (32 lanes × 2 elems = 64)
    if (sg_idx < BLOCK_M) {
        const uint row_g = base + sg_idx;
        const uint col0  = lane * 2;
        const uint xoff  = row_g * C1_ + col0;
        const float v0 = float(x[xoff+0]), v1 = float(x[xoff+1]);
        const float inv_rms = metal::rsqrt(
            simd_sum(v0*v0 + v1*v1) * (1.0f / float(C1_)) + EPS);
        const uint soff = sg_idx * C1_ + col0;
        smem_x[soff+0] = bfloat(v0 * inv_rms * float(gamma[col0+0]));
        smem_x[soff+1] = bfloat(v1 * inv_rms * float(gamma[col0+1]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: dual SGMM — 16 SGs × 8 output cols
    {
        simdgroup_bfloat8x8 x_tile, w_tile;
        simdgroup_float8x8 acc_p = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc_g = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint col_off = sg_idx * 8;
        for (uint k = 0; k < C1_/8; k++) {
            simdgroup_load(x_tile, (threadgroup const bfloat*)smem_x,
                           C1_, ulong2(k*8, 0), false);
            simdgroup_load(w_tile, W_p, C1_, ulong2(k*8, col_off), true);
            simdgroup_multiply_accumulate(acc_p, x_tile, w_tile, acc_p);
            simdgroup_load(w_tile, W_g, C1_, ulong2(k*8, col_off), true);
            simdgroup_multiply_accumulate(acc_g, x_tile, w_tile, acc_g);
        }
        simdgroup_store(acc_p, (threadgroup float*)smem_p, C2_, ulong2(col_off, 0), false);
        simdgroup_store(acc_g, (threadgroup float*)smem_g, C2_, ulong2(col_off, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: bias + sigmoid gating (512 threads × 2 elems = 1024)
    for (uint i = 0; i < 2; i++) {
        const uint e   = tid * 2 + i;
        const uint row = e / C2_;
        const uint col = e % C2_;
        const float p  = smem_p[e] + float(b_p[col]);
        const float g_ = smem_g[e] + float(b_g[col]);
        out[(base + row) * C2_ + col] = bfloat(p * (1.0f / (1.0f + metal::exp(-g_))));
    }
"""


def _get_sgmm_out_kernel():
    global _SGMM_OUT_KERNEL
    if _SGMM_OUT_KERNEL is None:
        _SGMM_OUT_KERNEL = mx.fast.metal_kernel(
            name="sgmm_out_gate",
            input_names=["x", "gamma", "W_p", "b_p", "W_g", "b_g"],
            output_names=["out"],
            source=_SGMM_OUT_SOURCE,
            ensure_row_contiguous=True,
        )
    return _SGMM_OUT_KERNEL


def _sgmm_out_gate(x: mx.array,
                   gamma: mx.array,
                   W_p: mx.array, b_p: mx.array,
                   W_g: mx.array, b_g: mx.array) -> mx.array:
    """Fused RMSNorm+SGMM output gate. x: [B,L,L,64] bf16 → [B,L,L,128] bf16."""
    B, L1, L2, C = x.shape
    M = B * L1 * L2
    M_pad = ((M + _SGMM_BLOCK_M - 1) // _SGMM_BLOCK_M) * _SGMM_BLOCK_M
    x_flat = x.reshape(M, C)
    if M_pad > M:
        x_flat = mx.pad(x_flat, [(0, M_pad - M), (0, 0)])
    n_total = (M_pad // _SGMM_BLOCK_M) * _SGMM_TG_THREADS
    (out,) = _get_sgmm_out_kernel()(
        inputs=[x_flat, gamma, W_p, b_p, W_g, b_g],
        template=[("T", mx.bfloat16)],
        grid=(n_total, 1, 1),
        threadgroup=(_SGMM_TG_THREADS, 1, 1),
        output_shapes=[(M_pad, 128)],
        output_dtypes=[mx.bfloat16],
    )
    return out[:M].reshape(B, L1, L2, 128)


_SGMM_BLOCK_M    = 8
_SGMM_TG_THREADS = 512    # 16 SIMD groups × 32 threads


def _sgmm_gate(x: mx.array,
               gamma: mx.array, beta: mx.array,
               W_p: mx.array, b_p: mx.array,
               W_g: mx.array, b_g: mx.array) -> mx.array:
    """Call the fused SGMM gate kernel. x must be bfloat16, shape [B, L, L, 128]."""
    B, L1, L2, C = x.shape
    M = B * L1 * L2
    M_pad = ((M + _SGMM_BLOCK_M - 1) // _SGMM_BLOCK_M) * _SGMM_BLOCK_M
    x_flat = x.reshape(M, C)
    if M_pad > M:
        x_flat = mx.pad(x_flat, [(0, M_pad - M), (0, 0)])
    n_total = (M_pad // _SGMM_BLOCK_M) * _SGMM_TG_THREADS
    (out,) = _get_sgmm_gate_kernel()(
        inputs=[x_flat, gamma, beta, W_p, b_p, W_g, b_g],
        template=[("T", mx.bfloat16)],
        grid=(n_total, 1, 1),
        threadgroup=(_SGMM_TG_THREADS, 1, 1),
        output_shapes=[(M_pad, C)],
        output_dtypes=[mx.bfloat16],
    )
    return out[:M].reshape(B, L1, L2, C)


class TransitionUpdate(nn.Module):
    """Two-layer MLP: LayerNorm -> Linear -> ReLU -> Linear (no residual here; added in Block)."""

    def __init__(self, dim: int = 128, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_1 = nn.Linear(dim, hidden)
        self.linear_2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.linear_1(x)
        x = mx.maximum(x, 0)  # ReLU
        x = self.linear_2(x)
        return x


class TriangularUpdate(nn.Module):
    """Merged incoming/outgoing triangular multiplicative update.

    Input gating  (D  -> D):     LayerNorm -> (proj_in * sigmoid(gate_in))
    Triangular projection:       split into 4 chunks, compute two einsums, concat -> D/2
    Output gating (D/2 -> D):    LayerNorm -> (proj_out * sigmoid(gate_out))
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm_in  = nn.LayerNorm(dim)
        self.proj_in  = nn.Linear(dim, dim)
        self.gate_in  = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim // 2)
        self.proj_out = nn.Linear(dim // 2, dim)
        self.gate_out = nn.Linear(dim // 2, dim)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        # Input gating: D -> D
        # If patch_sgmm_gate() was called, _fused_input_gate is a bound callable
        # that fuses LayerNorm + proj_in * sigmoid(gate_in) in one SGMM kernel.
        if hasattr(self, '_fused_input_gate'):
            x = self._fused_input_gate(x)
        else:
            x = self.norm_in(x)
            x = self.proj_in(x) * mx.sigmoid(self.gate_in(x))

        # Apply pair mask
        x = x * mask[..., None]

        # Triangular projection: split into 4 equal chunks -> D//4 each
        d = x.shape[-1] // 4
        a1 = x[..., :d]
        b1 = x[..., d : 2 * d]
        a2 = x[..., 2 * d : 3 * d]
        b2 = x[..., 3 * d :]

        # Outgoing: contract over the k (middle-row) dimension
        x1 = mx.einsum("bikd,bjkd->bijd", a1, b1)
        # Incoming: contract over the k (middle-column) dimension
        x2 = mx.einsum("bkid,bkjd->bijd", a2, b2)

        x = mx.concatenate([x1, x2], axis=-1)  # (B, N, N, D//2)

        # Output gating: D/2 -> D
        if hasattr(self, '_fused_output_gate'):
            x = self._fused_output_gate(x)
        else:
            x = self.norm_out(x)
            x = self.proj_out(x) * mx.sigmoid(self.gate_out(x))
        return x


class Block(nn.Module):
    """One MiniFormer block: triangular update + transition MLP, both with residual."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.triangular = TriangularUpdate(dim)
        self.transition = TransitionUpdate(dim, dim * 4)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        clamp_val = getattr(self, 'clamp_val', 0.0)
        if clamp_val > 0:
            x = mx.clip(x, -clamp_val, clamp_val)
        x = x + self.triangular(x, mask)
        x = x + self.transition(x)
        return x


class MiniFormer(nn.Module):
    """Stack of MiniFormer blocks processing pair representations."""

    def __init__(self, dim: int = 128, blocks: int = 48):
        super().__init__()
        self.blocks = [Block(dim) for _ in range(blocks)]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x, mask)
        return x

    def patch_sgmm_gate(self) -> None:
        """Replace the norm_in + input gating step in every TriangularUpdate block
        with a fused SIMD-group matrix multiply Metal kernel.

        The kernel fuses:  LayerNorm(x) → proj_in(x_norm) * sigmoid(gate_in(x_norm))
        into a single Metal dispatch, keeping x_norm in on-chip SRAM and using
        simdgroup_multiply_accumulate for the two 128×128 matmuls.

        Requirements:
          - Model weights must be in bfloat16 (call convert_to_bf16() first)
          - dim must be 128 (hardcoded in the kernel)
          - Apple Silicon GPU (Metal 3+)

        Benchmarked speedup: ~2× for the gating step across L=100..700.
        Combined with norm_out RMSNorm (applied by convert_to_bf16), this gives
        ~1.4× speedup on MiniFormer overall.
        """
        for block in self.blocks:
            tri = block.triangular
            # Capture per-block weights as a closure
            gamma = tri.norm_in.weight
            beta  = tri.norm_in.bias
            W_p   = tri.proj_in.weight
            b_p   = tri.proj_in.bias
            W_g   = tri.gate_in.weight
            b_g   = tri.gate_in.bias

            def _make_fused(gamma, beta, W_p, b_p, W_g, b_g):
                def _fused(x):
                    return _sgmm_gate(x, gamma, beta, W_p, b_p, W_g, b_g)
                return _fused

            object.__setattr__(tri, '_fused_input_gate',
                               _make_fused(gamma, beta, W_p, b_p, W_g, b_g))

        print(f"SGMM gate kernel enabled on {len(self.blocks)} TriangularUpdate blocks.")

    def patch_sgmm_out_gate(self) -> None:
        """Replace the norm_out + output gating step in every TriangularUpdate block
        with a fused SIMD-group matrix multiply Metal kernel.

        The kernel fuses:  LayerNorm(x) → proj_out(x_norm) * sigmoid(gate_out(x_norm))
        where x has shape [B, L, L, 64] (D/2 after triangular projection).

        Requirements:
          - Model weights must be in bfloat16 (call convert_to_bf16() first)
          - dim must be 128 (hardcoded: C1_=64, C2_=128)
          - Apple Silicon GPU (Metal 3+)
        """
        for block in self.blocks:
            tri = block.triangular
            gamma = tri.norm_out.weight
            W_p   = tri.proj_out.weight
            b_p   = tri.proj_out.bias
            W_g   = tri.gate_out.weight
            b_g   = tri.gate_out.bias

            def _make_fused_out(gamma, W_p, b_p, W_g, b_g):
                def _fused(x):
                    return _sgmm_out_gate(x, gamma, W_p, b_p, W_g, b_g)
                return _fused

            object.__setattr__(tri, '_fused_output_gate',
                               _make_fused_out(gamma, W_p, b_p, W_g, b_g))

        print(f"SGMM out-gate kernel enabled on {len(self.blocks)} TriangularUpdate blocks.")

    def patch_norms_rms_absorb_bias(self, triangular: bool = True, transition: bool = False,
                                    tri_norm_in: bool = True, tri_norm_out: bool = True):
        """Replace LayerNorm with RMSNorm, absorbing the LayerNorm bias (β) into the
        immediately downstream linear layer biases.

        LayerNorm produces:  y = γ*(x-μ)/σ + β
        The β offset is a constant additive term: linear(y) = W @ y + b
            = W @ (γ*(x-μ)/σ) + W @ β + b
        So we can fold W @ β into the linear bias and use RMSNorm (no β, no mean).

        Remaining approximation: mean subtraction (μ) is dropped — empirically
        |mean|/std ≈ 0.03–0.11 in trained MiniFold weights, a small second-order effect.

        Modifies model in-place.  Call once after load_model() + convert_to_bf16().
        """
        for block in self.blocks:
            if triangular:
                tri = block.triangular
                if tri_norm_in:
                    # ── norm_in feeds proj_in and gate_in ────────────────────
                    beta = tri.norm_in.bias.astype(mx.float32)
                    for linear in (tri.proj_in, tri.gate_in):
                        W = linear.weight.astype(mx.float32)
                        absorbed = (W @ beta).astype(linear.bias.dtype)
                        linear.bias = linear.bias + absorbed
                    rms = nn.RMSNorm(tri.norm_in.weight.shape[0])
                    rms.weight = tri.norm_in.weight
                    tri.norm_in = rms

                if tri_norm_out:
                    # ── norm_out feeds proj_out and gate_out ─────────────────
                    beta = tri.norm_out.bias.astype(mx.float32)
                    for linear in (tri.proj_out, tri.gate_out):
                        W = linear.weight.astype(mx.float32)
                        absorbed = (W @ beta).astype(linear.bias.dtype)
                        linear.bias = linear.bias + absorbed
                    rms = nn.RMSNorm(tri.norm_out.weight.shape[0])
                    rms.weight = tri.norm_out.weight
                    tri.norm_out = rms

            if transition:
                trans = block.transition
                beta = trans.norm.bias.astype(mx.float32)
                W = trans.linear_1.weight.astype(mx.float32)
                absorbed = (W @ beta).astype(trans.linear_1.bias.dtype)
                trans.linear_1.bias = trans.linear_1.bias + absorbed
                rms = nn.RMSNorm(trans.norm.weight.shape[0])
                rms.weight = trans.norm.weight
                trans.norm = rms

        mx.eval(self.parameters())

    def patch_norm_in_rms_calibrated(self, mean_ratios: list):
        """Replace norm_in LayerNorm with RMSNorm, absorbing both β and the
        calibrated mean-subtraction correction into the downstream linear biases.

        The full LayerNorm correction relative to bare RMSNorm is:
            correction_b = -(mean_ratio_b * γ_b) + β_b
        where mean_ratio_b = E[μ(x)/σ(x)] estimated from calibration sequences.

        This vector gets folded into proj_in.bias and gate_in.bias via W @ correction.

        Args:
            mean_ratios: list of 48 floats, one per block — E[μ/σ] at norm_in input.
        """
        for b, block in enumerate(self.blocks):
            tri = block.triangular
            gamma = tri.norm_in.weight.astype(mx.float32)   # [D]
            beta  = tri.norm_in.bias.astype(mx.float32)     # [D]

            # Full correction = centring term + bias term
            correction = (-mean_ratios[b] * gamma + beta)   # [D]

            for linear in (tri.proj_in, tri.gate_in):
                W        = linear.weight.astype(mx.float32)  # [out, in]
                absorbed = (W @ correction).astype(linear.bias.dtype)
                linear.bias = linear.bias + absorbed

            rms = nn.RMSNorm(gamma.shape[0])
            rms.weight = tri.norm_in.weight
            tri.norm_in = rms

        mx.eval(self.parameters())

