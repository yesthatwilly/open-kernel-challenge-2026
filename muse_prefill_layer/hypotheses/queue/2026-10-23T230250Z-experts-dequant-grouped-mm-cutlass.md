---
problem: muse_prefill_layer
slug: experts-dequant-grouped-mm-cutlass
created: 2026-10-23T230250Z
base: master
area: "w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton)"
change: >
  Keep the per-call dequant and the gate of experts-dequant-triton-moe-16k and replace its two
  invoke_fused_moe_kernel calls with torch._grouped_mm (the CUTLASS sm90 BF16 grouped GEMM) on an
  expert-major row layout built with fixed-size torch ops only: ids = local_ids.view(-1) with
  non-local slots (-1) mapped to 32; order = torch.sort(ids, stable=True).indices; counts =
  zeros(33, int32).index_add_(0, ids, ones); padded = (counts[:32] + 127) // 128 * 128; offs =
  cumsum(padded).to(int32) [32]; unpadded_start and padded_start by exclusive cumsum; dest[j] for
  the j-th sorted slot = padded_start[ids_sorted[j]] + j - unpadded_start[ids_sorted[j]] for local
  slots and M_fixed - 1 (a dump row) for the non-local ones, with M_fixed = T*8 + 32*128 + 1 so
  the dump row and every padding row lie beyond offs[31]. A1 = torch.empty([M_fixed, 4096], bf16);
  A1.index_copy_(0, dest, x[order // 8]) (one gather, one index_copy; padding rows are never read
  by a group because each group spans exactly its padded count, and the dump row is beyond all
  groups). gemm1: torch._grouped_mm(A1, w13_bf16.transpose(1, 2), offs=offs) -> [M_fixed, 8192];
  the parity silu_and_mul on the whole buffer without expert_ids -> h [M_fixed, 4096]; gemm2:
  torch._grouped_mm(h, w2_bf16.transpose(1, 2), offs=offs) -> out [M_fixed, 4096]; y = out[dest]
  [T*8, 4096] gathered back into slot order (non-local slots read the dump row, which
  fused_post_expert_norm_weighted_sum skips via slot_expert_ids=-1, as it skips the torch.empty
  rows in the Marlin path); then the parity weighted sum on y.view(T, 8, 4096). In prepare() run
  one tiny untimed torch._grouped_mm (2D x 3D with offs) to confirm the op exists and accepts offs
  whose last entry is below A's row count; if it does not exist, raise at import so the run fails
  fast instead of silently falling back.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: CUTLASS sm90 grouped GEMM (warp-specialized, TMA, 128x128 or 128x256 tiles) typically beats the Triton fused_moe kernel by 15-30% on 1.65 TFLOP of BF16 work, which is ~0.5-1 ms per shape, against ~130 MB of extra gather/scatter traffic (~60 us) and the sort/cumsum chain (~30 us)
  - shapes: "*load=turn_1*"
    direction: flat
    why: below the gate (control)
  - shapes: "*load=turn_4*"
    direction: flat
    why: below the gate (control)
risk: high
depends_on: experts-dequant-triton-moe-16k
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# experts-dequant-grouped-mm-cutlass

## Change

Keep the per-call dequant and the gate of experts-dequant-triton-moe-16k and replace its two invoke_fused_moe_kernel calls with torch._grouped_mm (the CUTLASS sm90 BF16 grouped GEMM) on an expert-major row layout built with fixed-size torch ops only: ids = local_ids.view(-1) with non-local slots (-1) mapped to 32; order = torch.sort(ids, stable=True).indices; counts = zeros(33, int32).index_add_(0, ids, ones); padded = (counts[:32] + 127) // 128 * 128; offs = cumsum(padded).to(int32) [32]; unpadded_start and padded_start by exclusive cumsum; dest[j] for the j-th sorted slot = padded_start[ids_sorted[j]] + j - unpadded_start[ids_sorted[j]] for local slots and M_fixed - 1 (a dump row) for the non-local ones, with M_fixed = T*8 + 32*128 + 1 so the dump row and every padding row lie beyond offs[31]. A1 = torch.empty([M_fixed, 4096], bf16); A1.index_copy_(0, dest, x[order // 8]) (one gather, one index_copy; padding rows are never read by a group because each group spans exactly its padded count, and the dump row is beyond all groups). gemm1: torch._grouped_mm(A1, w13_bf16.transpose(1, 2), offs=offs) -> [M_fixed, 8192]; the parity silu_and_mul on the whole buffer without expert_ids -> h [M_fixed, 4096]; gemm2: torch._grouped_mm(h, w2_bf16.transpose(1, 2), offs=offs) -> out [M_fixed, 4096]; y = out[dest] [T*8, 4096] gathered back into slot order (non-local slots read the dump row, which fused_post_expert_norm_weighted_sum skips via slot_expert_ids=-1, as it skips the torch.empty rows in the Marlin path); then the parity weighted sum on y.view(T, 8, 4096). In prepare() run one tiny untimed torch._grouped_mm (2D x 3D with offs) to confirm the op exists and accepts offs whose last entry is below A's row count; if it does not exist, raise at import so the run fails fast instead of silently falling back.

## Expected

- `*load=first_turn*` faster: CUTLASS sm90 grouped GEMM (warp-specialized, TMA, 128x128 or 128x256 tiles) typically beats the Triton fused_moe kernel by 15-30% on 1.65 TFLOP of BF16 work, which is ~0.5-1 ms per shape, against ~130 MB of extra gather/scatter traffic (~60 us) and the sort/cumsum chain (~30 us)
- `*load=turn_1*` flat: below the gate (control)
- `*load=turn_4*` flat: below the gate (control)

## Risk

High risk on API: torch._grouped_mm 2D x 3D with offs needs torch >= 2.8 in the judge image and may require offs[-1] == A.shape[0] (if so the hypothesis is dead; do not pad expert 31 to cover the tail, that would compute the non-local rows); group sizes are multiples of 128, satisfying its alignment. Correctness: the dump row must be beyond offs[31] (it is by construction) and index_copy_ with repeated dump-row indices is a benign race on a row nobody reads. Rules: same as the parent (BF16 copy inside the timed call only, BF16 activations, fp32 accumulate). Attribution: only the GEMM engine and its layout glue change against the parent.
