---
problem: muse_bf16_proj
slug: pooler-skinny-tiled-weight-layout
created: 2026-10-23T225303Z
base: master
area: pooler GEMM
change: >
  Lossless weight repack for the parent's skinny kernel: in prepare(), for the pooler weight only,
  build state['pooler_tiled'] = W.view(N/BLOCK_N, BLOCK_N, K).transpose(1, 2).contiguous(), i.e. a
  [N/32, K, 32] bf16 tensor where program i's whole working set is one contiguous 512 KB block
  laid out K-major, with the same values and dtype (bf16 stays bf16, nothing rounded). In the
  Triton kernel from pooler-triton-skinny-m32, change only the weight tl.load addressing: the tile
  for K-step j is the contiguous [BLOCK_K, BLOCK_N] slab at base + i*K*BLOCK_N +
  j*BLOCK_K*BLOCK_N, loaded directly in the [BLOCK_K, BLOCK_N] orientation tl.dot(x, w) wants, so
  no in-register transpose and every warp's loads are 128-byte-aligned fully sequential lines
  instead of 32 strided row segments of 512 bytes. Everything else (grid, BLOCK sizes, num_stages,
  x path, store) stays exactly as the parent. The original W is kept in state for m >= 128
  (cuBLAS) so memory grows by 67 MB once, untimed.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: same bytes, strictly sequential per-program stream and no transpose; a few percent of the parent if the parent is memory-pipeline bound
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: same
  - shapes: layer=pooler load=decode_32 *
    direction: faster
    why: same
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: same at m=8
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 reads the original W via cuBLAS; control
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: untouched; control
risk: low
depends_on: pooler-triton-skinny-m32
priority: 4
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-24T040515Z
review_companions: null
hold_reason: "skinny-Triton family settled: pooler neutral, qkv-gate regression, unpooler-full neutral, o neutral (1.006x), and the BLOCK_M=16 control showed the m=32 penalty is not the tile (20.8 us vs 18.4 us in-run at m=16). The reviewer expected both of this hypothesis's mechanisms to be ~zero, and it builds on the unpromoted pooler branch, so a gain could not be promoted without a rebase and rerun."
---

# pooler-skinny-tiled-weight-layout

## Change

Lossless weight repack for the parent's skinny kernel: in prepare(), for the pooler weight only, build state['pooler_tiled'] = W.view(N/BLOCK_N, BLOCK_N, K).transpose(1, 2).contiguous(), i.e. a [N/32, K, 32] bf16 tensor where program i's whole working set is one contiguous 512 KB block laid out K-major, with the same values and dtype (bf16 stays bf16, nothing rounded). In the Triton kernel from pooler-triton-skinny-m32, change only the weight tl.load addressing: the tile for K-step j is the contiguous [BLOCK_K, BLOCK_N] slab at base + i*K*BLOCK_N + j*BLOCK_K*BLOCK_N, loaded directly in the [BLOCK_K, BLOCK_N] orientation tl.dot(x, w) wants, so no in-register transpose and every warp's loads are 128-byte-aligned fully sequential lines instead of 32 strided row segments of 512 bytes. Everything else (grid, BLOCK sizes, num_stages, x path, store) stays exactly as the parent. The original W is kept in state for m >= 128 (cuBLAS) so memory grows by 67 MB once, untimed.

## Expected

- `layer=pooler load=decode_1 *` faster: same bytes, strictly sequential per-program stream and no transpose; a few percent of the parent if the parent is memory-pipeline bound
- `layer=pooler load=decode_16 *` faster: same
- `layer=pooler load=decode_32 *` faster: same
- `layer=pooler load=verify_1 *` faster: same at m=8
- `layer=pooler load=verify_16 *` flat: m=128 reads the original W via cuBLAS; control
- `*layer=qkv_gate*` flat: untouched; control

## Risk

Pure layout tune on the parent, so it attributes to addressing alone; if it is flat the parent's limit is elsewhere (occupancy or latency) and the tiled layout should not be carried to the sibling layers. Rules: repacking weights in prepare() is explicitly allowed and this is a permutation of the same bf16 values (lossless). The repack must use the same BLOCK_N the kernel is launched with; a mismatch is a wrong answer, caught by the m=7 pooler test. Risk is low but so is the expected size of the effect; if the parent lands at >= 4.3 TB/s skip this.

## Adversarial review

Reviewed 2026-10-24T040515Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A cheap, lossless, fully attributable layout probe on the parent's pooler kernel: only the four pooler m<=32 rows can move and nothing landed touches them, so one run is interpretable without companions. Its two stated mechanisms are each close to zero on this hardware, so the honest expectation is flat (within the run's CI) with the readout being the _pooler_skinny_kernel us per row against the parent's 20.2/20.2/20.5/25.2 us, not the headline speedup.

### Justification
Shapes and weights are consistent with TASK.md (every shape weighs 62; pooler is 67.1 MB bf16, floor 14.0-14.1 us at 4.8 TB/s; the four 'faster' globs are exactly the parent's Triton rows, verify_16 is a correct cuBLAS control; pooler verify_32 (m=256, cuBLAS) is missing and should be added as flat). The profile data say the parent's kernel is NOT addressing-bound: pooler-triton-skinny-m32.result.json shows _pooler_skinny_kernel at 20.2 us (m=1, 3.32 TB/s), 20.2 us (m=8), 20.5 us (m=16, 3.27 TB/s) and 25.2 us (m=32, 2.66 TB/s) against nvjet's main kernel at 18.3-18.9/19.3/20.2 us (3.3-3.7 TB/s); the parent's per-row gains at m<=16 (1.057-1.063x) came entirely from dropping the 1.6-1.9 us splitKreduce launch and its graph gap, and it lost 11% at m=32 with byte-identical weight traffic. That m-dependence (20.5 -> 25.2 us when only BLOCK_M and the x tile change) points at the x re-stream from L2 (128 programs x 32 steps x 16 KB = 64 MB of L2 reads at m=32), smem per stage (24/32 KB -> 96/128 KB at 4 stages) and bytes in flight (~3 stages ahead x 16 KB x 128 programs = ~6 MB, right at Little's law for 4.8 TB/s at ~1.3-1.5 us loaded latency), none of which the tiled layout touches. Weakest link: both claimed mechanisms are near-zero. (1) The parent's [32, 256] tile is already fully coalesced: each 512 B row segment is four complete 128 B lines at a 128 B-aligned offset (k offsets are multiples of 512 B, row stride 16 KB), and the program's 512 KB working set is ALREADY one contiguous block in the row-major layout (rows n0..n0+31 are adjacent), so the repack only reorders accesses inside that block; HBM3e channel/bank interleaving at sub-KB granularity treats 32 x 512 B strips and one 16 KB run about equally, so the DRAM-page-locality gain is at most a fraction of a percent. (2) The 'in-register transpose' it removes does not exist as a cost: at BLOCK_M in {16, 32} Triton cannot emit wgmma (minimum M = 64) and lowers tl.dot to mma.sync with ldmatrix, where ldmatrix.trans supplies the transposed bf16 B operand for free; even on the wgmma path the smem descriptor supports MN-major bf16 B. Master's landed router kernel already reads [BLOCK_K, BLOCK_N] tiles from K-contiguous storage with strided pointers and is fast, which is the same orientation trick without a repack. Magnitude: the hypothesis's own 'a few percent of 20 us' on four rows that are ~28-36% of four loads is <= 1% per load, inside the judge's +-1.5% CI, so the headline will read flat either way; the informative number is kernel us per row, whose cross-run resolution is ~0.2-0.4 us. The decode_32 'faster' is the least defensible row: the parent's 4.7 us deficit there is in the x path / smem, not the weight stream.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (landed 965b046, and the parent's base e64f27e): no confound; pooler m<=32 is replaced wholesale by the parent's Triton kernel and pooler m>=128 is the same _bf16_gemm_dispatch_impl call on both sides, so this change behaves identically with or without it.
- router-hilo-at-m8 (landed f9dc05d): no confound on the pooler rows (router only, each shape timed in its own CUDA graph), but it is ABSENT from the parent branch this will be built on (hyp/muse_bf16_proj/pooler-triton-skinny-m32, impl d2840d0, base e64f27e), so the run's router verify_1 row and headline must be read against the parent's result.json (0.998x), never against BEST.
- router-triton-hilo-fused-control-splitk-all-m (landed dba4ffd, BEST 1.151x): same as above, absent from the build base; its 3-D split-K Triton router is the reason the headline of this run will sit near 1.00 rather than 1.15. It does not interact with the pooler kernel (different weight, different graph), but a gain here cannot be promoted as-is: the merged master+parent+this code would be a never-run submission needing its own run.

### Interference and A/B needs
No companion is warranted. The change is a single sub-change (contiguous K-major 32-column tiles, with the [BLOCK_K, BLOCK_N] load orientation that follows from it) in one kernel plus an untimed repack; the only rows that can move are the four pooler m<=32 rows, each captured and timed in its own graph with the one Triton launch and a torch.empty. A gain is unambiguously the layout (nothing else differs from the parent's code), and a loss is equally attributable (the tiled 64 B-row tile may give Triton a worse smem swizzle / ldmatrix pattern than the 512 B-row tile, which is itself the finding). ab-revert is inapplicable: nothing landed touches this path, and the two router hypotheses on master are not even on the parent branch. ab-tune is inapplicable: no landed tunable reaches the pooler. A control splitting contiguity from orientation (tiles [N/32, K/256, 32, 256] so each K-step is still one contiguous 16 KB run but in the parent's [N, K] orientation with tl.trans) would only matter if this LOSES and someone wants to carry the contiguous layout to the siblings anyway; that is a follow-on retune, not a disambiguator, and I would not spend a run on it. The parent's real axes (BLOCK_M=16 dual-accumulator at m=32, num_stages/BLOCK_K for bytes in flight, BLOCK_N=16 for 2 CTAs/SM) are separate hypotheses, one of which (unpooler-full-triton-skinny-m32-control-blockm16-all-m) is already queued on the unpooler sibling.

### Open questions
- Define the readout before the run: the layout effect is real only if _pooler_skinny_kernel drops by >= 0.5 us at all of m=1, 8, 16 (from 20.2/20.2/20.5 us); cross-run kernel-us noise is ~0.2-0.4 us and the headline speedup will be inside its CI either way.
- Reframe the decode_32 expectation: the parent is 4.7 us slower than its own m=16 kernel with identical weight bytes, which is an x-path / smem effect the layout cannot fix; 'flat' there is the honest prediction, and a drop there would be evidence of something other than the stated mechanism.
- Add pooler verify_32 (m=256, cuBLAS) as a flat control so the attribution grid is complete; qkv_gate stays cuBLAS on the pooler branch (the in-flight qkv-gate-triton-skinny-m32 is a sibling branch, not an ancestor), so that control is valid.
- Promotion path: a gain cannot be merged to master directly because the parent is unpromoted and master now carries two router changes absent from this branch; the merged code would need its own run. Weigh that against the <= 1%-per-load ceiling of this probe when deciding whether it is worth today's slot versus a direct fix of the parent's m=32 deficit.
- If flat, record in FINDINGS that weight-tile addressing is not the plain Triton skinny stream's limit on H200 (it was already line-aligned and block-contiguous), so the siblings should not adopt a repack; the next axes are bytes in flight (num_stages=6 or BLOCK_K=512), 2 CTAs/SM (BLOCK_N=16, 256 programs) and the x re-stream (BLOCK_M=16 dual accumulator at m=32).

