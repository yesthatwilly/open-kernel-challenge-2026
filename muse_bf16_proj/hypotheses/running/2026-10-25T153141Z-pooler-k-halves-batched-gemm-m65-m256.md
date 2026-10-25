---
problem: muse_bf16_proj
slug: pooler-k-halves-batched-gemm-m65-m256
created: 2026-10-25T153141Z
base: master
area: pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only
  (scored rows verify_16 m=128 and verify_32 m=256; no test uses this band), replace master's path
  with a user-level split-K of 2. Master's path is torch.mm on the NN weight at m=65..128 (nvjet
  128x64 splitK 20.2 us + cublasLt splitKreduce 3.0 us) and _bf16_gemm_dispatch_impl at m=256
  (nvjet 128x64 TNT 30.4 us). WHY: pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) do the
  same FLOPs and bytes, but cuBLAS runs the long-K, narrow-N pooler worse. At m=256 the pooler
  takes 31.3 us (128x64 tiles, about 550 TFLOP/s) while the unpooler takes 24.5 us (128x128
  tiles); at m=128 it is 22.9 vs 18.9 us. Two K-halves of the pooler have the unpooler's tile
  count: n 4096 x k 4096 x 2 batches gives the same 128 CTAs of 128x128 (m=256) or 128x64 (m=128)
  tiles, each over k=4096, as the unpooler's one GEMM. CHANGE: xh = x.view(m, 2, 4096).permute(1,
  0, 2), a strided view [2, m, 4096] with strides (4096, 8192, 1) and no copy. For the weight,
  either candidate T, a TN view of state['pooler'] as [2, 4096, 4096] with strides (4096, 8192, 1)
  used transposed, or candidate N, the existing NN weight state['pooler_t'].view(2, 4096, 4096)
  (contiguous halves, no new memory). Then p = torch.bmm(xh, w_halves) into [2, m, 4096] bf16, and
  out = torch.add(p[0], p[1]): two graph nodes, with the add reading 4 MB and writing 2 MB at
  m=256. Precision: each half rounds once to bf16 (about 0.2% per element, L2 well under 1%). The
  single-nonzero rows stay exact, because the other half is exactly 0. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The
  profiler assertion must show exactly one nvjet/cutlass GEMM kernel plus one elementwise add, and
  no copy kernel; if torch inserts a contiguous() copy of xh, print it and keep master. Every
  other layer and m stays byte-identical.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: nvjet 128x64 30.4 us / row 31.33, compute floor 17.37; batched halves should match the unpooler m=256 nvjet 128x128 (24.8 kernel) + add ~1.2 -> row ~26
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: NN splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; halves should match unpooler m=128 (row 18.95) + add ~1.0 -> row ~20
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: m <= 32 untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T161612Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T161612Z
base_sha: e2a24e6c8d5b0c6b1ec53f88d2d22e18c50aa1ae
base_blob: f919748b5af6e815f319e5d2d828ed68e5b28a71
branch: hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256
worktree: .worktrees/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T162132Z
impl_commit: e5de059ed5d1ec059df17caeab66fcd66e95804f
submission_sha256: 8757f255f0ba42c534b903ef01b065cf9196e6e4ef7260c292324799801405d0
norm_sha256: 09e289eec8075e0271d86f9b68fb56431f86d1fcaa7ae97f4e290f45d87042e8
submission_blob: 37b0e68c31722a46fd7377aaca28eba9d5955c68
lint_warnings: 21
---

# pooler-k-halves-batched-gemm-m65-m256

## Change

Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only (scored rows verify_16 m=128 and verify_32 m=256; no test uses this band), replace master's path with a user-level split-K of 2. Master's path is torch.mm on the NN weight at m=65..128 (nvjet 128x64 splitK 20.2 us + cublasLt splitKreduce 3.0 us) and _bf16_gemm_dispatch_impl at m=256 (nvjet 128x64 TNT 30.4 us). WHY: pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) do the same FLOPs and bytes, but cuBLAS runs the long-K, narrow-N pooler worse. At m=256 the pooler takes 31.3 us (128x64 tiles, about 550 TFLOP/s) while the unpooler takes 24.5 us (128x128 tiles); at m=128 it is 22.9 vs 18.9 us. Two K-halves of the pooler have the unpooler's tile count: n 4096 x k 4096 x 2 batches gives the same 128 CTAs of 128x128 (m=256) or 128x64 (m=128) tiles, each over k=4096, as the unpooler's one GEMM. CHANGE: xh = x.view(m, 2, 4096).permute(1, 0, 2), a strided view [2, m, 4096] with strides (4096, 8192, 1) and no copy. For the weight, either candidate T, a TN view of state['pooler'] as [2, 4096, 4096] with strides (4096, 8192, 1) used transposed, or candidate N, the existing NN weight state['pooler_t'].view(2, 4096, 4096) (contiguous halves, no new memory). Then p = torch.bmm(xh, w_halves) into [2, m, 4096] bf16, and out = torch.add(p[0], p[1]): two graph nodes, with the add reading 4 MB and writing 2 MB at m=256. Precision: each half rounds once to bf16 (about 0.2% per element, L2 well under 1%). The single-nonzero rows stay exact, because the other half is exactly 0. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The profiler assertion must show exactly one nvjet/cutlass GEMM kernel plus one elementwise add, and no copy kernel; if torch inserts a contiguous() copy of xh, print it and keep master. Every other layer and m stays byte-identical.

## Expected

- `layer=pooler load=verify_32 m=256` faster: nvjet 128x64 30.4 us / row 31.33, compute floor 17.37; batched halves should match the unpooler m=256 nvjet 128x128 (24.8 kernel) + add ~1.2 -> row ~26
- `layer=pooler load=verify_16 m=128` faster: NN splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; halves should match unpooler m=128 (row 18.95) + add ~1.0 -> row ~20
- `layer=* load=decode_* m=*` unchanged: m <= 32 untouched

## Risk

Overlaps the cells of the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (pooler m=256) and of bf16-verify16-cutlass-m128-onetile-splitk-rewrite (pooler m=128), but this is pure cuBLAS dispatch at about 30 lines; if it wins, the CUTLASS gates measure against it. Not a retry of pooler-nn-layout-widen-m256 (flat), which kept one long-K GEMM.

## Adversarial review

Reviewed 2026-10-25T161612Z by an adversarial subagent (model opus).

**Verdict: proceed.** The conjecture is sound and this one run can confirm or refute it on its own two rows: two K-halves of the pooler as one strided-batched cuBLAS GEMM plus a bf16 add should reach the unpooler's 128x128 tile efficiency. The hypothesis is written against a stale base, though. Master is now blob f919748b, where pooler m=256 already runs NN torch.mm at 29.94 us per row, not dispatch TNT at 31.33. The build must re-baseline against that, and must put the new path under its own state flag, picked once in prepare() and inherited by state B, so an A/B self-check miss cannot turn off nn_ok or nn_p256_ok.

### Justification
A. Mechanism. Pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) have the same FLOPs and bytes. At m=256 the pooler runs 128 CTAs of 128x64 tiles over k=8192, the unpooler 128 CTAs of 128x128 over k=4096. Per CTA the MACs are equal, but the 128x128 tile reads about 33% fewer operand bytes per MAC from shared memory ((128+128)/128^2 against (128+64)/(128*64)). On H200 wgmma this is a real efficiency difference, and it matches the measured rows (unpooler 24.55 us, pooler 29.94 NN or 31.33 TN, compute floor 17.4). So 'match the unpooler's tile with a user-level split-K of 2' is plausible. Making the split K by 2 is lossless in bytes: the weight is read once and x is 4 MB. The extra traffic is the 4 MB bf16 partials written and read back plus a 2 MB write, about 2 us at realistic elementwise bandwidth in a graph, not 1.0-1.2 as stated. At m=128 the incumbent is already cuBLAS split-K (NN splitK 20.2 us + fp32 splitKreduce 3.0 us), so the gain there comes only from a better tile and replacing the fp32 reduce with a bf16 add. The m=128 margin is smaller and less certain than the m=256 one.

Numbers to correct. The hypothesis's m=256 baseline (dispatch TNT 30.4 us kernel, 31.33 us row) is from blob 353e3b19. On master f919748b (pooler-nn-layout-widen-m256, job 85c91578) the row is 29.94 us on an NNT kernel. A realistic prediction is about 24.5 (GEMM) + about 2 (add) = about 26.5 us at m=256, roughly 1.13x on the row, and about 20-21 us against 22.9-23.2 at m=128, roughly 1.1x. Each saves about 2.5-3.5 us on a load of about 73-94 us, so about +0.5% on the geomean per row and about +1% in all. That is inside the at-least-±1.5% CI, so judge the run on the two pooler rows' paired ratios and their kernel names, not the headline. The expected shapes and directions agree with TASK.md: only verify_16 and verify_32 pooler move, decode is untouched (m <= 32 gates come first in kernel_fn), and no test case sits in pooler 65..256.

Precision is fine. Each half rounds once to bf16 and the add rounds once more, giving about 1.5-2x a single rounding, about 2-3e-3 rel L2. That is under the master self-check's 6e-3 and the judge's 1%. Single-nonzero rows stay exact, because the other half's dot product is exactly 0 and the add of 0 is exact.

WEAKEST LINK: the conjecture assumes cuBLAS's heuristic for a strided-batched bf16 GEMM with batch=2, m=256, n=4096, k=4096 and lda=8192 will pick the unpooler's 128x128 nvjet tile and run it at the unpooler's speed. Nothing in FINDINGS has measured cuBLAS bmm on these shapes. The heuristic may instead pick 128x64 (256 CTAs, two waves) or a split-K variant, and the non-unit batch stride on x may also change the choice. The selector and the profiler assertion contain this risk: they protect correctness and keep master if the heuristic loses. If the selector keeps master, though, the scored run is a no-op on the board and teaches only what the prepare log prints.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 (landed, f919748b): this is now the m=256 incumbent (NN torch.mm, NNT kernel, row 29.94 with a paired 1.047 gain), not the dispatch TNT path the hypothesis describes. The selector compares against 'master's path', so it is correct by construction. Any credit in FINDINGS must be against 29.94, not 31.33, and the expected gain shrinks from about 5 us to about 3.5 us. It also owns nn_p256_ok and the AB cell (pooler, 4096, 8192, 256). If the new path rides under nn_p256_ok, a self-check miss on the new path also takes away the NN m=256 gain. The new path needs its own flag.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (landed): owns pooler_t, which candidate N reuses, and the shared nn_ok flag that also serves unpooler m=128. If the k-halves path at pooler m=128 is mapped to nn_ok in _ab_component, a self-check miss switches unpooler verify_16 back to TN as well, and that regression would be wrongly read as this hypothesis's effect. Map pooler 65..256 to a new flag (e.g. pooler_khalf_ok) checked BEFORE the nn_ok and nn_p256_ok ranges, and make a miss fall back to master's NN torch.mm, not to dispatch.
- sglang-gemm-dispatch-parity (landed): no longer the incumbent on either target row (both are torch.mm NN on master). It remains the fallback when NN flags are off, so if the attempt log shows nn_ok or nn_p256_ok False the comparison baseline has shifted and the run is void for attribution.
- _ab_selfcheck builds a second state B through _prepare_state. If the selector runs inside _prepare_state it runs twice and A and B can choose different paths. The choice must be made once on state A and passed to B through _ab_flags, like nn_p256_ok, otherwise the A/B replays compare different code paths.

### Interference and A/B needs
No companion. (i) An ab-revert of pooler-nn-layout-widen-m256 or of the NN restack would only re-create the stale incumbent the hypothesis was written against, and candidate N needs pooler_t anyway. (ii) An ab-tune has nothing to tune: split-K count 2 is the conjecture itself, and a split-K-4 variant would be a new hypothesis. (iii) A control isolating 'bmm without the add' cannot produce a correct output. The hypothesis already carries its own in-process control. The prepare-time selector times master's path and both candidates in the same process with L2 flushed, and the profiler names the kernel. So one run yields three readable facts: the batched GEMM's kernel name and tile (does it pick 128x128?), the in-process medians of master against the candidates for each (layer, m), and the judge's paired ratio on the two pooler rows. Both a gain and a loss are therefore interpretable without spending a second scored run. A loss with a 128x128 kernel refutes 'tile parity is enough'. A loss or keep-master with a 128x64 or split-K kernel refutes 'cuBLAS will pick the unpooler's tile'. A gain with a 128x128 kernel confirms it. Requirements that keep this true: print both candidates' medians and kernel names even when master is kept; record the final flags (nn_ok, nn_p256_ok and the new flag all True) and the chosen path per m in the attempt file; and if the selector keeps master at both m, log the outcome as 'refuted at prepare-time' rather than as a neutral board result.

### Open questions
- Does torch.bmm accept xh with strides (4096, 8192, 1) and pass lda=8192 and strideA=4096 straight to cublasGemmStridedBatched, or does it call contiguous()? The profiler assertion catches a copy. Also print xh.stride() and confirm no aten::contiguous or copy_ appears.
- Which candidate wins, T (TN view of state['pooler']) or N (pooler_t halves)? At m=256 NN already beat TN by about 4.7% on master, so N is the likelier winner. Record both.
- If cuBLAS picks 128x64 for the batched shape, the natural follow-up is a cuBLASLt matmul with a prepare-time-pinned algo (a handle and plan in state) for the 128x128 tile, not a retry of this hypothesis.
- Overlap with the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 and bf16-verify16-cutlass-m128-onetile-splitk-rewrite (same cells) and with combo-cublaslt-single-node-m128-m256-plus-naive (B4 touches pooler m=256 NN): whichever lands second must re-baseline these rows against the other.
- Should the selector's 1.02x threshold be per m? m=128 has the smaller expected margin and the larger risk of a false win from noise. With 41 reps the median error should be small, but print the IQR.


## Implementation notes

Base: worktree submission.py blob equals master's f919748b, so no merge was needed. Edited only muse_bf16_proj/submission.py (+181/-5).
- New `_khalf_forward` (bmm of the two K halves of x, then torch.add) with candidates T (TN view of state['pooler']) and N (pooler_t halves, no new copy). Reuses pooler_t.
- `_khalf_select(state)` runs once, on state A only, at the end of `_prepare_state(flags=None)`. For m=128 and 256 it captures master's NN torch.mm and both candidates as graphs and checks rel L2 and the profiler kernel names (exactly one GEMM and one elementwise add, no copy, memset or splitK reduce). It replays interleaved with a 256 MB flush, 41 reps, empty-graph baseline subtracted, and routes a candidate only at master/cand >= 1.02. Prints both medians, IQR and kernel names even when master is kept, plus xh.stride() and a PATH CHOSEN line per m.
- Review fixes: (1) re-baselined: the selector compares against master's actual NN torch.mm at both m. (2) New flag `pooler_khalf_ok`, plus a per-m choice dict `pooler_khalf`. `_ab_component(layer, m, state)` returns it BEFORE the nn_ok and nn_p256_ok ranges, only for m in the chosen dict. A miss turns off only that flag, so master's NN path stays and unpooler m=128 NN is untouched. (3) The choice is made once on A and inherited by B via `_ab_flags`. (4) Final flags and a `FINAL PATH pooler m=..` line per m are printed in prepare(). (5) No new weight copy. The per-m 1.02x threshold is applied per m, and the IQR is printed.
- Deviation: the route is exact-m (128, 256) and not the whole 65..256 band, because only those are measured and self-checked. The kernel_fn gate adds one dict lookup keyed on cfg m; there are no prints, state mutation or try/except in kernel_fn.
- Note: if nn_ok or nn_p256_ok turn off later in the A/B check, the baseline shifts (the review flagged this); the attempt log must record all three flags.

## Rules check

- [x] No subprocess/fork/spawn added.
- [x] No triton.autotune; no host sync, print or try/except in kernel_fn (diff checked).
- [x] Nothing derived from inputs cached; no device pointers or tensors in module globals (all in `state`).
- [x] Both states: B inherits the choice through _ab_flags; the A/B graph self-check covers pooler m=128 and 256 for both.
- [x] Precision: two bf16 roundings, below the 6e-3 self-check bound; the selector rejects a candidate above it.
- [x] Any selector failure keeps master's path and never raises.
- [x] No sm_90a PTX/CUDA load_inline added: n/a.
- [x] First-line marker kept; not committed, okc not run.

okc_lint: OK with warnings (only existing prepare()-side host-sync and flush-dtype warnings); py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (model opus, scored submit). **Verdict: dispatch-with-fixes.**

- Fix: `_khalf_names_ok` no longer asserts an exact kernel count (`len(gemm)==1 and len(elem)==1 and len(low)==2`) or rejects memsets. It now disqualifies only a copy kernel (memcpy/copy/contiguous on a non-GEMM name) or a cuBLAS splitKreduce, and prints the counts. Reason: nothing in this repo shows torch.profiler returning CUDA events inside the judge image. If CUPTI is unavailable, or the GEMM name matches none of nvjet/cutlass/gemm/sm90, or cuBLAS adds a memset, the exact count rejects both candidates, so the path never runs and the scored run is wasted (category 1). Numerics (rel L2 <= 6e-3) and the graph-replay selector still decide the path. An empty event list prints "name check skipped".
- Checked, no change needed. kernel_fn adds only a cfg/shape gate and a dict lookup on cfg m. `_khalf_forward` makes views, then bmm into [2, m, 4096] and add into [m, 4096], both sized by shape only. No prints, try/except or state mutation. Earlier routes (swapab16 m<=16, pooler split-K m<=16, swapab 17..32) cannot catch m=128/256, so the selector's master baseline (torch.mm on pooler_t) is master's actual path at both m. The T/N view math was verified in numpy against x @ W.T. `_ab_flags` carries pooler_khalf_ok and a copy of the pooler_khalf dict; B never re-runs the selector. `_ab_component` returns pooler_khalf_ok before the NN ranges, so an A/B miss turns off only that flag and kernel_fn falls through to master's NN torch.mm with nn_ok and nn_p256_ok untouched. A selector exception keeps master and never raises. Only a failed B-state build turns the NN flags off, which is master's existing behaviour.
- okc_lint OK (only pre-existing prepare-side warnings); py_compile OK; __pycache__ removed.

Residual risk: cuBLAS may pick a 128x64 or split-K tile for the strided bmm, so the selector keeps master and the run only measures master. The prints make that readable as "refuted at prepare-time".
