---
problem: muse_moe_w4a16
slug: wg-stream-k-tail-split
created: 2026-10-25T153417Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Wave quantisation fix inside master's persistent wg_kernel, both kernels, 128 <= T <= 512
  (verify_32 bs=256; verify_16 only if its flag is on). MECHANISM: items cost almost the same
  (each streams one 128-row tile of one expert over the full K, and the kernel is dequant-bound on
  weight bytes, not MMA-bound on N), so with total = nc x n_tiles items strided over G = 132 CTAs
  the makespan is ceil(total / G) items while the mean is total / G: at bs=256 w13 has ~2048 items
  (15.5 waves, the last ~half empty) and w2 ~1024 (7.8 waves); about half an item time (~19 us
  each) of SM time is idle per kernel on average. CHANGE: with q = total / G and r = total mod G
  computed on device in the prologue (no host branching), items it < q x G run as now. The r tail
  items are each split into S = 2 K-halves (S = 1 if 2r > G, i.e. when the tail is already more
  than half full) and the 2r half-items are strided over the CTAs from blockIdx.x = 0 after the
  full waves, so every CTA does at most q full items plus one half. A half-item accumulates its K
  half in fp32 as now, then writes the raw fp32 accumulator tile (128 x N, before any
  scale/silu/bf16) to a per-state fp32 workspace slot (sized G x 128 x 128 fp32 = 8.4 MB,
  allocated in prepare), __threadfence, and takes a per-item ticket with one gpu-scope atomicAdd
  (the split-K ticket pattern LESSONS records as correct: masked partial store, fence, acq_rel
  ticket, winner reloads with ld.global.cg, fixed-order sum, ticket reset by the winner). The
  second arriver sums both halves in fixed order (half 0 then half 1) and runs master's epilogue
  unchanged (silu for w13, plain for w2) from the summed accumulator. Ticket array: per-state
  int32 [G] zeroed in prepare and reset by the winner, so replays and state copies are safe. The
  producer's flat stage sequence just gets half the k-stages for a half-item (its B-slab TMA
  coordinates start at kb0 = 0 or KS/2). Numerics: fp32 sums of two halves in fixed order (not
  bitwise equal to master, within fp32 rounding). PREPARE: s5 bucket canary and three-state copy
  check on the new kernel, with the self-check routings chosen so r spans 0, small, ~G/2 and >
  G/2; then the _wg_select bucket selector vs master's kernel (cold-L2 graph A/B, margin 1.02x),
  PATH CHOSEN line naming the tail-split kernel, and per-kernel event times. Flag
  state['wg_tail_split']; kernel_fn branches on shape and flags only. BUILD (as master's wg
  module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode
  arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under
  __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy /
  L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes /
  smem (a spill line is a bug: fix before spending the check). No module-global device pointer,
  tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded
  per call from the tensor passed in that call (master's s5 rule; the judge prepares two states
  and copies them by cloning tensors). First launch in-process after a small synthetic canary,
  then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first
  measurement is an okc check, never a scored submit.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "verify_32: w13 ~303 us and w2 ~148 us judged; expected tail recovery ~5-10 us per kernel on average (half an item of ~19 us times the idle fraction), target -2 to -4% on the shape (<= 395 us). Read the two wg_kernel rows against master's"
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin unless the selector flips
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T203429Z
review_companions: null
hold_reason: "adversarial review: Tail mechanism real but small; numbers copied from the wrong profile; the spec's own condition is an okc check first and only scored submits remain today; a mas"
outcome: expired
verdict_detail: "retired: superseded per the 2026-10-25 moe diagnosis (do not run; replaced by the wg2-based line or measured dead)"
expired_at: 2026-10-25T212024Z
---

# wg-stream-k-tail-split

## Change

Wave quantisation fix inside master's persistent wg_kernel, both kernels, 128 <= T <= 512 (verify_32 bs=256; verify_16 only if its flag is on). MECHANISM: items cost almost the same (each streams one 128-row tile of one expert over the full K, and the kernel is dequant-bound on weight bytes, not MMA-bound on N), so with total = nc x n_tiles items strided over G = 132 CTAs the makespan is ceil(total / G) items while the mean is total / G: at bs=256 w13 has ~2048 items (15.5 waves, the last ~half empty) and w2 ~1024 (7.8 waves); about half an item time (~19 us each) of SM time is idle per kernel on average. CHANGE: with q = total / G and r = total mod G computed on device in the prologue (no host branching), items it < q x G run as now. The r tail items are each split into S = 2 K-halves (S = 1 if 2r > G, i.e. when the tail is already more than half full) and the 2r half-items are strided over the CTAs from blockIdx.x = 0 after the full waves, so every CTA does at most q full items plus one half. A half-item accumulates its K half in fp32 as now, then writes the raw fp32 accumulator tile (128 x N, before any scale/silu/bf16) to a per-state fp32 workspace slot (sized G x 128 x 128 fp32 = 8.4 MB, allocated in prepare), __threadfence, and takes a per-item ticket with one gpu-scope atomicAdd (the split-K ticket pattern LESSONS records as correct: masked partial store, fence, acq_rel ticket, winner reloads with ld.global.cg, fixed-order sum, ticket reset by the winner). The second arriver sums both halves in fixed order (half 0 then half 1) and runs master's epilogue unchanged (silu for w13, plain for w2) from the summed accumulator. Ticket array: per-state int32 [G] zeroed in prepare and reset by the winner, so replays and state copies are safe. The producer's flat stage sequence just gets half the k-stages for a half-item (its B-slab TMA coordinates start at kb0 = 0 or KS/2). Numerics: fp32 sums of two halves in fixed order (not bitwise equal to master, within fp32 rounding). PREPARE: s5 bucket canary and three-state copy check on the new kernel, with the self-check routings chosen so r spans 0, small, ~G/2 and > G/2; then the _wg_select bucket selector vs master's kernel (cold-L2 graph A/B, margin 1.02x), PATH CHOSEN line naming the tail-split kernel, and per-kernel event times. Flag state['wg_tail_split']; kernel_fn branches on shape and flags only. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit.

## Expected

- `bs=256 *` faster: verify_32: w13 ~303 us and w2 ~148 us judged; expected tail recovery ~5-10 us per kernel on average (half an item of ~19 us times the idle fraction), target -2 to -4% on the shape (<= 395 us). Read the two wg_kernel rows against master's
- `bs=128 *` flat: B128 stays Marlin unless the selector flips
- `bs=1 *` flat: control
- `bs=16 *` flat: control

## Risk

Competes with wg-pdl-gather-w13-w2-chain for the same idle tail (that one only helps the w13 -> w2 boundary; this one also covers w2's own tail before the post-norm); measure each standalone, then stack. Risk: the gain depends on r mod G for the judge's routing, so one routing may show nothing; the log should print q, r and S for the scored T. Not touched by the queued consumer-side lines, but it edits the same item loop: whichever lands second rebases.

## Adversarial review

Reviewed 2026-10-25T203429Z by an adversarial subagent (model opus).

**Verdict: hold.** Tail mechanism real but small; numbers copied from the wrong profile; the spec's own condition is an okc check first and only scored submits remain today; a master-based gain (<= ~2% on verify_32, ~+0.3% headline) cannot pass the 1.4021x occ2 board entry. Hold until checks reset, like wg-producer-direct-x-rows-no-gather.

### Justification
wg_kernel walks total = nc x n_tiles items over G=132 CTAs; makespan ceil(total/G). Real per-kernel times on master: wg_kernel<true> ~238 us, <false> ~112-119 us (~15 us per item), not 303/148. The S=2 rule only helps when r = total mod G <= G/2 (64*nc mod 132 for w13, 32*nc mod 132 for w2): a coin flip per kernel, expected 0-15 us of 358-405 us. True stream-k or S = min(KS, floor(G/r)) recovers the tail for every r with the same ticket/workspace machinery. Prior 'none' is defensible (wg-pdl-gather-w13-w2-chain targets the same tail from the dependent side, flat).

### Confounders on the current kernel
- wg-pdl-gather-w13-w2-chain (landed): PDL hides ~2-6 us of the same tail; savings overlap.
- align-two-phase-port-from17 / combo F2 align2 front end: set nc and so r; log q, r, S per kernel.
- combo-wg-occ2-v64-hot-interleave-align2 (board best, not on master): G=264, NMAX 64; a master result does not carry over.

### Interference and A/B needs
No companion: self-attributing via printed q, r, S and the prepare-time per-kernel A/B; the real question (which r the judge routing gives) is a print.

### Open questions
- Compute nc and r under the judge routing (_wg_judge_ids) offline before spending a run.
- True stream-k or S = floor(G/r)?
- Rebuild on the occ2 branch's wg2_kernel (G=264) where it can move the board?
- Assert identical producer/consumer item sequences in the canary.
- Correct expected rows to master's real per-kernel times.

