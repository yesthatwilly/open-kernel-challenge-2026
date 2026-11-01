---
problem: muse_moe_w4a16
slug: wg-producer-direct-x-rows-no-gather
created: 2026-10-25T153417Z
base: master
area: MoE dispatch (wgmma x gather)
change: >
  Drop the wg_gather_kernel launch and the wg_xs round trip on master's wgmma path at 128 <= T <=
  512 (verify_32 bs=256; verify_16 only if its flag is on). Align, both GEMM consumers, w2's B
  slab (wg_hs), epilogues and post-norm are unchanged. MECHANISM: today wg_gather_kernel (4.7 us
  at bs=256 in the profile) copies x[sorted_ids[slot] / 8] into wg_xs ([slots, 4096] bf16, ~3.7 MB
  at bs=256, zeros for padded slots), and wg_kernel<w13>'s producer TMAs the B slab from wg_xs.
  CHANGE: in wg_kernel<w13> only, producer warps 1-3 (idle today; 96 threads, still at
  setmaxnreg.dec 40) load the B slab of each stage directly from x: for the stage's N rows (the
  chunk's slots slot0 .. slot0+N-1) and its 128-k column window, each thread issues 16-byte
  cp.async.cg (plain form, no cache hint: the form the judge has run) from x[token, kb*128 + 8c]
  into the slab address that the 128B swizzle of master's tensor map would produce (16-byte chunk
  index c XOR (row & 7), two 64-k halves at +SLAB_HALF), zero-filling rows whose slot is a padded
  sentinel (st.shared of zeros; no src-size zfill form). Completion:
  cp.async.mbarrier.arrive.noinc.shared.b64 on the stage's full barrier by each of the 96 threads
  after its copies, with the full barrier's expected count raised from 1 to 1 + 96 and the
  expect_tx reduced to the weight bytes only (WST). Warp 0 keeps the weight bulk copies. The
  per-call tensor map for w13's B (maps.m[bucket] over wg_xs) is no longer encoded for w13 (w2's
  maps over wg_hs stay). kernel_fn no longer launches wg_gather. Slots map to tokens via
  sorted_ids (route / 8), read once per stage per row from L2. PREPARE: the s5 bucket canary for
  N16/N32/N64/N128 and three-state copy check on the new w13 kernel (outputs must be bitwise
  identical to master's w13+silu: same data, same math), the _wg_select selector re-run, PATH
  CHOSEN line naming the direct-x producer, and w13 event times with and without it. Flag
  state['wg_direct_x']; kernel_fn branches on shape and flags only. BUILD (as master's wg module):
  own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode
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
    why: "verify_32: removes the 4.7 us wg_gather row and one launch gap; the w13 row must stay within +-2% of master's 239-303 us (B-slab bytes now come from x via L2 instead of wg_xs via TMA). Target -1.5 to -2.5% on the shape; attributable by the vanished gather row and the unchanged w13 row"
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-25T203206Z
review_companions: null
hold_reason: "own condition: first measurement must be an okc check (new cp.async/mbarrier producer can fault or hang); only scored submits remained on 2026-10-25, and a gain on master cannot move the board (best is the occ2 branch). Unhold when checks reset; apply the reviewer's fence.proxy.async and nvalid-bound fixes when building"
---

# wg-producer-direct-x-rows-no-gather

## Change

Drop the wg_gather_kernel launch and the wg_xs round trip on master's wgmma path at 128 <= T <= 512 (verify_32 bs=256; verify_16 only if its flag is on). Align, both GEMM consumers, w2's B slab (wg_hs), epilogues and post-norm are unchanged. MECHANISM: today wg_gather_kernel (4.7 us at bs=256 in the profile) copies x[sorted_ids[slot] / 8] into wg_xs ([slots, 4096] bf16, ~3.7 MB at bs=256, zeros for padded slots), and wg_kernel<w13>'s producer TMAs the B slab from wg_xs. CHANGE: in wg_kernel<w13> only, producer warps 1-3 (idle today; 96 threads, still at setmaxnreg.dec 40) load the B slab of each stage directly from x: for the stage's N rows (the chunk's slots slot0 .. slot0+N-1) and its 128-k column window, each thread issues 16-byte cp.async.cg (plain form, no cache hint: the form the judge has run) from x[token, kb*128 + 8c] into the slab address that the 128B swizzle of master's tensor map would produce (16-byte chunk index c XOR (row & 7), two 64-k halves at +SLAB_HALF), zero-filling rows whose slot is a padded sentinel (st.shared of zeros; no src-size zfill form). Completion: cp.async.mbarrier.arrive.noinc.shared.b64 on the stage's full barrier by each of the 96 threads after its copies, with the full barrier's expected count raised from 1 to 1 + 96 and the expect_tx reduced to the weight bytes only (WST). Warp 0 keeps the weight bulk copies. The per-call tensor map for w13's B (maps.m[bucket] over wg_xs) is no longer encoded for w13 (w2's maps over wg_hs stay). kernel_fn no longer launches wg_gather. Slots map to tokens via sorted_ids (route / 8), read once per stage per row from L2. PREPARE: the s5 bucket canary for N16/N32/N64/N128 and three-state copy check on the new w13 kernel (outputs must be bitwise identical to master's w13+silu: same data, same math), the _wg_select selector re-run, PATH CHOSEN line naming the direct-x producer, and w13 event times with and without it. Flag state['wg_direct_x']; kernel_fn branches on shape and flags only. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit.

## Expected

- `bs=256 *` faster: verify_32: removes the 4.7 us wg_gather row and one launch gap; the w13 row must stay within +-2% of master's 239-303 us (B-slab bytes now come from x via L2 instead of wg_xs via TMA). Target -1.5 to -2.5% on the shape; attributable by the vanished gather row and the unchanged w13 row
- `bs=128 *` flat: B128 stays Marlin
- `bs=8 *` flat: control
- `bs=32 *` flat: control

## Risk

Avoids the align area (the in-flight align-two-phase-port-from17 touches the Marlin path's align; this keeps the wgmma path's moe_align_block_size call as is). Code overlap: the queued producer-scale-expand also wants producer warps 1-3 for scale conversion; if it lands first, give x loads warp 3 only or split the 96 threads, and re-check the setmaxnreg budget. Effect is small (~1-2% on one load), near the noise floor; judge it on the kernel rows.

## Adversarial review

Reviewed 2026-10-25T203206Z by an adversarial subagent (model opus).

**Verdict: proceed.** Real, new, attributable mechanism (deletes the gather launch and the wg_xs round trip rather than overlapping it; not a duplicate), but the payoff is ~1% on verify_32 (~0.2% headline), smaller than claimed, and the spec misses a fence.proxy.async that wgmma needs, so as written it can be wrong.

### Justification
Producer warps 1-3 idle after setmaxnreg.dec 40 (submission.py ~1314-1338). wg_gather_kernel is 3.8 us (0.9%) of 402.4 us at verify_32 (combo-toprow b395f5b6); ceiling ~-1.2% on bs=256 incl. the launch gap; bs=256 err_ns ~3.2 us. Weakest link: w13 row flat when B moves from 2 TMA boxes to hundreds-thousands of 16-byte cp.async by 96 threads at 40 regs; a 2% w13 regression erases the gain. Hoist sorted_ids loads per item; copy only nvalid rows. Correctness gaps: (1) fence.proxy.async.shared::cta between generic-proxy writes (cp.async, zero-fill) and wgmma reads; (2) warps 1-3 must run the item/stage loop and wait on empty barriers; (3) bound rows to nvalid / sorted_ids.numel(). Prior (missing): wg-pdl-gather-w13-w2-chain (different mechanism).

### Confounders on the current kernel
- wg-pdl-gather-w13-w2-chain: same launch chain; read WG MODULE BUILT and wg_pdl lines before attributing.
- combo-toprow / align-two-phase-port-from17: front end sets sorted_ids layout and bucket mix; check PATH CHOSEN / wg_align2 lines match.
- Board best is the occ2 branch (wg2_kernel, 1.4021x), not master (1.3866x): a clean gain here cannot move the board.

### Interference and A/B needs
No companions: the gather row vanishing and wg_kernel<true> staying within ~1 us of 258 us is the readout.

### Open questions
- Empty-barrier waits and full-barrier arrive counts for warps 1-3 with zero rows.
- Hoist sorted_ids per item.
- Bucket mix at verify_32 (N64/N128 share).
- Interaction with queued producer-scale-expand (also wants warps 1-3).

