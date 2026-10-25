---
problem: muse_attention_decode
slug: full-decode-sk-s5-deferred-ticket-fault-fix
created: 2026-10-25T212052Z
base: master
area: full-layer attention (decode 1<B<=64, stream-K ticket epilogue)
change: >
  DIAGNOSIS action 4: re-measure the deferred ticket arrive, which has never been measured because
  it faulted. Re-apply done/failed/full-decode-sk-s4-deferred-ticket-arrive.patch (built on
  c3c1b01) onto CURRENT master 5d528bf3 (the stream-K kernel splitkv_attn_sk_kernel and its
  sk_pick/sk_prepare are unchanged there; resolve by hand). Keep its mechanism exactly: template
  flag DEFER on the single-level ticket path (B > 1): a finished non-whole segment stores its
  partial and records c_b in flag_s[2 + ndef] (ndef < 2, flag_s is int32[4] in Lay), the loop
  keeps streaming, and after the last tile ONE __threadfence + __syncthreads, then each recorded
  request draws its ticket and the last arrival merges + finalize_store + resets. Job 7f10d2d8
  failed 0/16 with an illegal address that surfaced inside the DEFER matrix (cuCtxSynchronize) and
  then the sk-diag; the sticky error poisoned every test. Fix the surface, not the design: (1)
  drop the CNT (variant 5, writes gticket[2047], which is also the two-level group slot c_b*32+gi
  for c_b=63) and SKIPM (variant 6) diagnostic instantiations; build only DEFER as variant 4. (2)
  Device guard in the DEFER epilogue: if cb is outside [0, B) or bx outside [c_lo, c_hi], set the
  error word and skip the draw instead of touching memory. (3) Host asserts before every DEFER
  launch: ws_acc/ws_ml rows >= grid + B, ticket numel >= B, and the matrix allocates its
  workspaces per cell from that cell's grid and B (assert, do not assume the self-check's sizes).
  (4) Matrix only at B=16 and B=32 (the scored bands) x grid 2xSM / SM; B 33..64 stays master
  variant 0. (5) Order: master's own sk-diag first, unchanged; the DEFER validate + matrix last;
  '[sk5] launching v4 B=.. grid=..' / 'returned' with torch.cuda.synchronize() around every DEFER
  launch, so a fault names its launch in the 15 KB log tail. (6) A canary per matrix cell (same B
  and grid, lengths 1..130) before its 80K timing. Route a band only if DEFER beats variant 0 by
  >= 1.02x in the interleaved graph replay and passes the fp32 self-check and the two-state
  alternating replay; print both numbers. First measurement is an okc check, never a scored
  submit. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no
  timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may
  hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting
  int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo,
  in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM
  and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2
  graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path
  WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "sk-diag: the ticket epilogue costs 112.7 vs 106.9 us no-ticket at B=16; deferring the in-loop fence/atomic should recover ~5 us of the 108.2 us kernel (0.107%/us, ~+0.5%)"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: 221.4 vs 218.6 us no-ticket at B=32; ~-3 us of 222.7 (~+0.13%)
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: two-level / c320 path untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
prior: "full-decode-sk-s4-deferred-ticket-arrive: same DEFER mechanism, which faulted before it was ever timed (illegal address, 0/16); this rebases onto current master, drops the CNT/SKIPM diagnostic kernels (CNT writes gticket[2047], a live two-level slot), adds device and host bounds guards, per-cell workspaces and canaries, and per-launch sync+print so any fault is named"
rewrite_stage: 5
replaces: "full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue"
review: required
status: queued
---

# full-decode-sk-s5-deferred-ticket-fault-fix

## Change

DIAGNOSIS action 4: re-measure the deferred ticket arrive, which has never been measured because it faulted. Re-apply done/failed/full-decode-sk-s4-deferred-ticket-arrive.patch (built on c3c1b01) onto CURRENT master 5d528bf3 (the stream-K kernel splitkv_attn_sk_kernel and its sk_pick/sk_prepare are unchanged there; resolve by hand). Keep its mechanism exactly: template flag DEFER on the single-level ticket path (B > 1): a finished non-whole segment stores its partial and records c_b in flag_s[2 + ndef] (ndef < 2, flag_s is int32[4] in Lay), the loop keeps streaming, and after the last tile ONE __threadfence + __syncthreads, then each recorded request draws its ticket and the last arrival merges + finalize_store + resets. Job 7f10d2d8 failed 0/16 with an illegal address that surfaced inside the DEFER matrix (cuCtxSynchronize) and then the sk-diag; the sticky error poisoned every test. Fix the surface, not the design: (1) drop the CNT (variant 5, writes gticket[2047], which is also the two-level group slot c_b*32+gi for c_b=63) and SKIPM (variant 6) diagnostic instantiations; build only DEFER as variant 4. (2) Device guard in the DEFER epilogue: if cb is outside [0, B) or bx outside [c_lo, c_hi], set the error word and skip the draw instead of touching memory. (3) Host asserts before every DEFER launch: ws_acc/ws_ml rows >= grid + B, ticket numel >= B, and the matrix allocates its workspaces per cell from that cell's grid and B (assert, do not assume the self-check's sizes). (4) Matrix only at B=16 and B=32 (the scored bands) x grid 2xSM / SM; B 33..64 stays master variant 0. (5) Order: master's own sk-diag first, unchanged; the DEFER validate + matrix last; '[sk5] launching v4 B=.. grid=..' / 'returned' with torch.cuda.synchronize() around every DEFER launch, so a fault names its launch in the 15 KB log tail. (6) A canary per matrix cell (same B and grid, lengths 1..130) before its 80K timing. Route a band only if DEFER beats variant 0 by >= 1.02x in the interleaved graph replay and passes the fp32 self-check and the two-state alternating replay; print both numbers. First measurement is an okc check, never a scored submit. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_16 *` faster: sk-diag: the ticket epilogue costs 112.7 vs 106.9 us no-ticket at B=16; deferring the in-loop fence/atomic should recover ~5 us of the 108.2 us kernel (0.107%/us, ~+0.5%)
- `*layer=full load=decode_32 *` faster: 221.4 vs 218.6 us no-ticket at B=32; ~-3 us of 222.7 (~+0.13%)
- `*layer=full load=decode_1 *` flat: two-level / c320 path untouched
- `*layer=sliding *` flat: untouched
- `*load=verify_* *` flat: untouched

## Risk

Weighted effect ~+0.6-0.7% headline: below the 1.24% margin, so it can only land stacked (with the merge pair or a verify win); judge this check on the splitkv_attn_sk_kernel rows at decode_16/32 and the printed DEFER gate.

## Milestones

Rewrite stage 5 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
