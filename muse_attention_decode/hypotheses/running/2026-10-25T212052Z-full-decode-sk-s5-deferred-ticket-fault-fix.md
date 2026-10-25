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
review: proceed
status: running
review_at: 2026-10-25T212956Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T212957Z
base_sha: d6a7be8f77635e613d619df51fa1de4dea8f9afd
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix
worktree: .worktrees/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T212956Z by an adversarial subagent (model opus).

**Verdict: proceed.** Worth one okc check: DEFER has never been timed, and s4's log shows the DEFER kernel passed all 26 eager self-check cases and 6 two-state replays (B=16/32/64, both grids) before the fault surfaced in the 80K matrix. But the brief's rebase premise is false. On master 5d528bf3, variants 4-7 of the sk kernel are already the PSEUDO instantiations, so 'build DEFER as variant 4' would silently become a pseudo launch. Separately, the 1.02x gate cannot route B=32 even if DEFER recovers that band's entire ticket gap.

### Justification
Target: full decode_16 is 110.8 us (sk kernel 108.2 us, floor 63.2 us, 1.71x the floor). Full decode_32 is 226.9 us (kernel 222.7 us, floor 126.4 us). The ticket-vs-no-ticket sk-diag gap of ~5.8 us at B=16 and ~2.8 us at B=32 is real, and DIAGNOSIS action 4 asks for exactly this measurement. The prior field is accurate: this is the same DEFER mechanism, which was never measured, plus guards. That is a fault-fix stage of a rewrite, not a retry of a measured loss, and nothing in Do-not-try covers it. Evidence on the fault: the archived log tail (archive/full-decode-sk-s4-deferred-ticket-arrive/7f10d2d820cd49ee.log) shows `_sk_defer_matrix raised ... cuCtxSynchronize`. That is the except branch that runs only after `_sk_defer_validate` returned None. So the DEFER kernel survived all 13x2 eager cases, including B=64 mixed at grid 7/SM, plus the two-state replays. The fault lives in a config only the matrix touched. The two new candidates are B=64 x 80000 (5.12M cache rows; no sk launch at B=64 x 80K had ever run, since sk-diag uses only B=16/32) and the variant-5 CNT `_sk_case(count B=.. x 80000)`. The brief's fixes (drop CNT/SKIPM, drop B=64 from the matrix) may remove the fault by avoidance, without identifying it. Per-launch sync+print is therefore the most valuable item: it guarantees the next fault is named. WEAKEST LINK 1, the rebase: the brief says splitkv_attn_sk_kernel and sk_pick/sk_prepare are 'unchanged' on master. They are not. The template is now <STAGES, TICKET, TWO, PSEUDO, TOK>, and sk_pick maps v4..v7 to PSEUDO kernels under SK_PSEUDO. sk_attn treats `variant >= 4` as pseudo (R = 8B; TORCH_CHECK B <= 32, so B=64 is rejected). sk_smem(v >= 4) returns Lay<4, PS_MAXR, true>::BYTES. `static bool done[8]` has no slot for a variant 8, so the host write would be out of bounds. A literal 'DEFER = variant 4' therefore launches the sliding-verify pseudo kernel on full-decode data, which is itself a candidate for a new illegal address. DEFER must be a new variant id (8), outside `#ifdef SK_PSEUDO` so the fallback non-PSEUDO build still has it. That also means SK_MAX_VARIANT and the pseudo/tok predicates must exclude 8, done[] must grow to 9, and _SK_LABELS must be extended. DEFER also goes as a 6th template parameter after TOK, with static_assert(!(DEFER && (PSEUDO || TWO))). WEAKEST LINK 2, the expected rows versus the gate: 1.02x at B=32 needs >= 4.4 us of a 222.7 us kernel, but the whole ticket gap there is ~2.8 us. So decode_32 'faster' is unreachable by construction, and decode_32 will stay on variant 0. At B=16, 1.02x needs >= 2.2 us of the 5.8 us gap. The s4 reviewer argued that most of that gap is the end-of-kernel serial merge tail, which DEFER keeps and can lengthen when a CTA becomes last arriver for both its head and tail request. Dropping SKIPM (variant 6) removes the only measurement that splits in-loop fence cost from the merge tail. SKIPM was not implicated in the fault: it ran only in sk-diag, after the matrix had already faulted. The likeliest honest outcome is a correct run with PATH CHOSEN = master at both bands and the DEFER-vs-v0 numbers printed. That reaches Milestone 1, and the numbers are the useful read.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: (a) it put FUSED_PRE skF in kernel_fn ahead of _full_decode_sk (`if splitkv and _fused_gate(cfg): return _full_decode_skf`), behind a runtime 1.02x gate it measured at 0.988x/1.014x. If noise flips decode_32 (or decode_16) to skF on this run, that band never reaches the DEFER route, and the shape row moves for reasons unrelated to DEFER. Read the [skF] PATH CHOSEN lines first. The DEFER gate compares against sk variant 0, not against skF. (b) Its PSEUDO instantiations share splitkv_attn_sk_kernel and occupy variants 4-7, which is the numbering collision above.
- restack-sliding-decode-sk-window-on-cluster-master: sliding decode_32 (and the sd route) runs splitkv_attn_sk_kernel PSEUDO TOK=1 (14.0 us row in the board profile). Any change to shared template code that is not constexpr-dead for PSEUDO instantiations moves sliding rows. Check that the regs and local-bytes attrs of variants 4-7 match master.
- full-decode1-streamk-s3-two-level-ticket: the B=1 TWO path shares the template, and DEFER sits on the !TWO branch. Full decode_1 (sk kernel ~18.6 us) must stay flat, and any move there means the shared prologue changed.
- full-decode-cuda-gather-splitkv: owns _splitkv_gate, which defines the 1<B<=64 band. It does not confound the kernel-row read, but B 33..64 has no scored shape, so B=64 rows in any matrix are unscored work.

### Interference and A/B needs
No companion is warranted. This is a kind: rewrite stage judged on the sk kernel's own top_kernels row at full decode_16/32 and on the in-prepare interleaved A/B (DEFER vs variant 0 at 2xSM/SM), which costs no check. A loss cannot reach the headline because the gate falls back to variant 0. A gain is attributable on the kernel row, provided the [skF] gate kept both bands on pre-attn + v2. Neither ab-revert nor ab-tune of a landed hypothesis would disambiguate anything the printed matrix does not already. Zero-check improvements to make inside this run instead: (1) Keep SKIPM as a diag-only variant, numbered after DEFER, in sk-diag at B=16/32. It never faulted and is the only split of fence cost vs merge tail. (2) Per the new LESSONS rule (single-round 1.02x selectors mis-routed), measure two interleaved rounds and route B=16 only when both rounds give >= 1.03x with a printed IQR. At B=32 just print the numbers: no gate value there can route a win that is physically capped at ~1.3%. (3) Since the matrix crash is unexplained, also run one eager variant-0 launch at B=64 x 80000 at the very end, after the matrix, with sync+print. That tells the next stage whether the s4 fault was B=64-at-80K-specific (master code) rather than DEFER.

### Open questions
- Which matrix cell faulted in 7f10d2d8? The 16 KB tail lost it. Was it the B=64 x 80000 band (never run before for any sk variant) or the variant-5 count case? If it was B=64, the s5 matrix drops it and the root cause stays unknown. Does master variant 0 itself survive B=64 x 80K?
- Will the builder renumber DEFER to variant 8 and fix sk_attn's `pseudo = variant >= 4` / `tok` / B<=32 check, sk_smem, done[] size and SK_MAX_VARIANT in both the SK_PSEUDO and non-PSEUDO builds? The brief as written says variant 4, which collides with PSEUDO single-level TOK=8.
- Did FUSED_PRE skF route either full band on this run? If it did, the DEFER result for that band is unread on the scored path.
- With the gate at 1.02x (or 1.03x x2 per LESSONS), B=32 cannot route. Is the stage's real deliverable the printed B=16 DEFER-vs-v0 delta plus the SKIPM gap split, and should the expected decode_32 row be 'flat'?

