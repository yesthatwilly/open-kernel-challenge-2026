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
stage: dispatched
claimed_at: 2026-10-25T212957Z
base_sha: d6a7be8f77635e613d619df51fa1de4dea8f9afd
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix
worktree: .worktrees/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T010812Z
impl_commit: 1f1606ddfd698aad1cae3543a4307890889d36a6
submission_sha256: 59ee0e087888e997328347f5a9ac23cf336319a41be37cc4261b067968f1e514
norm_sha256: 07544ddb1060da31e6763941bb5cc4d3c0758fd842cdccc9c1cd7995c9cc784b
submission_blob: 88bb41b311e7024af915b0e57604182680a03e7a
lint_warnings: 74
attempt_file: 2026-10-26T012539Z-check-full-decode-sk-s5-deferred-ticket-fault-fix.md
prepared_at: 2026-10-26T012539Z
job: f1d6d493c24244f4
dispatched_at: 2026-10-26T012615Z
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


## Implementation notes

Edited only `.worktrees/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix/muse_attention_decode/submission.py` (+477/-23). Not run (no GPU on this node).

**CUDA (module source `_SK_CUDA`)**
- `splitkv_attn_sk_kernel` gets two new trailing template parameters after TOK: `DEFER` (6th) and `SKIPM` (7th), with `static_assert(!(DEFER && (PSEUDO || TWO)))` and an assert that SKIPM excludes PSEUDO/TWO/DEFER. Both are constexpr-dead for variants 0-7, so those instantiations compile as before.
- Variant 8 = DEFER `<4, true, false, false, 8, true>`, variant 9 = SKIPM `<4, true, false, false, 8, false, true>` (diag only). Both are picked OUTSIDE `#ifdef SK_PSEUDO`, so the non-PSEUDO fallback build has them. `SK_MAX_VARIANT = 9` in both builds; `sk_variant_ok()` keeps 4-7 invalid in the non-PSEUDO build; `sk_is_pseudo(v) = 4..7` replaces `variant >= 4` in `sk_attn` and `sk_smem`, and `tok` is 1 only for 6/7. 8 and 9 are excluded from the pseudo / tok / B<=32 predicates and use `Lay<4>` smem. `done[]` is `SK_MAX_VARIANT + 1` = 10 entries (the review said 9, but variant 9 SKIPM needs index 9). `_SK_LABELS` extended to ten entries.
- DEFER mechanism as in s4 (record c_b in `flag_s[2+ndef]`, `ndef < 2`, one fence + sync after the last tile, then the ticket draws). Device guard: a recorded request outside [0, B) or a CTA outside [c_lo, c_hi] does `atomicExch(ticket + 64, 1)` and skips the draw (no memory touched).
- Error word: slot 64 of the existing static `_sk_ticket` int32 buffer (grown 64 -> 128 ints; slots >= 64 are used by nothing else, unlike gticket[2047], which is a live two-level slot). It is read only in prepare (`_s5_err_word`); `_rezero_tickets` clears it, and every ticket-sum check also trips on it.
- Host asserts in `sk_attn` for variant 8 (2 <= B <= 64, ticket numel > 64 and >= B, ws rows >= grid + B with the right shape) and Python asserts (`_s5_assert_defer`) before every matrix DEFER launch; each matrix cell allocates its own workspaces from its grid and B.

**Prepare (Python)**
- Order: master's sk-diag first (unchanged except the added SKIPM variant-9 diag at B=16/32, one announced + synced eager launch first, then timed, with the fence-vs-merge gap split printed), then `_s5_phase()` LAST: `_s5_validate` (attrs of all sk variants incl. 4-7 local-byte warnings and the variant-0 identity line; 10 eager cases at B=2/16/32 + B=24x80000, no B=64; two-state alternating replays at B=16 for both grids, replaying pre-attn + variant 8 directly so skF cannot intercept) -> `_s5_band(16, route=True)` -> `_s5_band(32, route=False)` -> `_s5_probe_b64` (one eager variant-0 launch at B=64 x 80000, sync + print, after the matrix).
- Every DEFER launch prints `[sk5] launching v8 B=.. grid=..` and `returned (error word ..)` with `torch.cuda.synchronize()` around it. A canary per cell (lengths 1..130, same B and grid) runs before that cell's 80K timing.
- Gate (B=16 only): master's routed path (variant 0, `_SPLITKV_GRID or 2*SM`, flag-off) vs DEFER at 2xSM and SM; two interleaved rounds (rotation reversed in round 2), cold L2 (256 MB flush), empty graph subtracted, median/IQR of 20 printed per cell per round. Route only if BOTH rounds >= 1.03x AND the cell's IQR <= 1.5x master's in both rounds AND the band is not owned by skF (`_fused_gate`). B=32: numbers only, never routes. Route lands in `_SK_ROUTE[16] = (8, mult)`, read in `_full_decode_sk` by batch only; a miss leaves master's exact (variant 0, 2xSM) path.
- Deadlines: the phase starts only below 235 s of import clock, each band/probe below 275 s (`skipped: deadline` printed). All PATH CHOSEN/GATE lines of the phase are repeated as the last lines of prepare (`[sk5] SUMMARY`).

**Deviations / notes**
- `done[]` grown to 10, not 9 (see above). The review's variant-4-7 "keeps master's regs/local" check is a printed comparison: master's pseudo-variant reg counts are not recorded anywhere in the repo, so the `[sk5]` attrs lines must be compared by eye with the `[ps]` lines of master's log; variant 0 is checked against the landed 121/0/95264/2.
- No `_ab_flags`/NEW_CELL_FLAGS pattern exists in this problem, so there is no per-state flag; a miss falls back by construction.
- Biggest risk: variant 8 compile/regs (DEFER adds live state; a spill refuses it, < 2 CTAs/SM is only warned) and the unexplained s4 fault, which may not be DEFER-specific.

## Rules check

- [x] CUDA-graph capturable: kernel_fn change is one dict `.get` by batch and the same launch; no sync/.item()/alloc on values in kernel_fn (all syncs/.item() are prepare-only, lint: host-sync warnings only outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: `_SK_ROUTE` is keyed by config batch, written only in prepare; the ticket buffers are self-resetting int32 (error word is a device-only int, read in prepare).
- [x] Weights: n/a: no weight handling touched.
- [x] Activations / KV / queries: KV FP8, e4m3 queries, fp32 softmax/merges, bf16 out, unchanged (DEFER only reorders when the ticket draw happens).
- [x] Attention sees every key: tile ranges and merges unchanged; DEFER merges exactly the same slots.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: timing only in prepare, no child processes.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only `host-sync` warnings for prepare-only synchronize()/.item() outside kernel_fn's call graph); `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Pre-flight audit

Verdict: dispatch-with-fixes. Evidence: diff of master vs worktree saved at /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/s5.diff (read by hand; no new CUDA index maps, so no numpy emulation).
- Variant ids: `sk_is_pseudo` = 4..7 drives pseudo, sk_smem and the B<=32 check; `tok` is 1 only for 6/7; 8/9 are non-PSEUDO in both builds (sk_pick 8/9 sit outside `#ifdef SK_PSEUDO`; `sk_variant_ok` rejects 4-7 in the non-PSEUDO build); `done[10]`, `SK_MAX_VARIANT=9` and the 10 `_SK_LABELS` agree. OK.
- Error word: ticket[64] of the 128-int buffer; master indexes ticket[c_b], c_b < 64, so it never reaches 64..127; `_rezero_tickets` zero_()s all 128 and the ticket-sum checks include slot 64. OK.
- Guard: no spin-waits anywhere, so a skipped draw leaves the ticket nonzero (caught by the error word / sum / re-zero) and cannot hang. OK.
- Order: master sk-diag, then DEFER validate, B=16, B=32, B=64 v0 probe last; every DEFER launch is synced and printed. OK.
- Fix 1: phase start/band deadlines 235/275 s -> 420/470 s (import clock starts after `import torch`; master compiles alone are ~200 s per DIAGNOSIS.md and the phase takes under a minute, TASK.md allows 900 s, so 235 s risked skipping the phase on every run).
- Fix 2: the IQR veto (`<= 1.5x master IQR`) gets a 0.5 us floor on master's spread (`1.5 * max(master_iqr, 0.5)`).
Residual risk: a DEFER or B=64 v0 illegal address would poison the context for the remaining prepare and tests (B=64 x 80K is neither scored nor tested), and a compile error in the shared _SK_CUDA module would lose master's sk path.
