---
problem: muse_moe_w4a16
slug: engine-a-xl-arms-t256-postnorm-stack
created: 2026-10-30T002759Z
base: master
mint_blob: "d80df5ecc71315466253dae86b28e4f6dfc41642"
best_at_mint: 2.0352
area: "moe expert block: Engine A XL dequant arms (all classes) + T=256 ea_combine replacing the post-norm weighted sum"
change: >
  Stack two shelved moe riders on master (2.0352, f2b510f5): (1) the Engine A XL arms from
  takemyjob hopper/ea-enc2-1029 04306fa (patch_enc2.py: E4XL / E5XL = E4/E5 + ldmatrix.x4 B
  preload + per-record loop trim; bit-exact; research/h100-ea-enc2-1029.md: consume-only
  1.014-1.055 over the routed arm, largest at T=256 E5XL 1.039-1.055) added as extra arms in
  master's per-class [ea-enc] race (same rules: exact, <= 0.98 vs master every round; at T=256
  also print the isolated cold-half ratio); (2) at T=256 only, the hybrid's post-norm weighted sum
  replaced by ea_combine with PDL (takemyjob hopper/glue-1029 f3d0fea; research/h100-glue-1029.md:
  x1.016 on the T=256 row, rel L2 4.19e-3 vs master 4.59e-3, not bitwise), raced at T=256 through
  kernel_fn with a tolerance gate (fp32 reference, _ea_canary's rule) and <= 0.99 every round,
  else master's post-norm. Keep the build/join deadlines (EAE_JOIN_S 220) and size < 1,000,000 B
  (moe is 948,330 B).
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "T=256: E5XL consume 1.039-1.055 over E5/E4 + ea_combine x1.016"
  - shapes: load=decode_16*
    direction: faster
    why: E4XL consume 1.02-1.047 over E4 at T8-32
  - shapes: load=decode_32*
    direction: faster
    why: E4XL
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "engine-a-a4b-a5b-encodings-race: landed E4/E5; this adds the XL loop trim and the T=256 combine swap that the ea-enc2 and glue lanes measured"
target: check
value_gain: 1.5
value_p: 0.5
value_ev: 0.75
value_cost: 400k tokens, 1 judge run
evidence: >
  H100 ea-enc2 (runs 20261029-160207-gpunode06): E4XL consume-only 1.014-1.047x at T8-32, E5XL
  1.039-1.055x at T=256 over the routed arm, SASS 617 -> 566 issued/record; H100 glue
  (research/h100-glue-1029.md): T=256 post-norm 10.0 us of a 321.4 us row, ea_combine+PDL x1.016
  (2 replicates x 50 pairs); judge 702f1004: consume gains transferred at ~60-70%
user_requested: true
review: proceed
status: done
review_at: 2026-10-30T003155Z
review_companions: null
review_best: 2.035180868162415
review_blob: d80df5ecc71315466253dae86b28e4f6dfc41642
stage: harvested
claimed_at: 2026-10-30T003155Z
base_sha: 3c2132ae83986e59ae1e72def1637af0f8535c5f
base_blob: d80df5ecc71315466253dae86b28e4f6dfc41642
branch: hyp/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack
worktree: .worktrees/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-30T004456Z
impl_commit: 28cd633359d5c9a8c8abd9d852b3bb8b056eb210
submission_sha256: 091c05b8670abe6ec6ce3a4b93e66944ca8e40d8c47c098bb0c1bd0830800c17
norm_sha256: 6c66cf5a8bc6742be6765f395510f38f7aa5cb1c3084d5247d09a7332ea534c9
submission_blob: 5220385a87251699b3c366632c532841f8a389f0
lint_warnings: 204
attempt_file: 2026-10-30T004457Z-check-engine-a-xl-arms-t256-postnorm-stack.md
prepared_at: 2026-10-30T004457Z
job: 73156455c9f242f6
dispatched_at: 2026-10-30T004507Z
outcome: gain
verdict_detail: gain
speedup: 2.0666647252878665
speedup_ci: [2.0356647544085487, 2.0976646961671843]
harvested_at: 2026-10-30T005157Z
best_at_harvest: 2.035180868162415
promotion: pending
mechanism: null
---

# engine-a-xl-arms-t256-postnorm-stack

## Change

Stack two shelved moe riders on master (2.0352, f2b510f5): (1) the Engine A XL arms from takemyjob hopper/ea-enc2-1029 04306fa (patch_enc2.py: E4XL / E5XL = E4/E5 + ldmatrix.x4 B preload + per-record loop trim; bit-exact; research/h100-ea-enc2-1029.md: consume-only 1.014-1.055 over the routed arm, largest at T=256 E5XL 1.039-1.055) added as extra arms in master's per-class [ea-enc] race (same rules: exact, <= 0.98 vs master every round; at T=256 also print the isolated cold-half ratio); (2) at T=256 only, the hybrid's post-norm weighted sum replaced by ea_combine with PDL (takemyjob hopper/glue-1029 f3d0fea; research/h100-glue-1029.md: x1.016 on the T=256 row, rel L2 4.19e-3 vs master 4.59e-3, not bitwise), raced at T=256 through kernel_fn with a tolerance gate (fp32 reference, _ea_canary's rule) and <= 0.99 every round, else master's post-norm. Keep the build/join deadlines (EAE_JOIN_S 220) and size < 1,000,000 B (moe is 948,330 B).

## Expected

- `load=verify_32*` faster: T=256: E5XL consume 1.039-1.055 over E5/E4 + ea_combine x1.016
- `load=decode_16*` faster: E4XL consume 1.02-1.047 over E4 at T8-32
- `load=decode_32*` faster: E4XL

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 ea-enc2 (runs 20261029-160207-gpunode06): E4XL consume-only 1.014-1.047x at T8-32, E5XL 1.039-1.055x at T=256 over the routed arm, SASS 617 -> 566 issued/record; H100 glue (research/h100-glue-1029.md): T=256 post-norm 10.0 us of a 321.4 us row, ea_combine+PDL x1.016 (2 replicates x 50 pairs); judge 702f1004: consume gains transferred at ~60-70%
- **Gain if it works:** 1.5% headline; **p** 0.5; **EV** 0.75% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.5% < 2.5%; EV 0.75% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge run.

## Risk

Each part ~+0.5-1%; stacked ~+1.5%. The E encodings' node consume-only gains transferred on the judge (702f1004: 4-10% consume -> 3-5% per class), so XL's 2-5% should show ~1.5-3% per class.

## Adversarial review

Reviewed 2026-10-30T003155Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: CLAIMS.md:108 [shelf] XL reopen condition ('ride in the next Engine A candidate as extra raced arms') is met and the board has not moved (2.0352, blob d80df5ec); proceed as a user-requested below-bar stack. The XL half carries nearly all the value. The post-norm half is worth about 0.15-0.2% headline on the judge and will usually lose its 0.99-every-round bar.

### Justification
XL is plausible. It cuts per-record loop bookkeeping (SASS 617 -> 566 issued/record, a per-SM issue cut, which is the kind of change that transfers). It is bitwise E4/E5 (same loads, same order, same arithmetic) and was measured exact at every draw over 9 replicates. The E4/E5 landing transferred node consume-only 4-10% into judge kernel_fn E/m 0.955-0.965 (582bb334 log lines 118-122), and at 0.5-0.8x the XL consume gains of +1.4..+5.5% become about 1-2.5% per class. The probe's split bits match master's routes: v4xs at T<=16, v4cxs at 17..32/128/256, which agrees with the judge's [ea-enc] lines (T8/16 v4xs, T32/128/256 v4cxs). XL's run arithmetic (cnt0 = NK - (b & (NK-1)), flush at super-group end or e, setpass only at pass change) reproduces the master loop's flush points for any b/e, including cost-weighted splits, and master's body returns early on b >= e, so an empty CTA cannot reach XL's count-down loop with cnt 0. The `expected` list leaves out two rows XL also targets: verify_1 (T=8, E4XL 1.014-1.037) and verify_16 (T=128, E5XL 1.020-1.033). Add them. Weakest link 1: the port is not mechanical. patch_enc2.py anchors on the probe tree (eak::consume<DQ> with scale_enc/frag_a4b and dq 27/28 selected by `dq`). Master's E module is _eae_src4x(), which uses consume_e/ext_a4b/mul_bf16x2_h/scale_pair_imad_rt and selects E5 by pdl bit 4096 with ea['dq'] fixed at 2. So consume_lt has to be re-derived from master's consume_e, and XL needs a new pdl selector bit; check it against EA_PDL_BITS for collisions. Bitwise equality then holds only if the race's exact h/y gate passes. That gate exists (_eae_hy, NaN-poisoned) and must cover the XL arms. Weakest link 2: at T=256, XL only speeds up the cold half, which runs on the W6-M fork beside the hot half. The node had it off the critical path. On the judge, cold-iso E4 0.964 showed up as kernel_fn 0.975 (about 70%). The probe also measured T=256 on full routing, not on the hybrid's cold id set (experts with <= 16 routes), so the T=256 1.04-1.055 is an upper bound. The verify_32 gain is probably about 1-2% of the row. Post-norm swap: the hypothesis's x1.016 is the node figure. The glue note itself says the judge's SGLang post-norm is already 5.7 us, versus 9.95 on the node, and expects x1.008-1.011 on the row and about 0.15-0.2% headline. A <= 0.99 every-round bar against a ~0.99 true ratio with about ±0.3% round spread will mostly fail, so expect T=256 to stay on master's post-norm. The prior is right: the XL loop trim and X preload are new on top of the landed E4/E5. The probe found X null (E4X 0.99-1.02), so build only the XL instances. No do-not-try overlap: the refuted combine items (CLAIMS:39, cmbf) are Triton-on-Marlin and w2-epilogue fusion, not ea_combine at the T=256 tail, and ea_combine is already master's class-128 path. Build time: the [ea-enc] 4x module (4 kernels: E4/E5 x w13/w2) built in 52 s concurrently with ea4x 56 s, ea6 51 s and es1a 57 s. The join wait was 0.0 s and the phase started at 150 s against EAE_JOIN_S 220 from t0. Adding E4XL/E5XL x w13/w2 doubles that module's instantiations to 8, about 90-110 s under the same CPU contention. That still fits, but it eats about half the 70 s slack. Put the XL instances in their own module and thread ('4xl'), not in the 4x module, so a slow XL build is skipped and does not take E4/E5 down with it. The race grows from 1.8 s to about 4-5 s, well inside EAE_END_S 255. Size: 948,330 B plus consume_lt (~60 lines), the XL loop (~90 lines) and the race/selector glue comes to about +9-12 KB, roughly 960 KB, under 1,000,000 B. The post-norm swap reuses ea_combine, so it adds about 2-3 KB.

### Confounders on the current kernel
- engine-a-a4b-a5b-encodings-race (landed, bbd1661f) + eae-cold-iso-keepalive-fix (f2b510f5): XL is built on the E4/E5 bank copies and the [ea-enc] race. Any XL reading is relative to the E arm the race would otherwise pick, so the per-class line must print XL/E4 and XL/E5 directly, not only XL/m. The keep-alive fix closed a prepare-only use-after-free in that same race. Adding arms and banks adds lifetimes, so the UAF class can come back (sticky illegal address = whole run fails, research/h100-glue-1029.md side finding).
- v4x256c / W6-M fork (T=256 hybrid, hot half wh5ds on the fork stream): how much an XL cold-half gain is worth at T=256 depends on how much of the cold half sits on the critical path next to the hot half. A gain or loss at verify_32 can come from fork/hot-half overlap, not from XL. The isolated cold-half ratio the hypothesis already prints separates the two.
- hothalf-wh-a5b-port (7da461db): it shortened the hot half, which pushes the cold half onto the critical path. This raises XL's T=256 payoff relative to the node (where the cold half was off-path), so expect judge > node at 256.
- carrier (class-128 CUDA ea_combine) and the T=256 hybrid's join: the post-norm swap puts ea_combine with PDL right after a cross-stream event join. Its gain depends on which kernel is the programmatic predecessor (hot w2 on the capture stream) and on whether the cold half finishes last on that draw.
- stack2 EPLF (fixed-role fix-up, T16): not a confounder on today's routes. The judge routes T=16 to v4xs (582bb334 log line 119), and _eae_base admits only v4x s twins, so XL never stacks with EPLF. If a route change put EPLF back at T16, XL would silently not apply there.

### Interference and A/B needs
No companion is needed. Both sub-changes are raced in-process against master with per-arm, per-round prints, and that race is the A/B. XL vs E4/E5 vs m comes per T class from the [ea-enc] lines (with the cold-iso ratio at 256). The post-norm swap is its own arm at T=256 through kernel_fn. A separate control run would only repeat what the prepare race measures on the same draws. The one ambiguity the race cannot settle is whether a verify_32 headline move comes from XL's cold half or from the post-norm swap. That can be split by having the T=256 race print the four cells {E, EXL} x {post-norm, ea_combine} rather than racing the two independently. This is a print requirement, not an extra judge run.

### Open questions
- Build must print: (1) [ea-enc] build per module with seconds, separate for 4x / 4xl / 6, plus join wait and phase-at time. (2) regs/lmem/occ for K1e4xl, K1e5xl, K2e4xl, K2e5xl. Gate each arm on lmem 0 and occ 2. The probe read 92-94 regs on cu13.0, close to the 96 cap of __launch_bounds__(THREADS, 2), so a different nvcc on the judge could spill. (3) Per class T=8/16/32/128/256: exact h/y per arm, E/m per round for m/E4/E5/E4XL/E5XL, and the direct XL/E ratio. (4) At T=256: cold-iso ratios for all four E arms, the {E, EXL} x {post-norm, ea_combine} kernel_fn cells, and the canary rel-L2 / worst row of ea_combine vs master's. (5) PATH CHOSEN per T. (6) submission size in bytes.
- Selection rule among several passing arms: the race picks on worst round today. With XL vs E4 differing by about 1-2% and a round spread of about ±1%, say whether ties (|XL/E - 1| < 0.5%) default to the incumbent E arm, which is the safer choice.
- At T=256 the probe showed E5 beats E4 consume-only by 1-2.6%, yet the judge picked E4 at 256 (582bb334: kernel_fn E4 0.975 vs E5 0.980, cold iso 0.964 vs 0.961). Race E4XL and E5XL both at 256. Do not pre-select E5XL from the node numbers.
- Post-norm bar: a true judge ratio of about 0.99 against a <= 0.99 every-round bar will mostly fail. Either accept that the post-norm half is likely a no-op (fine, about 0.2% at most) or gate it on mean <= 0.995 with worst <= 1.000. Decide before the run, not after reading it.
- Verify that the 4xl module's pdl selector bit does not collide with existing EA_PDL_BITS (256, 512, 4096, EA4C_CW << 4) and that _eae_info / ok{} are widened from 4 to 8 rows. Today they are hard-coded to K1e4/K1e5/K2e4/K2e5.


## Implementation notes

**Mechanism:** Engine A XL arms (E4XL / E5XL = E4 / E5 + ldmatrix.x4 B preload + trimmed stream-K record loop, bitwise E) raced as extra arms in the [ea-enc] per-class race on the v4x s-twin K1 + K2 (T=8/16/32/128 routes and the T=256 hybrid's cold half).

**Scope cut (main thread, recorded in DECISIONS):** only the Engine A XL arms are built. The T=256 post-norm -> ea_combine swap (part 2 of `change`) is DROPPED: no kernel_fn or combine change, and no {E, EXL} x {post-norm, ea_combine} cells or canary. Expected rows are the XL rows only (verify_1, verify_16, verify_32, decode_16, decode_32).

What changed (worktree `muse_moe_w4a16/submission.py`, [ea-enc] block only, +231/-22):
- `_EAE_LT` (new CUDA text, namespace eak): `lds128` / `lds64` / `imad_step` and `consume_lt<DQ>`. consume_lt is re-derived from master's `consume_e<DQ, 1, 1>` (ext_a4b / mul_bf16x2_h / scale_pair_imad_rt for E4, the lop3 A5b decode for E5, X1 four-ldmatrix.x4 B preload), not from the probe's scale_enc / frag_a4b helpers. Same loads in the same order on caller-carried shared addresses.
- `_EAE_LT_LOOP`: the probe's trimmed consumer loop (eaenc2 L), ported onto eak4x's expert_body_x (anchored on the eak4x-only `pend0` / acc-init block). It runs under `if constexpr (DQ >= 3 && !DEFER)`, after master's `b >= e` empty-CTA early return, so that return is kept. It has the same flush points (super-group end or e - 1), the same ticket / finish / epi code, and consume2 for passes of more than 8 routes.
- `_eae_src4xl()`: a new module '4xl' = `_eae_src4x()` + `_EAE_LT` + the loop + a host `TORCH_CHECK(pdl & EAE_XL_BIT)` in both launchers. `_eae_start` builds it in its own thread (`eaenc-4xl`, started after 4x and 6). A slow or failed XL build is just not joined by EAE_JOIN_S (still 220), and E4 / E5 are unaffected.
- Selector: `EAE_XL_BIT = 16384` (bit 14). It is clear of EA_PDL_BITS (1, cw << 4 = bits 4..7, 256, 512) and of E5's 4096; the ea4x kernels read only bits 0 and 4..7. `EAE_ARMS` maps arm digit -> (bank copy, XL): 3 E4, 4 E5, 5 E4XL, 6 E5XL. XL reuses the E4 / E5 recoded banks (`w13e3` / `w13e4`), so there is no new weight copy. `_eae_pair`, `_eae_xcore` and `_EAE_CORES` take digits 5 / 6. The bank-drop `used` set maps through EAE_ARMS, so an XL pick keeps its bank alive. `_eae_off` also pops `extexl`.
- `_eae_info` returns every row the module reports. `ok{}` now has 8 v4x rows (4x + 4xl: K1/K2 x E4/E5 each). XL rows are admitted only with lmem == 0 and occ >= 2; E rows keep master's gate.
- Race (`_eae_run`): for 'x' and 'cold' kinds the arms are E4, E5, E4XL and E5XL (E5 / E5XL only when the scales allow A5b), at every class including T=256 (cold half). Exactness uses master's `_eae_hy` (NaN-poisoned h / y on the same draws, including the v4cxs cost-weighted splits) plus kernel_fn out before and after the timed replays plus err / tickets. Bar: `<= 0.98` vs master every round. Tie rule: the best passing E arm wins unless the best passing XL arm's per-round XL/E median is `< EAE_XL_TIE = 0.995`. XL is picked outright only when no E arm passes.
- Lifetimes: XL adds no new per-call buffers. Every captured graph and the cold-iso `h3/y3/l3` stay in `keep` through `_rt_time`, as on master.
- Prints: build seconds per module (`4x / 4xl / 6`), join wait and phase start; `regs/lmem/occ` per module, with the 4xl row labelled `K1e4xl K1e5xl K2e4xl K2e5xl`; per T, exact y/n plus ratio median[min, max] over rounds for each arm, `cold iso` for all four arms at T=256, and `XL/E` (per-round E4XL/E4 and E5XL/E5); PATH CHOSEN per T.
- Deviation: no runtime print of the submission's size. It would need a stat of `__file__`, which I left out to avoid tripping the harness-I/O rule. Size: **959,292 B** (< 1,000,000).
- Review fixes applied: the port is non-mechanical (consume_lt from consume_e); a new non-colliding pdl bit; 8 info / ok rows; XL only (no X), in its own module / thread; EAE_JOIN_S 220; lmem 0 / occ 2 gate; E4XL and E5XL both raced at 256 (no E5XL pre-select); the tie rule defaults to E; the 4x / 4xl / 6 build print. The review's post-norm questions are moot under the scope cut.

## Rules check

- [x] CUDA-graph capturable: kernel_fn's path changes only which ext / bank / pdl bits `_eae_pair` / `_eae_xcore` pass (dict lookups on the route string set in prepare). The diff adds no sync, `.item()` or input-value branch on the timed path. The new syncs / `.sum()` reads are in the prepare-only race, as on master.
- [x] Nothing derived from `data` is cached across calls: the XL arm is chosen in prepare from synthetic draws; no input-derived state.
- [x] Weights: XL reads master's existing lossless E4 / E5 recoded copies (bit permutation of the NVFP4 codes, A5b scale fields). No new copy, no precision change.
- [x] Activations BF16 where SGLang keeps BF16: same mma bf16 x bf16 -> fp32, same epilogue and same h / y bf16 as E (bitwise, gated exact).
- [x] n/a: attention (moe problem); every local route is computed.
- [x] No timing patches, harness frames / I/O, module edits, reference import, subprocesses or network: the extra module goes through master's `_rt_build` path like 4x / 6. No process or os calls were added.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
OK with warnings: .worktrees/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack/muse_moe_w4a16/submission.py
(warnings: master's pre-existing prepare-only host-sync / uint8-dtype notes; none new on the kernel_fn graph)
```
py_compile OK; __pycache__ removed.

### Node validation (gpunode06 GPU 1, nodejudge `xlmoe-val`, cold builds, budgets enforced, 2026-10-30 00:41Z)
Bytes sha256 091c05b8670a... (959,292 B). Prepare took 131 s of 300; all 5 TASK tests passed (rel L2 4.1e-3..5.6e-3); 6 shapes x 6 reps ran clean. SGLang was NOT timed: the config derived from hhwh-moe has `baseline: null`, so the run has no headline.
```
[ea-enc] build 4x 37s, 4xl 37s, 6 37s (bg from 0s), join wait 0.0s, phase at 129s
[ea-enc] regs/lmem/occ 4x K1e4 82/0/2 ...; 4xl K1e4xl 94/0/2 K1e5xl 92/0/2 K2e4xl 94/0/2 K2e5xl 92/0/2; ... E5 on
[ea-enc] T=8 v4xs: E4 y 0.989[0.984,1.004] E5 y 0.996[..] E4XL y 0.985[0.976,0.999] E5XL y 0.994[..] | XL/E E4XL 0.995 E5XL 0.996
[ea-enc] T=32 v4cxs: E4 y 0.992 E5 y 0.991 E4XL y 0.988[0.965,1.004] E5XL y 0.994 | XL/E 0.997 / 1.000
[ea-enc] T=128 v4cxs: E4 y 0.997 E5 y 0.991 E4XL y 0.991 E5XL y 0.988[0.969,1.004] | XL/E 0.997 / 1.001
[ea-enc] T=256 v4xs: E4 y 0.995 E5 y 0.993 E4XL y 0.991[0.980,0.993] E5XL y 0.991 | cold iso E4 0.993 E5 0.987 E4XL 0.988 E5XL 0.984[0.976,0.991] | XL/E 0.996 / 0.996
[ea-enc] PATH CHOSEN all T master (no arm <= 0.98 every round; T=16 route v4xe, T=1 v1) (race 2.4s, prepare at 131s)
```
Every XL arm is exact at every class. XL/E is 0.995-0.997 at T8/32/128/256 for E4XL and at 256 for E5XL. On the node no arm, E or XL, clears the 0.98-every-round bar, so the node picks master everywhere, the same as master's own E arms on the node. Results: scratchpad `build-moe-xl/hp/prototypes/hopper-probe/runs/20261029-174132-gpunode06/out/`.

## Result

- job `73156455c9f242f6` (check), status done, passed True, tests 5/5
- speedup 2.0667x, 95% CI [2.0357, 2.0977], score 6507.344822526427 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 2.035180868162415, "ci": [2.0046531551399784, 2.065708581184851], "job": "575820c989c8412b"}; base result: muse_moe_w4a16/hypotheses/BEST.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/engine-a-xl-arms-t256-postnorm-stack.result.json`, `muse_moe_w4a16/hypotheses/done/gain/engine-a-xl-arms-t256-postnorm-stack.patch`; branch `hyp/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 28.02 | 28.93 | 1.0061 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 102.96 | 100.91 | 1.0154 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 132.13 | 132.03 | 1.004 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 76.62 | 71.96 | 1.0535 | gained |
| bs=128 load=verify_16 sharding=ep8 | 60 | 220.72 | 222.5 | 1.0117 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 264.5 | 263.2 | 1.0029 | flat |

gained on verify_1 verify_1. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

