---
problem: muse_decode_layer
slug: hothalf-wh5ds-moe-port
created: 2026-10-29T202838Z
base: master
mint_blob: "315119e26cb18e755adf23c3336af2c15c72aafc"
best_at_mint: 1.6904
area: moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
change: >
  Port moe's landed wh5ds hot half back to decode: take the module from
  muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms D0=0,
  a3=1, wh=7, wh5ds=14 = A3 + A5b code re-encoding / 7-bit packed scales + per-stage descriptor +
  split commit on the wh exact-N/ring kernel; isolated-pair route gate <= 0.97 vs master and D0
  every round, wh5ds also <= 0.98 vs wh; production no-loss gate) and decode's own routed wh
  module (branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wh routed T=128
  and T=256). Replace decode's wh-only arm set with D0/a3/wh/wh5ds, route per T class (128 and
  256) by the isolated pair, plus a no-loss gate on decode's layer timing at each class (wh/m
  layer ratio <= 1.00 every round, as decode's 44f30364 FORK lines measured). Keep decode's fork
  setting (CAMPAIGN_HCF False), routing, cold half and PDL unchanged. Prints as in 44f30364
  ([hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK/PATH CHOSEN, <= 1.5 KB at the end of prepare). Size:
  decode is 955,413 B on master; 44f30364's wh module brought it to 984,021 B; the extra wh5ds arm
  must still fit under 1,000,000 B (trim a3nx and dead diag text).
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "moe judge: wh5ds pair 92.8 vs wh 98.9 vs master 115.1 us; decode 44f30364 wh verify_32 sliding hot pair 153.5 -> 122.8 us; H100 hh3 wh5ds vs wh x0.986 forked"
  - shapes: load=verify_16*
    direction: faster
    why: decode T=128 wh pair .90 (44f30364)
  - shapes: load=decode_*
    direction: same
    why: T<=32 do not use the hot half
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "hothalf-a3-exactn-ring-wh: same routed wh plus the wh5ds arm (A5b+D+SC) that landed on moe, and a no-loss layer gate"
target: check
value_gain: 1.53
value_p: 0.5
value_ev: 0.77
value_cost: 300k tokens, 1 judge run
evidence: >
  moe landed a5318a1f 1.9748 (check 54b70249 1.9827: verify_32 291.1 -> 274.4 us, pair wh5ds 92.8
  vs master 115.1); decode 44f30364 wh routed T=128/256, verify_32 sliding hot pair 153.5 -> 122.8
  us, row 427.7 -> 403.2; H100 hh3: decode +1.32% (wh) -> +1.53% (wh5ds)
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T203620Z
review_companions: null
review_best: 1.6903955680487197
review_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
stage: implemented
claimed_at: 2026-10-29T203620Z
base_sha: 906305a67ac68840ccfdb0d90d3edf861f6b9536
base_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
branch: hyp/muse_decode_layer/hothalf-wh5ds-moe-port
worktree: .worktrees/muse_decode_layer/hothalf-wh5ds-moe-port
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-29T212726Z
impl_commit: 620b3802854f1e237e8fbc87ce1fee4de846250a
submission_sha256: 4e9ab0d31a26d7d51a730baf98fa45b709b0fba873009602a4e4146aaa9cc2df
norm_sha256: 6b2ad167c047e8dd7372255093ed087c6dbfecf2c08698254493a791247b0f04
submission_blob: 7bb37a39e11419ee04a23303c491d0feccc95576
lint_warnings: 209
---

# hothalf-wh5ds-moe-port

## Change

Port moe's landed wh5ds hot half back to decode: take the module from muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms D0=0, a3=1, wh=7, wh5ds=14 = A3 + A5b code re-encoding / 7-bit packed scales + per-stage descriptor + split commit on the wh exact-N/ring kernel; isolated-pair route gate <= 0.97 vs master and D0 every round, wh5ds also <= 0.98 vs wh; production no-loss gate) and decode's own routed wh module (branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wh routed T=128 and T=256). Replace decode's wh-only arm set with D0/a3/wh/wh5ds, route per T class (128 and 256) by the isolated pair, plus a no-loss gate on decode's layer timing at each class (wh/m layer ratio <= 1.00 every round, as decode's 44f30364 FORK lines measured). Keep decode's fork setting (CAMPAIGN_HCF False), routing, cold half and PDL unchanged. Prints as in 44f30364 ([hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK/PATH CHOSEN, <= 1.5 KB at the end of prepare). Size: decode is 955,413 B on master; 44f30364's wh module brought it to 984,021 B; the extra wh5ds arm must still fit under 1,000,000 B (trim a3nx and dead diag text).

## Expected

- `load=verify_32*` faster: moe judge: wh5ds pair 92.8 vs wh 98.9 vs master 115.1 us; decode 44f30364 wh verify_32 sliding hot pair 153.5 -> 122.8 us; H100 hh3 wh5ds vs wh x0.986 forked
- `load=verify_16*` faster: decode T=128 wh pair .90 (44f30364)
- `load=decode_*` same: T<=32 do not use the hot half

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** moe landed a5318a1f 1.9748 (check 54b70249 1.9827: verify_32 291.1 -> 274.4 us, pair wh5ds 92.8 vs master 115.1); decode 44f30364 wh routed T=128/256, verify_32 sliding hot pair 153.5 -> 122.8 us, row 427.7 -> 403.2; H100 hh3: decode +1.32% (wh) -> +1.53% (wh5ds)
- **Gain if it works:** 1.53% headline; **p** 0.5; **EV** 0.77% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.53% < 2.5%; EV 0.77% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge run.

## Risk

Decode's wh check 44f30364 routed and moved the verify_32 sliding hot pair x0.80, but the headline read flat (1.6794 vs 1.6904) on a noisy draw; moe's wh5ds then landed (54b70249 1.9827, scored a5318a1f 1.9748). Read this on the [hh-wh] lines and the wg_kernel<G,14> profile rows first, headline last.

## Adversarial review

Reviewed 2026-10-29T203620Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: fresh (blob 315119e2 = mint, BEST 1.6904 unchanged). The mechanism is [shelf] (CLAIMS:99), whose reopen is 'ride on the next decode candidate'; running it standalone is waived by today's user steer (user-requested, below-bar, H100-derived first). Proceed with no companion, but re-derive the value case and change the readout. 44f30364 read in per-load speedup terms (the headline's own arithmetic) gave wh only +0.17% headline, not the ~+1.15% the row-us reading suggests: SGLang's in-run time also fell x0.959 at verify_32. A realistic gain is ~+0.8-1.1%, so p(beating the 1.24% margin) is ~0.3.

### Justification
A. The mechanism is sound and judge-proven where it matters. moe's wh5ds pair (wg_kernel<G,14>) read 92.8 vs wh 98.9 vs master 115.1 us on the judge (54b70249). moe's verify_32 load speedup went 2.121 -> 2.264 while SGLang's own time stayed flat (x1.007), so +1.08% headline came from T=256 alone: a clean transfer. Decode's 44f30364 proved the wh half in-run on the judge: bit-exact 16/16, isolated pair wh/m .898-.902 (T=128) and .838-.842 (T=256), whole-layer wh/m .973-.976 and .960-.979 under the real ea4x cold half. hh3 adds A5b+D+SC on top at x0.987/0.988 forked (node) and x0.967-0.973 isolated. WEAKEST LINK: the value case. value_gain 1.53% comes from hh3's H100 stand-in elasticities. Decode's own judge evidence says less. (1) The in-run FORK lines of 44f30364 put wh at T=256 -12.6 us and T=128 -8.4 us of layer time, which the same elasticities turn into ~+0.8%. wh5ds adds ~+0.2-0.4% (hh3 +0.21%; moe's judge pair wh5ds/wh .938 is better than the node's .973). Expect ~+1.0-1.1%, under the 1.24% margin. (2) The judged rows show even less once read as the headline computes it: per load, SGLang's in-run time over yours, geomean over 6 loads. In 44f30364 vs board 7cc034f9, the verify_32 load had SGLang x0.959 and yours x0.945, so the load speedup went 1.636 -> 1.659 (+0.24% headline). The verify_16 load had SGLang x0.984 and yours x0.988, so its speedup went 1.626 -> 1.619 (-0.07%). The 'verify_32 sliding 427.7 -> 403.2 us' reading (LOG 1954Z, CLAIMS:99) mostly measured a faster GPU draw: the baseline sped up 4% on the same rows. Net judged wh effect: +0.17% headline. The other four loads cost -0.77%: decode_1 -0.29%, decode_16 -0.38%, verify_1 -0.22%, decode_32 +0.07%. That is why the result read flat. T=128 shows ~0 judged transfer despite pair .90 and layer .975 in prepare. This fits CLAIMS:69 (prepare-time margins do not track judged rows) and the T=128 cold half being ~2-2.8x the hot. Expected shapes and directions match TASK: only verify_16/32 (T=128/256) run the hybrid, and decode_*/verify_1 should be flat in expectation. The prior field is honest: CLAIMS:99 is the same routed wh, now with the wh5ds arm. Nothing on FINDINGS' Do-not-try list blocks it. hothalf-a3-exactn-ring-wh is listed there as within-ci, but CLAIMS:99's newer [shelf] line supersedes that listing and names exactly this follow-up. Q (verify_16 profile, hot pair 69.6 -> 92.8 us vs in-run race .90): no red flag for the kernel. Per-kernel GPU time in this profile includes PDL and overlap waits (the profile footer says so; CLAIMS:53 / DISTILL:266 refuted sizing from such rows). Three readings of the same run show it. (a) In that same verify_16 sliding row, the cold ea4x pair fell 197.1 -> 130.9 us, so hot+cold fell 266.7 -> 223.7 while the row fell only 3.8 us. (b) On untouched rows, verify_1 full ea4x went 73.5 -> 107.6 us with the row flat (159.3 -> 158.5). (c) At verify_32 sliding, the cold pair rose 140.9 -> 182.7 while the hot pair fell 153.5 -> 122.8. So the hypothesis's own cited 'x0.80 hot pair' evidence is equally unreadable. The profile is also a single routing draw at T=128, where the hot set (> 16 routes per expert) varies a lot per draw. It is a yellow flag in one sense only: together with the flat verify_16 load speedup, it says T=128 bought nothing judged in 44f30364. Also, 92.8 numerically equals moe's T=256 wh5ds pair (92.8 us); the two are unrelated quantities, so do not conflate them.

### Confounders on the current kernel
- v4xd-stack (CLAIMS:95; EAD4X_BOARD {16: 'd'} at submission.py:13976; arms at :12243). The T=16 ea4x pooled-hysteresis race drew v4cxs (ea4x_*_kernel<2,false>) on the board run 7cc034f9 and v4cxd (<2,true>, pdl & 256) in 44f30364 ('16:d>s D1.010 PASS(8) pooled near 6r'). decode_16 sliding moved 161.1 -> 169.5 us, and the decode_16 load cost -0.38% headline. This change does not touch that race, but it decides the headline as much as the hot half does. Read the [ea4x-dl] route line before the headline.
- v4xd-stack fork guard / W6-M side-stream fork (CLAIMS:94-95, _sfk_gate). It sets how much of the hot cut surfaces through the fork. The wh phase runs before _sfk_gate, so the no-loss layer gate must hold in whichever mode _sfk_gate then picks. In 44f30364, fork and serial wh/m agreed within .002, so requiring both costs nothing.
- pinned trim H-16 at T=128/256 (CLAIMS:69). It fixes the hot set, so it fixes the N distribution that exact-N and SC (N <= 32 only) see. Its lesson (prepare-time layer margins do not track judged rows) means the layer gate may only veto. It must not choose between wh and wh5ds.
- Engine A v4x port (CLAIMS:93): the T=128 cold half is v4cxs at ~2-2.8x the hot pair. This is why T=128's hot cut hides (verify_16 load flat in 44f30364) and why T=128 is unlikely to add judged gain.
- W6-B diet cells and W6-G fast silu in wg (CLAIMS:64, 68): they share the wg epilogue and the T>=128 launch sequence. Exact-N and SC change loop bounds and commit grouping. The torch.equal exactness gate on every routed class (as in 44f30364 and 54b70249) covers this; keep it.
- hothalf-a3-exactn-ring-wh (CLAIMS:99, branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh): this is the port's base for decode-specific plumbing, but it is not on master. This check therefore lands wh and wh5ds together, and one headline cannot split them. Only the in-run ROUTE line's wh5ds/wh pair ratio can.

### Interference and A/B needs
C. No companion. The two classes the change touches are two different loads (verify_16 = T=128, verify_32 = T=256), so the per-load speedups attribute the result per class directly. The in-run [hh-wh] lines (D0 fidelity, wh/m, wh5ds/wh, layer gate per class) separate kernel transfer from fork hiding. An ab-revert of any landed item would not sharpen a ~+1% effect against ~+-2% per-load drift. A T=256-only control (T=128 forced to master, as moe routed) is the one candidate. It is not warranted, because the per-load reading already isolates T=128. What the build and readout must do instead: (1) READ PER LOAD, as the headline does. Compute load speedup = in-run SGLang load time over yours, for verify_16 and verify_32, vs board 7cc034f9 (1.626 / 1.636) and vs 44f30364 (1.619 / 1.659). Do not read row us; SGLang's own time drifts x0.95-1.01 per load between runs. (2) Before the headline, report the T=16 ea4x route (board drew s; d cost ~-0.38% in 44f30364) and the T=1 draw (CLAIMS:unresolved T=1 lucky-draw line). (3) Use profile rows only to show that wg_kernel<G,14> (or <G,7>) ran per row, never for time (CLAIMS:53). (4) The layer gate is a veto only. The chosen arm's whole-layer ratio vs master must be <= 1.00 in every round, in both serial and fork modes. If wh5ds fails, fall back to wh if wh passes, else master. Never choose between wh and wh5ds on layer time; choose on the isolated pair (wh5ds <= 0.98 vs wh, every round, as moe did). Decision rule: a scored run only if the check beats 1.6904 by the 1.24% margin. Otherwise append a CLAIMS line with the per-load numbers: verify_32 load speedup up >= ~2% over the board with SGLang flat = T=256 transfer holds (carry it as a rider); verify_16 flat again = mark T=128 [unresolved] or refuted for the judged headline.

### Open questions
- Size: master is 955,413 B. 44f30364's wh module added 28,608 B net (984,021). moe's full wh/wh5ds module section is ~34.3 KB (muse_moe_w4a16/submission.py from '# ---- hothalf-wh-a5b-port'). A naive port lands near ~990-991 KB, over the 990,000 campaign line (DECISIONS:455), though under the 1,000,000 okc cap. Plan the trims up front so the result is <= ~988,500 B and leaves room for later riders (forkguard and the like): drop a3nx (already attributed in 44f30364); shrink the 2x2 FORK diagnostic to the veto gate (master vs routed arm, both modes); cut the step markers and dead diag text. Is the a3 arm still worth keeping as anything but a fallback? D0 is the only control that matters.
- T=128: 44f30364's verify_16 load speedup was flat to slightly down (1.626 -> 1.619) with wh routed, despite pair .90 and layer .975 in prepare. moe kept T=128 on master. Should decode route T=128 at all, or carry it only if this run's verify_16 load speedup rises over the board with SGLang flat? Keeping it is cheap (bit-exact, no-loss veto). If it reads flat twice, retire it.
- Does the judge's SGLang per-load drift (x0.959 at verify_32 in 44f30364; x0.949-1.008 across loads in moe) make any single-check verdict on a ~1% decode lever meaningful? If this reads flat again, the next read should be the scored-run pair or a rider on a larger decode candidate (CLAIMS:99's own reopen), not a third standalone check.
- Should the T=16 ea4x race be pinned to the board's draw (v4cxs) in a separate determinism change? It cost ~-0.38% headline in 44f30364, more than this whole lever's T=128 share. That is out of scope here; keep this check's routing unchanged so it stays a clean port.
- moe's no-loss gate timed kernel_fn T=256 with the fork as routed. Decode's analog is _ead_layer_gate per T class, in both _sfk_gate modes because the fork route is decided after this phase. Confirm that the builder runs the gate after the wh5ds/wh pair pick and gates the picked arm, not each arm.


## Implementation notes

**Mechanism:** moe's landed wh/wh5ds wgmma hot half (wg_kernel<G,7>/<G,14>: A3 or A5b dequant + exact-N + A ring + per-stage descriptor + split commit) routed per T class at decode's T=128/256 hybrid hot half (_hc_experts), with the pick made on the isolated pair and the layer gate used only as a veto.

- Built 2026-10-29T2126Z in `.worktrees/muse_decode_layer/hothalf-wh5ds-moe-port/muse_decode_layer/submission.py`. Size **988,094 B** (master 955,413; +32,681). Diff: 1 file, +685 / -5.
- **Kernel text:** `_WH_CPP`, `_WH_DQ`, `_WH_RING`, `_WH_NX_SWITCH`, `_WH_EDITS` and `_WH_MAPS` are copied verbatim from muse_moe_w4a16/submission.py on master. I applied `_wh_src` to decode's `_WG_SRC_HEAD/_BODY` offline via AST: every asserted edit target matched exactly once, and the generated CUDA TU is **byte-identical** to moe's (48,422 B). That is the module that compiled and ran on the judge (54b70249/a5318a1f), with the same register and spill attributes. I made one change from moe: the ext name is `muse_decode_wg_wh5_<sha10>`. Arms: D0=0, a3=1, wh=7, wh5ds=14. a3nx is dropped.
- **Plumbing** comes from decode's 44f30364 branch:
  - `prepare()` → `_wh_start` (a non-daemon nvcc thread; TORCH_CUDA_ARCH_LIST=9.0a, restored in `finally`), then `_prepare_main`, then `_wh_join(restore=True)`. The build thread is joined before `prepare` returns.
  - `_wh_phase` runs before `_sfk_gate`.
  - `_wg_wgmma_src(nbs=WG_BUCKETS)` gains a parameter.
  - `_hc_experts`' wg hot half calls `_wh_pair(state, whk[_hc_class(T)], ...)`. This is the only dispatch site. `_wh_pair` with v=None runs master's two `_wg_gemm` calls, unchanged.
- **Route (per class, review fix 4):**
  - Exactness: torch.equal on 2 router draws plus 4 all-N synthetic routings, output and hs, every arm. The err word must be clean.
  - Timing: an isolated-pair graph race of m/D0/a3/wh/wh5ds over 2 inputs × rounds. D0 must be exact and D0/m must be within 0.97..1.03.
  - Candidates are wh5ds and wh. Each must be ≤ 0.97 vs m and vs D0 in every round; wh5ds must also be ≤ 0.98 vs wh in every round. Priority is wh5ds, then wh. a3 is a candidate only when neither passes.
  - The first candidate that passes replay 16/16 is picked.
- **Gate (veto only):** `_wh_gate` → `_ead_layer_gate` on {m, picked} × {serial, fork} (`side_serial`). Every round must be ≤ 1.00 in both modes. If wh5ds is vetoed, wh is tried (replayed first if it had not been) and gated. Otherwise the class stays on master. The gate never chooses between arms.
- **Unchanged:** master routing, the T=16 ea4x race, cold half, PDL, CAMPAIGN_HCF, and all of kernel_fn except the hot-half call.
- **Kept:**
  - the scale scan (0x08..0x7E)
  - the per-chunk inverse checks for the A3 permute and A5b ENC4/pack copies
  - `_wh_selftest`
  - the attrs check (regs×384 ≥ 168-reg pool, lmem 0, sm_90 image)
  - the 240 s phase guard (checked before every step)
  - `_EAD_INPUTS` trimming
- Bank copies that end up unrouted are freed: whk_x5 unless a routed v has bit 8, whk_bank unless one has bit 1.
- **Prints:** at the very end of prepare (after the SFK end line), ≤ 300 chars each. They are:
  - BUILD, ATTRS (4 arms × w13/w2), SCALES
  - ROUTE T=128/256 (wh/m, wh/D0, wh5ds/wh per round; a3/m max; D0/m range; us; exactness; replay)
  - GATE T=… (picked arm layer/m, s and f, per round; PASS/VETO; fallback)
  - PATH CHOSEN with the `wg_kernel<G,V>` profile symbol
  - the last `[ea4x-dl] route (board …)` line from EA4X_LINES, re-printed

  Estimated total is ~1.4 KB.
- **Deviations:**
  - (a) The per-step `> step` marker prints are cut, as MUST 1 asks (this overrides BUILD_BRIEF's announce-each-launch rule). `step()` still syncs before each step and records its name, and every OFF/raise line reports `at <step>`, so a fault still names its step.
  - (b) The 2x2 FORK diagnostic is replaced by the veto GATE line (master vs picked arm, both modes).
  - (c) T=128 still routes when its isolated pair qualifies. The review left this open; the route is cheap, bit-exact and vetoed.
- **Not validated:** no GPU, torch or nvcc on this node. The optional H100 node validation was skipped because the coordinator's 20-minute limit was too short for a full decode prepare. The new Python is unexercised. The main risks are in `_wh_phase`'s new route/gate control flow, especially the gate's graph capture of kernel_fn with `whk[cls]` set to 14 under fork mode.

## Rules check

Each item below was checked against `git -C .worktrees/muse_decode_layer/hothalf-wh5ds-moe-port diff`.

- [x] CUDA-graph capturable: kernel_fn's only change is `_hc_experts` calling `_wh_pair`. That function chooses the arm from a dict keyed by `_hc_class(num_tokens)`, which is shape-only. It makes no `.item()`/`.cpu()`/sync calls and allocates nothing new. All syncs, `.tolist()` and `int()` calls are in prepare-only `_wh_*` helpers; lint marks them "outside kernel_fn's call graph".
- [x] Nothing derived from `data` is cached across calls. `state["whk"]`, `whk_bank`, `whk_x5`, `whk_ext` and `whk_err` are set in prepare from the weights only. The `_EAD_INPUTS` additions are trimmed back.
- [x] Weights are repacked losslessly in prepare only. The A3 copy is a code-bit permutation; the A5b copy is an ENC4 code permutation plus 7-bit (b-8) scale packing that is valid only when every scale byte is in 0x08..0x7E (scanned). Each chunk is checked by its inverse, and the dequant equivalence is self-tested. NVFP4/int4 stays 4-bit and BF16 stays BF16.
- [x] Activations: BF16 in and out, unchanged. The outputs are torch.equal to master's on 6 routings plus 16 replays, or the class stays on master. KV and queries are untouched.
- [x] n/a: attention is untouched (MoE hot half only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds no subprocess, os.fork, multiprocessing or spawn: the build is a `threading.Thread` around `load_inline`, as on master's moe.
- [x] All work happens inside the timed call; there is no output caching or replay. kernel_fn computes the hot half fresh on every call.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a is set for prepare and restored in `finally`. WG_CUDA_CFLAGS (`-lineinfo`, WG_GENCODE) are reused. The attrs check requires the sm_90 image, 168 regs and 0 lmem, and any miss falls to master. Waits use master's bounded `mbar_wait` (err/dead).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
OK with warnings: .worktrees/muse_decode_layer/hothalf-wh5ds-moe-port/muse_decode_layer/submission.py
(209 warnings: master's 202 + 7 new `host-sync ... outside kernel_fn's call graph; fine in prepare()` in _wh_phase/_wh_replay; no errors)
```
py_compile: OK; __pycache__ removed.
