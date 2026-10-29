---
problem: muse_moe_w4a16
slug: engine-a-a4b-a5b-encodings-race
created: 2026-10-29T215527Z
base: master
mint_blob: "37ea90fc306c0ed8fb3d49094bf5f4acf6202307"
best_at_mint: 1.9748
area: Engine A dequant (dq2) in ea4x/v4x/ea6 kernels at T=1..32 and the T=128/256 cold half
change: >
  Add two lossless dequant arms to moe's Engine A kernels (ea6 K1/K2 at T=1; ea4x / v4x-family w13
  and w2 at T=8/16/32; the cold-half ea4x at T=128/256), ported from takemyjob hopper/ea-enc-1029
  d6f55c5 (probe njenc; research/h100-ea-enc-1029.md): E4 = A4b code re-encoding of the 4-bit
  codes in a prepare-time permuted copy of the bank (nibble->bf16x2 9 ops/word vs 11), E5 = E4 +
  A5b packed 7-bit (b-8) scale fields (one LOP3 per step; requires every scale byte in
  [0x08,0x7E], scanned in prepare). Both are value-identical to master's dq2 (bit-exact). Compile
  them as template variants of the existing kernels in a separate background-built extension (or
  as template args in the existing ones if the build budget allows), and route PER CLASS (T=1, 8,
  16, 32, 128-cold, 256-cold) by an in-prepare interleaved race through kernel_fn (read-flushed,
  like the existing class races): pick E4 or E5 only if torch.equal to master on every draw, <=
  0.98 vs master in every round, and replay-exact; else master. Expected picks from the node: E4
  at T<=32 and T=1, E5 at the cold half. Keep master's routing, fork, hot half (wh5ds) and PDL
  unchanged. Print an [ea-enc] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill per
  arm, scale scan, per-class per-round ratios E4/m and E5/m, exactness, PATH CHOSEN. Size: moe is
  993,167 B (cap 1,000,000): trim dead diagnostics first; report wc -c.
expected:
  - shapes: load=decode_16
    direction: faster
    why: node consume-only B/E4 T16 w13 1.067-1.090, w2 1.05-1.07
  - shapes: load=decode_32
    direction: faster
    why: T32 w13 1.079-1.086
  - shapes: load=decode_1
    direction: faster
    why: ea6 K1 consume 1.076-1.097
  - shapes: load=verify_*
    direction: faster
    why: cold half E5 1.05-1.07 consume; T8 1.05-1.08
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "cuda-nvfp4-swapab-mma-persistent-small-m-childcheck: the mma.sync small-m lineage Engine A grew from; this changes only the dequant's instruction form (lossless code/scale re-encoding, bit-exact), not the schedule, ring, prefetch or engine that the matched refuted claims tried"
target: check
value_gain: 1.5
value_p: 0.4
value_ev: 0.6
value_cost: 450k tokens, 1 judge run
evidence: >
  H100 gpunode06 GPU 3 (user-approved; runs/20261029-144741-gpunode06), 2 replicates x 10 rounds,
  consume-only: B/E4 T8 1.05-1.08, T16 w13 1.067-1.090, T32 w13 1.079-1.086, cold 1.04-1.05, ea6
  K1 1.076-1.097; B/E5 cold 1.05-1.07; exact at every draw; SASS -9..-12% issued/k16
  (research/h100-ea-enc-1029.md)
user_requested: true
review: proceed
status: done
review_at: 2026-10-29T215954Z
review_companions: null
review_best: 1.974770471501011
review_blob: 37ea90fc306c0ed8fb3d49094bf5f4acf6202307
stage: harvested
claimed_at: 2026-10-29T215954Z
base_sha: 973f86b3744c4b9b1d60469442fad05d8fca5783
base_blob: 37ea90fc306c0ed8fb3d49094bf5f4acf6202307
branch: hyp/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race
worktree: .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T221730Z
impl_commit: dca4f5e7c93735ed35e7952bccc04083e934e681
submission_sha256: 09b11a9d34b39697f9ed500d36a60435225866e74413186fcf5042ca67eb0db7
norm_sha256: d022c3f90a477239ff2f409af770cbd18ffffdfbbe474d16151598703e185230
submission_blob: c6e41c9acc6169ac8a532684437955b1d560835b
lint_warnings: 204
attempt_file: 2026-10-29T221730Z-check-engine-a-a4b-a5b-encodings-race.md
prepared_at: 2026-10-29T221730Z
job: 702f1004b45f4bfa
dispatched_at: 2026-10-29T221743Z
outcome: gain
verdict_detail: gain
speedup: 2.0645771117791423
speedup_ci: [2.033608455102455, 2.095545768455829]
harvested_at: 2026-10-29T222812Z
best_at_harvest: 1.974770471501011
promotion: merged
mechanism: null
promotion_commit: bbd1661f3d77230ea0673d67b390da4277ee5ea8
promotion_attempt: 2026-10-29T222813Z-submit-engine-a-a4b-a5b-encodings-race.md
---

# engine-a-a4b-a5b-encodings-race

## Change

Add two lossless dequant arms to moe's Engine A kernels (ea6 K1/K2 at T=1; ea4x / v4x-family w13 and w2 at T=8/16/32; the cold-half ea4x at T=128/256), ported from takemyjob hopper/ea-enc-1029 d6f55c5 (probe njenc; research/h100-ea-enc-1029.md): E4 = A4b code re-encoding of the 4-bit codes in a prepare-time permuted copy of the bank (nibble->bf16x2 9 ops/word vs 11), E5 = E4 + A5b packed 7-bit (b-8) scale fields (one LOP3 per step; requires every scale byte in [0x08,0x7E], scanned in prepare). Both are value-identical to master's dq2 (bit-exact). Compile them as template variants of the existing kernels in a separate background-built extension (or as template args in the existing ones if the build budget allows), and route PER CLASS (T=1, 8, 16, 32, 128-cold, 256-cold) by an in-prepare interleaved race through kernel_fn (read-flushed, like the existing class races): pick E4 or E5 only if torch.equal to master on every draw, <= 0.98 vs master in every round, and replay-exact; else master. Expected picks from the node: E4 at T<=32 and T=1, E5 at the cold half. Keep master's routing, fork, hot half (wh5ds) and PDL unchanged. Print an [ea-enc] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill per arm, scale scan, per-class per-round ratios E4/m and E5/m, exactness, PATH CHOSEN. Size: moe is 993,167 B (cap 1,000,000): trim dead diagnostics first; report wc -c.

## Expected

- `load=decode_16` faster: node consume-only B/E4 T16 w13 1.067-1.090, w2 1.05-1.07
- `load=decode_32` faster: T32 w13 1.079-1.086
- `load=decode_1` faster: ea6 K1 consume 1.076-1.097
- `load=verify_*` faster: cold half E5 1.05-1.07 consume; T8 1.05-1.08

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 GPU 3 (user-approved; runs/20261029-144741-gpunode06), 2 replicates x 10 rounds, consume-only: B/E4 T8 1.05-1.08, T16 w13 1.067-1.090, T32 w13 1.079-1.086, cold 1.04-1.05, ea6 K1 1.076-1.097; B/E5 cold 1.05-1.07; exact at every draw; SASS -9..-12% issued/k16 (research/h100-ea-enc-1029.md)
- **Gain if it works:** 1.5% headline; **p** 0.4; **EV** 0.6% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.5% < 2.5%; EV 0.6% < 1.5%), minted because the user described it.
- **Cost:** 450k tokens, 1 judge run.

## Risk

On the H100 every Engine A kernel is at the HBM wall in normal mode (ratios ~1.00), so only the judge (H200, estimated consumer-bound: consumer-alone 0.91-0.97x of the HBM floor) shows whether the 4-10% consume gain surfaces. The race decides per class; a class that reads level stays on master. Size is the binding constraint (993 KB). The matched refuted claims are other mechanisms (Engine S cold path, ea6 w2 prefetch, ea6 ring depth); t1-deep-ring's reopen ('the per-record consume rate is raised first') is exactly what this raises.

## Adversarial review

Reviewed 2026-10-29T215954Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: CLAIMS.md:106 [lead] is this exact mechanism, and its reopen condition (a judge check with an in-prepare exact race per class) is what this run is. Board and blob have not moved (1.9748, 37ea90fc). The run is self-controlled by its per-class race against master, so no companion is needed. The gain depends entirely on the H200, which the H100 cannot show in normal mode. Size, build deadline and the T=1 per-kernel arm choice need the notes below.

### Justification
Plausible, and supported by measured [cyc] data. In both replicates (runs/20261029-144741-gpunode06 ee-r1/ee-r2) every draw is exact (h/y torch.equal, err 0, tickets 0). Consume-only B/E4 reads T8 1.05-1.07, T16 w13 1.067-1.090, T32 w13 1.05-1.09, T128 w13 1.05; B/E5 reads T128/T256 w13 1.064-1.071. ea6 K1 c slope is 1.076-1.097. Registers drop (ea4x 93->82/83, ea6 79->70/64) with 0 lmem, matching SASS -9..-12% issued per k16. Normal mode on the H100 is ~1.00 everywhere: Bc/Bn is 0.66-0.86, so the node is HBM-walled. WEAKEST LINK: the H200 'consumer-bound' class is not measured. It is derived from a pre-L3/L4 H200 ladder rate (2,421 cyc/record) divided by bench factors, giving ~2,030-2,170 against a 2,240 HBM floor at 4.3 TB/s (ratio 0.91-0.97). By the research doc's own words that is 'mixed, at the consumer-bound edge', not consumer-bound. My cross-check gives the same answer: T16 p50 w13 Bn 96.1 us / 1.39 (H200/H100 practical BW) = ~69 us HBM against Bc 64.8 us consumer, ratio ~0.94. E4 takes the consumer to ~0.86 of the HBM floor, so a pure max() model predicts ~0. Only the coupling term the judge pays (it measured 2,390 cyc/record, above both floors) can surface the gain. History supports partial transfer: L3's bench x1.10-1.12 gave +5-7% per row, and L4's x1.066 gave +3.2% headline. Each further consumer cut transfers less, though, because the consumer is now below the HBM floor. The hypothesis's 1.5% headline at p 0.4 is honest, and I would not raise it. Second weak spot: T=1. ea6 K2 normal-mode B/E4 slope reads 0.957 [0.856,0.970] in r2 and 0.995 in r1 (E4 slower or flat). K1's normal slope is 1.062/1.064, but n3 is only ~1.01. decode_1 is also capped by its HBM floor and fixed costs. A per-class whole-arm switch at T=1 can lose K1's gain to K2's loss, so race K1 and K2 arms independently, as CLAIMS:106 itself says ('per-kernel arm choice'). Third: at T=256 p10 the normal-mode w13 B/E reads 0.968 (E4) / 0.953 (E5) in r1, so E is slower there; p50/p90 are ~1.0. That is likely noise or bank placement, but the race must cover it. Prior check: the cited prior is the mma.sync lineage, not an attempt at this mechanism, and no do-not-retry or refuted claim covers lossless code/scale re-encoding in Engine A. The difference is real. Shape labels: in master, T=128 has no hot half (hybrid-moe PATH CHOSEN T=128 = MASTER in a5318a1f's log), so '128-cold' is the whole v4cxs + CUDA combine path. Node data favour E5 at T=128 too (c 1.064-1.070 vs E4 1.043-1.051).

### Confounders on the current kernel
- L3/L4 (CLAIMS:74, :76): E4 keeps their paired scale decode, and E5 replaces it with A5b. Both are measured against master in the same run, so this is not a confounder for the A/B. It does explain why the consumer now sits below the HBM floor and further cuts transfer less.
- hothalf-wh-a5b-port (CLAIMS:105, wh5ds at T=256): the cold half runs concurrently with the hot half on the fork stream. A faster cold half shows on verify_32 only if the cold half is the critical path. A flat T=256 kernel_fn ratio is therefore ambiguous (H200 HBM-bound vs cold half off the critical path) unless the isolated cold-half ratio is printed too. The node bench also predates wh5ds (base c201dbc3 = 1.952), so the port must rebase onto 37ea90fc.
- W6-M fork / v4x256c hybrid H=-16 split (CLAIMS:55, :88): H was tuned with master's cold-half speed. A faster cold half could move the best H, but the race at the fixed H stays a valid A/B, and the gain is at worst understated.
- stack2 v6pe2 two-wave ea6 K2 (CLAIMS:97): the node's K2 instances were built on the stack2 base, so they transfer. K2's normal-mode E4 slope reads <= 1.0, so K2 may want to stay on B even where K1 takes E4/E5.

### Interference and A/B needs
No companion is warranted. The change has its own control: an interleaved, read-flushed race through kernel_fn per class against master in the same judge run, with an exactness gate. A headline gain can be attributed to whichever classes the race moved off master. A flat headline with all classes on master reads as 'the H200 does not surface the consume gain at the 0.98 bar' (the useful negative that closes CLAIMS:106). A loss can only come from a false race pick or from prepare/build cost, and both show in the [ea-enc] prints. A revert of L3/L4 or wh5ds would answer nothing this run needs. Two cheap in-run sub-controls replace a companion run: (1) at T=1, race K1 and K2 arms independently, or at least add the arms K1-E4/E5 + K2-B; (2) at T=256, print the isolated cold-half E/m ratio beside the kernel_fn ratio.

### Open questions
- SIZE (binding): master is 993,167 B against the 1,000,000 cap, leaving 6,833 B, and it is already above build.py's SIZE_BUDGET of 990,000. A separate extension must NOT carry a second copy of the ea4x/ea6 CUDA text. Build it from master's existing source strings (the same Python variables) plus a small suffix of DQ 3/4 helpers and explicit instantiations, so only the delta counts. The ext_a4b/scale_a5b/frag_a4b helpers in ea_common.cuh are likely inlined into several modules' strings, so add them only to the E extension's string. If still over, strip Python comments with build.py's AST-checked strip_comments (64.6 KB of comments in master) before docstrings (65.3 KB). Python comment/docstring stripping does not change the CUDA strings, so master's module-name hashes and kernels stay byte-identical. Prefer that to hand-trimming 'dead diagnostics', which may be load-bearing. Report wc -c and confirm master's ext names (okc_campaign_engine_a_v4x_5a6b0753a8, ea6, es1a, wh) are unchanged.
- BUILD/PREPARE BUDGET: on the judge (a5318a1f log), es1a 43.4 s, ea6 42.6 s, ea4x 45.7 s and hh-wh 57 s build in background, and prepare finishes at ~138 s of 300. A separate E extension (ea4x +4 instances, ea6 +4) adds ~45-60 s of nvcc competing for the same cores. Start it at t=0 in its own thread. Join with a hard deadline (suggest: skip the E race if not joined by ~220 s), cap the race at ~20 s, and stay master on timeout. Do not put the E instances as template args in master's existing modules: that changes their hashes and lengthens the builds master's routing already waits on.
- EXACTNESS: per class and per arm, require torch.equal of h, y and the final output against master on every race draw, on NaN-poisoned buffers, eager and graph replay. Also require err word 0 and tickets 0 after timing, plus the existing canary style (12 T x routings). Draws must include judge-like routings at T=128/256 and the p10/p50/p90 local-expert counts at T<=32, with the T=32 bits-112 pass layout.
- WHAT THE BUILD MUST PRINT ([ea-enc], <=1.5 KB, the last lines of prepare so they survive the 16,000-char tail): build wall s, join wait and prepare-at; regs/lmem/occ per E instance (expect ea4x E4 82 / E5 83, ea6 K1 E4 70 / E5 64, lmem 0); recode s and MB; the scale scan (min/max byte, count outside [0x08,0x7E], E5 eligible y/n); per class (T=1 K1 and K2 separately, 8, 16, 32, 128, 256): exact y/n per arm, median[min,max] of per-round kernel_fn ratios E4/m and E5/m, and for T=256 also the isolated cold-half ratio; PATH CHOSEN per class with the fallback reason (not exact / ratio > 0.98 in round k / deadline / build failed); and the final wc -c in the attempt file.
- Is 0.98 in every round too strict for an H200 effect expected at ~1-3% per row? It is the right guard against false picks. If every class lands at 0.98-1.00, though, read the result as 'below the bar on the H200', not as 'no effect', and say so in the result.


## Implementation notes

**Mechanism:** lossless A4b code (E4) / A4b+A5b scale (E5) dequant arms of Engine A's consumers (v4x twin ea4x_w13/w2 and FAST v6 K1/K2), built as a separate module pair from master's CUDA strings, raced per judged T (1 K1xK2, 8, 16, 32, 128, 256 cold half) through kernel_fn and routed only where exact and <= 0.98 every round.

Ported from takemyjob hopper/ea-enc-1029 d6f55c5 (patch_enc.py / encb.py / enc_check.py), rebased onto master 37ea90fc (wh5ds landed). Only `muse_moe_w4a16/submission.py` changed.

- **CUDA (no second copy of ea4x/ea6 text).** `_eae_src4x()` = `_ea4x_source()` and `_eae_src6()` = `"#define EA6_FAST 1\n" + _ea6_source()`, i.e. master's own strings, plus `_EAE_COMMON` (ext_a4b<I>, rtv) inserted into ea_common, `_EAE_CONSUME` (one `consume_e<DQ, XB, NT>`, covering consume XB 0/1 and consume2's two tiles; DQ 3 = A4b + dq2 paired scale decode, DQ 4 = A4b + A5b LOP3) before `eak::consume`, and `if constexpr (DQ >= 3)` branches in consume/consume2. The changes are anchored `str.replace` calls with exact counts; an anchor miss fails that build, and that T stays on master. Host replacements keep only the E instances: ea4x `<3,false>` / `<4,false>` (pdl bit 4096 = E5) and ea6 `<W, D, 2, true, 3|4>` (new template arg `EQ = 2`; flags bit 4096 = E5). Neither kernel reads bit 4096. I checked offline (AST-extracted strings): every anchor applies, the E sources are 101,345 B and 56,607 B, and consume_e passes `g++ -std=c++20 -fsyntax-only` against stubs. Master's 40 module-level strings are byte-identical, and the v4x name recomputes to `okc_campaign_engine_a_v4x_5a6b0753a8`; ea6, es1a, wh and stk are unchanged by construction (no string or name function touched).
- **Build:** `_eae_start` runs at `_prepare_main`'s first line: two non-daemon threads (`eaenc-4x`, `eaenc-6`), each `_rt_build` (sm_90a, -O3 -std=c++20 -lineinfo; the arch flags print on compile). `_eae_run` joins them with a hard deadline of EAE_JOIN_S = 220 s into prepare. A thread still alive means no race for that module (master).
- **Prepare-time copies:** a scale scan (min/max byte, count outside [0x08,0x7E]; E5 OFF everywhere if any). `_eae_recode` (encb's GPU recode) makes the A4b copy, plus the A5b copy when eligible, of `w13r`/`w2r` into `ea['w13e3'|'w2e3'|'w13e4'|'w2e4']` (~906 MB each). Copies no route uses are freed after the picks.
- **Routes (shape and prepare-time values only):** `ea['on'][T] = '<base>:e3|e4'` (new `_EAE_CORES` entries in `_ea_forward`'s table; `_ea3_forward_core` on `ext_key='exte'`, the E records, the twins' "x" buffers). At T=1 it is `'<FAST arm>:e<k1><k2>'` (`_ea6_forward_core` gains `enc=(k1,k2)`, which picks module and records per kernel). At T=256 it is `state['hm']['cold_e'][256] = 3|4` (`_hm_experts`' xarm block now calls `_eae_pair`, the same two launches as before when unset). E applies only where `_eae_base` finds master's route mirrorable: a v4x s twin, the hybrid's v4x s cold half, or a FAST K1+K2 v6 arm.
- **Race (`_eae_class`, all inside prepare):** per T, the p10/p50/p90 draws of `_ea4x_draws` (local-expert counts at T<=32; T=32 runs on the v4cxs bits-112 layout because the base's bits are used; `_ea_judge_routings` at 128/256). Per draw:
  - exactness: h and y on NaN-poisoned buffers (NaN positions must match, values torch.equal), plus kernel_fn's out eager and from the graph after the timed replays, all vs master's eager kernel_fn;
  - timing: EAE_ROUNDS 5 x EAE_REPS 15 interleaved, read-flushed `_rt_time` rounds of kernel_fn graphs (at 256 the isolated cold-half pair is timed in the same interleave);
  - after timing, tk13x/tk2x/errx and ea['err'] must all be 0.

  An arm is picked only if it is exact everywhere and its E/m <= 0.98 in every (draw, round); among those the lowest median wins. T=1 races all 8 K1 x K2 combinations (E4/m, m/E4, E4/E5, ...), so the K1-E + K2-B arms are included. The race stops starting classes after 20 s or 255 s into prepare. A tensor-cloned state copy (`_wg_copy_tree`) must give kernel_fn outputs equal to the original's at each picked T, or every pick reverts. Any exception reverts everything (`_eae_off`).
- **[ea-enc] block,** prepare's last prints (<= 400 chars/line, ~1.2-1.5 KB):
  - build s per module, join wait, phase-at;
  - regs/lmem/occ per E instance (K1e4 K1e5 K2e4 K2e5 for 4x and 6), recode s and MB, scale scan;
  - per T: exact y/n and median[min,max] of per-round E/m per arm (T=1 K1 and K2 separately and combined), plus T=256 `cold iso` ratios;
  - PATH CHOSEN per T with the fallback reason.
- **SIZE:** 1,018,791 B before stripping. Applied rewrite/build.py's AST-checked `strip_comments` (the AST is identical, so every string constant is unchanged): **948,272 B (`wc -c`)**. No diagnostics were hand-trimmed and no docstrings stripped.

Deviations, with reasons:
1. E routes only the exact judged T (1, 8, 16, 32, 128, 256). The other T of each class (2..7, 9..15, 17..31) stay on master because those are the only T the race measures.
2. Two E modules rather than one, so a failure in one does not cost the other. Both are separate, non-daemon, and start at t=0, as the review asks.
3. E5 is selected by pdl/flags bit 4096 instead of the `dq` argument, so master's host checks stay unchanged.
4. Draws are the 3 per T above; I did not add a separate 12-T canary, because routes exist only at the raced T.
5. A state-copy check was added; the review did not ask for it.
6. The race cap is checked before each class, so the last class can overrun 20 s by about one class (~1-3 s).
7. Node validation SKIPPED: the user's permission policy refuses node runs from subagents.

Biggest risks: nvcc for two more modules (~45-60 s each) at t=0 competes for CPU with es1a/ea6/ea4x/stk/wh, which may push master's own join deadlines (EA6_JOIN_S 170 s, EA4X_START_S 190 s). The CUDA was never compiled here: an nvcc error only costs the E lane, not master.

## Rules check
Checked against `git -C .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race diff`. The diff is 820+/825- lines, most of them comment-only line removals from strip_comments (AST identical). The functional hunks are `_prepare_main` (start, final block), `_ea_forward` (`**_EAE_CORES`), `_ea6_forward_core` (`enc`), `_hm_experts` (xarm -> `_eae_pair`) and the appended ea-enc section.
- [x] CUDA-graph capturable: kernel_fn's new paths are only the `_EAE_CORES` closures, `_ea6_forward_core`'s `enc` branch and `_eae_pair`. They do dict lookups on prepare-set keys and module launches, with no `.item()`/`.cpu()`/sync/print and no allocation or branch on input values. The race captures and replays every E route through `_rt_graph`. The two new `torch.cuda.synchronize()` calls are in `_eae_class`/`_eae_run` (prepare).
- [x] Nothing derived from `data` is cached across calls: the new state keys (`exte`, `ext6e`, `w13e*`/`w2e*`, `on[T]` strings, `hm['cold_e']`) come from the weights and prepare-time races only.
- [x] Weights repacked losslessly in prepare() only: the A4b copy is a fixed 32-bit permutation of each NVFP4 code word, with the scale bytes untouched. The A5b copy packs b-8 into 7-bit fields and is built only when every scale byte is in [0x08,0x7E], so it is lossless. Both stay NVFP4 codes and E4M3 scales, dequantized in-kernel. Every route requires h/y/out torch.equal to master.
- [x] Activations BF16 where SGLang keeps BF16: the same bf16x2 fma dequant and mma.sync fp32 accumulation; bit-exact to master is required to route.
- [x] Attention sees every key: n/a (MoE expert block, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the builds are `load_inline` in threads (as es1a/ea6/ea4x/wh are), and there is no subprocess/multiprocessing/os.fork in the diff.
- [x] All work happens inside the timed call; no output caching or replay: every E route still runs K1 + K2 (+ combine) per call. Prepare graphs are dropped after timing.
- [x] Size: `wc -c` = 948,272 B < 1,000,000.
- [x] First line kept (`# okc: problem=muse_moe_w4a16`); `python3 -I -m py_compile` OK; `__pycache__` removed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
... (master's pre-existing prepare-side host-sync / dtype warnings unchanged) ...
warn  host-sync    torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())   [new: _eae_class]
warn  host-sync    torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())   [new: _eae_run]
OK with warnings: .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race/muse_moe_w4a16/submission.py
```
(rc 0; 205 lines vs master's 203; the only new warnings are the two prepare-side syncs above.)

## Result

- job `702f1004b45f4bfa` (check), status done, passed True, tests 5/5
- speedup 2.0646x, 95% CI [2.0336, 2.0955], score 6513.924775815615 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.974770471501011, "ci": [1.945148914428496, 2.004392028573526], "job": "a5318a1f98984a5f"}; base result: muse_moe_w4a16/hypotheses/done/gain/hothalf-wh-a5b-port.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/engine-a-a4b-a5b-encodings-race.result.json`, `muse_moe_w4a16/hypotheses/done/gain/engine-a-a4b-a5b-encodings-race.patch`; branch `hyp/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 27.65 | 27.82 | 1.0351 | gained |
| bs=16 load=decode_16 sharding=ep8 | 60 | 104.06 | 106.68 | 1.0386 | gained |
| bs=32 load=decode_32 sharding=ep8 | 60 | 140.41 | 132.24 | 1.0552 | gained |
| bs=8 load=verify_1 sharding=ep8 | 60 | 75.15 | 72.96 | 1.0472 | gained |
| bs=128 load=verify_16 sharding=ep8 | 60 | 230.0 | 226.67 | 1.0432 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 274.38 | 264.69 | 1.0287 | flat |

gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16. Expected hit: 0/4. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

