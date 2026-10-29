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
status: running
review_at: 2026-10-29T215954Z
review_companions: null
review_best: 1.974770471501011
review_blob: 37ea90fc306c0ed8fb3d49094bf5f4acf6202307
stage: claimed
claimed_at: 2026-10-29T215954Z
base_sha: 973f86b3744c4b9b1d60469442fad05d8fca5783
base_blob: 37ea90fc306c0ed8fb3d49094bf5f4acf6202307
branch: hyp/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race
worktree: .worktrees/muse_moe_w4a16/engine-a-a4b-a5b-encodings-race
mode: null
restack_of: null
stale_base: null
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

