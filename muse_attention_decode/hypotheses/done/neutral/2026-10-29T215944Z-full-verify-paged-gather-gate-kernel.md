---
problem: muse_attention_decode
slug: full-verify-paged-gather-gate-kernel
created: 2026-10-29T215944Z
base: master
mint_blob: "2eb03f76ceefa9bf81e28616ced3b33be3a70443"
best_at_mint: 1.634
area: full-layer verify attention (FA3 call at verify_16 / verify_32)
change: >
  Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8
  KV, ctx ~80000) with the H100-validated gate kernel from takemyjob hopper/fa3gate 4b2a169
  (probes/fa3gate; research/h100-fa3gate-1029.md): a persistent paged-gather attention kernel, 3
  stages x 2 CTAs/SM = 64 KB in flight per SM, page indices prefetched a ring ahead, FP8 QK^T by
  wgmma straight from the swizzled cp.async ring, f16 PV, online softmax, grid-barrier distributed
  combine. Build it as a separate background-built load_inline extension; route per row (verify_16
  full, verify_32 full only; verify_1 and every sliding/decode row keep master) by an in-prepare
  interleaved race through kernel_fn (read-flushed, cold L2) against master's FA3 path: on only if
  within the problem tolerance on every draw (fp64 relL2 and the judge metric, which the node
  measured at 2.5-2.8e-2 vs bounds 0.083/0.085) and <= 0.95 vs master in every round; else master.
  Inputs fresh per call (page table, lengths); no caching of anything input-derived; no host sync
  in kernel_fn. Print an [fvg] summary (<= 1.5 KB) at the end of prepare: build s,
  regs/spill/smem/occupancy, per-row per-round ratios and errors, PATH CHOSEN.
expected:
  - shapes: layer=full load=verify_16*
    direction: faster
    why: node e2e 0.83-0.84 of master FA3 (v16)
  - shapes: layer=full load=verify_32*
    direction: faster
    why: node e2e 0.82 (v32); headline ~ 0.256 share x ln(1/0.85) ~ +4.2%
  - shapes: layer=sliding*
    direction: same
    why: not routed
  - shapes: layer=full load=verify_1*
    direction: same
    why: not routed (gate loses at v1, 1.42x)
risk: high
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "fa3-metadata-fold-preattn-full-verify: that removed FA3's prepare node and kept FA3; this replaces FA3's main kernel on v16/v32 full with a node-measured faster paged-gather kernel (memory parallelism 64 KB/SM vs FA3's ~24-32 KB)"
target: check
value_gain: 4.2
value_p: 0.4
value_ev: 1.68
value_cost: 500k tokens, 1 judge run
evidence: >
  H100 gpunode06 fa3gate (takemyjob hopper/fa3gate 4b2a169; research/h100-fa3gate-1029.md), 2 seeds,
  5-7 interleaved rounds, cold L2: e2e 272 vs master 331 us (v32, 0.82), 147-151 vs 176-181 (v16,
  0.83-0.84); error vs fp32 1.67e-3 (FA3 6.3e-3..3.9e-2), judge metric 2.46-2.79e-2 vs bounds
  0.076-0.085; attention triage shares verify_16 full 12.4%, verify_32 full 13.2% (board 4e386647)
review: proceed
status: done
review_at: 2026-10-29T220405Z
review_companions: null
review_best: 1.6339575246416982
review_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
stage: harvested
claimed_at: 2026-10-29T220405Z
base_sha: 91aa9ba6749c86329bf85d64ec022f8893a3af68
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel
worktree: .worktrees/muse_attention_decode/full-verify-paged-gather-gate-kernel
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T221722Z
impl_commit: 9f9e177c886e87d78e4c6ace4b3e52f36fbec327
submission_sha256: 8208363790e1fceed7afcdb891653ff13f066a2da67b85268ea9ba7056404b49
norm_sha256: 4933135fda60fb0e3bfa611e805cf05b3ed62938dc5b5a7902ec6b402af9b14a
submission_blob: 5af6d05f5e3bd04457ce4d0cd4b32bef563809c3
lint_warnings: 105
attempt_file: 2026-10-29T221723Z-check-full-verify-paged-gather-gate-kernel.md
prepared_at: 2026-10-29T221723Z
job: 06c051723643462f
dispatched_at: 2026-10-29T221743Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.6153707615545347
speedup_ci: [1.5911402001312167, 1.6396013229778525]
harvested_at: 2026-10-29T222811Z
best_at_harvest: 1.6339575246416982
promotion: null
mechanism: null
---

# full-verify-paged-gather-gate-kernel

## Change

Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8 KV, ctx ~80000) with the H100-validated gate kernel from takemyjob hopper/fa3gate 4b2a169 (probes/fa3gate; research/h100-fa3gate-1029.md): a persistent paged-gather attention kernel, 3 stages x 2 CTAs/SM = 64 KB in flight per SM, page indices prefetched a ring ahead, FP8 QK^T by wgmma straight from the swizzled cp.async ring, f16 PV, online softmax, grid-barrier distributed combine. Build it as a separate background-built load_inline extension; route per row (verify_16 full, verify_32 full only; verify_1 and every sliding/decode row keep master) by an in-prepare interleaved race through kernel_fn (read-flushed, cold L2) against master's FA3 path: on only if within the problem tolerance on every draw (fp64 relL2 and the judge metric, which the node measured at 2.5-2.8e-2 vs bounds 0.083/0.085) and <= 0.95 vs master in every round; else master. Inputs fresh per call (page table, lengths); no caching of anything input-derived; no host sync in kernel_fn. Print an [fvg] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill/smem/occupancy, per-row per-round ratios and errors, PATH CHOSEN.

## Expected

- `layer=full load=verify_16*` faster: node e2e 0.83-0.84 of master FA3 (v16)
- `layer=full load=verify_32*` faster: node e2e 0.82 (v32); headline ~ 0.256 share x ln(1/0.85) ~ +4.2%
- `layer=sliding*` same: not routed
- `layer=full load=verify_1*` same: not routed (gate loses at v1, 1.42x)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 fa3gate (takemyjob hopper/fa3gate 4b2a169; research/h100-fa3gate-1029.md), 2 seeds, 5-7 interleaved rounds, cold L2: e2e 272 vs master 331 us (v32, 0.82), 147-151 vs 176-181 (v16, 0.83-0.84); error vs fp32 1.67e-3 (FA3 6.3e-3..3.9e-2), judge metric 2.46-2.79e-2 vs bounds 0.076-0.085; attention triage shares verify_16 full 12.4%, verify_32 full 13.2% (board 4e386647)
- **Gain if it works:** 4.2% headline; **p** 0.4; **EV** 1.68% (bar: gain >= 2.5%, EV >= 1.5%).
- **Cost:** 500k tokens, 1 judge run.

## Risk

H100 -> H200: the kernel is math/latency-bound at the ceiling on the H100 (math-only 233 us at v32 = memory-ceiling time); per-SM structure transfers, absolute TB/s does not; projected H200 e2e 0.82-0.87 (untested; the H200 is held). Every earlier custom full-verify loop lost to FA3 on the judge (CLAIMS); this one is the first measured faster on the same SM. mma.sync e4m3 is emulated on sm_90a (the kernel uses wgmma for QK).

## Adversarial review

Reviewed 2026-10-29T220405Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed: E passes (fresh at 1.6340 / blob 2eb03f76; this is the judge check that CLAIMS.md:86 [lead] names as its reopen), and the in-prepare race with a <=0.95 bar caps the downside at the cost of one check. Expect the H100 edge to shrink on the H200: realistic e2e about 0.86-0.92 and a 2-3.5% headline, not 4.2%. The build has to remove a grid-barrier hang risk and gate on page_size/kv_scale.

### Justification
The mechanism holds up on this SM. FA3's full-verify main loop is limited by memory parallelism per SM (CLAIMS.md:84: one CTA/SM, 24-32 KB in flight, 2.04 TB/s, DRAM 62.5%). The gate keeps 64 KB/SM in flight, and in the same process with interleaved cold-L2 rounds over 2 seeds its mainloop beat FA3's main kernel 1.22-1.24x (259 vs 322 us at v32, 137-139 vs 167-171 at v16). End to end it was 0.82 / 0.83-0.84 of master's call, with tolerance held and 23x more accurate than FA3 vs fp32. It differs from the earlier custom full-verify loops that lost on the judge in four measured ways (s3-s7, FINDINGS:177-189, CLAIMS.md:63; s6's DIAG was 179/367 us vs FA3 148.6/285.9). (1) FP8 QK uses wgmma. mma.sync e4m3 is emulated on sm_90a (F2FP unpack + 2 HMMA; v1 of the lineage ran 467 us). (2) Lean load issue, about 140 fewer instr/warp/tile and no spills (v4: 292->258 us). (3) The combine is distributed after a grid barrier and costs 3-4 us. The ticket combine re-read partials already evicted from L2 by the KV stream and cost +47-84 us on the critical path. (4) It was raced against master's actual FA3 call on the same SM, which no earlier loop was. So it is not the same mechanism again, and prior is set correctly: the metadata fold kept FA3, this replaces it. WEAKEST LINK, the H100->H200 transfer. On the H100 the gate is effectively math-bound: math-only 233 us (v32) / 119-121 (v16) equals the loads-only time, and mainloop = math + about 26 us / 18 us of issue-slot and barrier contention from load-issue instructions. Both terms are per-SM and do not shrink on the H200 (same GH100 SM count and clocks). The H200's 4.8 vs 3.35 TB/s helps FA3, which is memory-parallelism-bound but still latency/HBM-sensitive: its main kernel already gains 1.13-1.15x (322->285.9, 170.9->148.6). It barely helps the gate. Estimate: gate mainloop on the H200 about 240-259 (v32) / 125-139 (v16), plus about 11-13 us combine/launch, against master's call of about 295 / 158. That gives e2e 0.86-0.92 (v32) and 0.87-0.95 (v16). v16 can miss the 0.95 bar. The H100's 0.82 is therefore an upper bound on the gain, not the expected value. Rough headline at r=0.875 on both rows (full verify about 0.79 / 0.74 of those loads, inferred from the 13.2% / 12.4% shares): about +1.7% (v32) + 1.5% (v16) = about 3.2%. Pessimistic, with only v32 routed at 0.92: about 1.3%. p(>=2.5% headline) is nearer 0.35 than 0.4, putting EV at about 1.1-1.4. That is marginal against the 1.5 bar, but the race makes the loss case free apart from the check. The pre-registered fa3gate rule (mainloop >= 2.6 TB/s) was NOT met (2.52). Proceeding rests on CLAIMS.md:86, after WS was closed (fa3ws/fa3math, 0.78-0.83x of the gate's math). Not a defect, but the build log must make the H200 reading interpretable (see open_questions). The expected shapes are consistent: only v16/v32 full move, verify_1 is excluded (1.42x loss, latency-bound combine), and sliding is untouched.

### Confounders on the current kernel
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup (+ ab-tune-batch-gate, patches in done/gain and done/neutral): on full verify B>1, master's fused pre-attention writes q in the PADDED/PAIRED layout for FA3's head-paired call, and _paired_norm_gate_kernel reads FA3's paired output layout. The gate kernel writes the plain [B,8,16,64] bf16 layout (out[((b*8+t)*HQ + hh*8+gq)*D]). The build must either read master's paired q and write the layout the paired norm-gate expects, or switch both endpoints for the routed rows. Otherwise there is an extra pad/unpad launch or a wrong-layout epilogue. That hits correctness, or eats several us of the margin, which the node e2e (call-to-call) did not include. Print which q/out layouts and which norm-gate kernel are used on routed rows.
- fused-preattention-triton (and the v3 fused pre-attention prologue): this kernel writes the new K/V at cache_loc before attention. The gate reads the cache right after, so ordering must stay in-stream (no PDL early trigger into the gate launch, see rules_flags). Otherwise the gain is independent.
- Not confounding: the sliding, decode and full decode_1/verify_1 landings (stream-K, cluster, onepass, s2b/sw17). Their rows are untouched and their kernels are not routed through this path. The sw17 gate 'flips between runs' (CLAIMS.md:60), so sliding verify_32 can move about 1-3% on its own. Read the full verify_16/32 rows from the profile, not the headline.

### Interference and A/B needs
No companion needed. The change is routed per row by its own in-prepare race against master's FA3 path, and the build prints per-row, per-round ratios and errors. The judge profile then attributes any movement directly to the full verify_16 / verify_32 rows, and the race itself is the control. If the race does not route, the run still yields the first H200 measurement of the gate's mainloop and of FA3 at v16/v32 (provided the diagnostics below are printed), which is the information the next decision needs. An ab-revert of the pairing landings would not make the single result any less ambiguous; it is a layout-integration question for the build, not an interaction effect.

### Open questions
- What the build MUST print (the [fvg] block plus PATH CHOSEN lines that survive the 16 KB log tail). Build s and whether it finished before the race. From cudaFuncGetAttributes: numRegs, localSizeBytes (spill, must be 0), static+dynamic smem. Occupancy blocks/SM (must be 2), multiProcessorCount and G. Per row and round: master call us, gate e2e us, ratio, fp64 relL2 vs fp32, judge-style metric and worst row vs bound. The route decision with the reason. ALSO, if build time allows, a one-shot H200 diagnostic: gate mainloop-only (MODE 0), math-only and loads-only us with the implied TB/s at v16/v32. That is the H200 gather ceiling and the math floor, the two numbers the node could not supply, and they decide whether a v16-fails/v32-passes outcome is the math floor (stop) or contention (tunable).
- Does master's paired q/output layout on full verify B>1 match what the gate reads and writes? If not, which side is adapted, and is its cost inside the race timing? (The race runs through kernel_fn, so it is, but the node's 0.82 did not include it.)
- Is the 0.95 bar applied per row independently? v16 could plausibly sit at 0.90-0.96 on the H200. Routing v32 alone is still worth about 1.3-1.7%.
- The per-SM CTA priority imbalance (lower-id CTA finishes about 20% earlier) should be the same on the H200 scheduler. A priority-aware static split (<= 2% modelled) is the cheapest follow-up if the gate routes but barely.
- Value estimate: H200 transfer likely gives r about 0.86-0.92, not 0.82-0.85. Re-derive value_gain to about 2-3.5% when logging the result, so the outcome is not scored as a miss against an inflated 4.2%.


## Implementation notes

**Mechanism:** full verify B=16/32 (q_len 8, page 1, kv_scale 1): FA3 paired call + _paired_norm_gate_kernel replaced by one persistent paged-gather gate kernel (fa3gate v6, 3 stages x 2 CTAs/SM, wgmma FP8 QK / f16 PV, grid-barrier combine with the RMSNorm x sigmoid(gate) epilogue fused), routed per row by an in-prepare race vs master.

- New `[fvg]` section before the module-level `if torch.cuda.is_available():` block (worktree `muse_attention_decode/submission.py`, +1001 lines, file 539,763 B). Hooks: `_fvg_start_build()` is the first statement of that block; `kernel_fn` takes `_fvg_forward` right after the `_SW_ROUTE` check when the frozen `_FVG_ROUTE` has the batch, `_fvg_gate(cfg)` holds (config only) and `_fvg_shape_ok(data)` holds (shape only). `_fvg_setup()` runs last, after `_sw_setup`. The `[fvg]` summary prints after the `[paths]` re-print, so it is the last part of the log.
- CUDA: only fa3gate.cu's `fa3y_kernel<NS=3, MINB=2, MODE, F=0, FN=3>` (cfg 8) was kept, with its helpers copied verbatim, both fences included (the cp.async->wgmma `fence.proxy.async` stays). Dropped: the lineage kernels v1-v5, the SM-split and fenceless variants, `g_dbg`/globaltimer/smid. MODE 1 is the routed mode. MODE 0, 2 and 3 are diag only. Changes from the source:
  1. The early `return` for empty ranges is gone (with a grid barrier it would hang). Ranges are now cut over `Ge = min(G, T)`, and CTAs >= Ge skip the mainloop but still arrive at the barrier and take part in the combine (`cta_of(.., Ge)`).
  2. The barrier spin is bounded (2^22 x 128 ns) and sets error bit 1. Bit 2 means more than 1024 tiles per CTA, which the host shape gate prevents. Bit 4 means a non-sm_90a build, under the `#if __CUDA_ARCH_FEAT_SM90_ALL` guard.
  3. The combine writes the final bf16 output, with RMSNorm and sigmoid(gate) applied in fp32 from qkv.
- Launch: `cudaLaunchKernelEx` with `cudaLaunchAttributeCooperative` only. There is no programmatic-serialization or PDL attribute, and the preceding Triton pre-attention kernel uses no PDL. G = 2 x `multiProcessorCount`. Prepare requires occupancy >= 2, localSizeBytes == 0 and cooperative-launch support, or it keeps master. A capture probe checks that a cooperative launch can be captured; if it cannot, the plain launch is used and the line says so. Each row (16, 32) has its own counters [arrive, depart, error] and its own partials, allocated in prepare from G and the batch. They are re-zeroed after every race step and in `finally`.
- Layout (review confounder): routed rows switch both ends. `_pre_attention_kernel` gets PAIRED=False (the unpaired e4m3 q, the same call master's B=1 verify makes), and the gate writes `[B*8, 1024]` bf16 with the epilogue fused. Neither `_paired_norm_gate_kernel` nor `rms_norm_gate` is launched. This is printed. The race times this whole path through kernel_fn.
- Build: `load_inline` runs in a daemon `threading.Thread` started at the top of the import, with `-gencode=arch=compute_90a,code=sm_90a` in `extra_cuda_cflags`. That flag contains "arch", so torch adds no arch flags, and the thread never touches `TORCH_CUDA_ARCH_LIST`, which the main thread's builds set and restore. The thread is joined with a timeout at a 470 s import-clock rule. If it has not joined, or the build failed, master stays.
- Race, per row and independently, through kernel_fn:
  - Three eager draws: agentic soft keys, agentic with coherent values, and short lengths with T < G (which exercises Ge). The agentic draws force tails (n+8)%64 of 1 and 63 and the longest row, 86000.
  - Each draw is checked against the submission's own fp64 references: e4m3-q relL2 <= 2e-2, judge-style relL2 vs the unrounded ref <= clamp(2.5 x ref error + 1.5%, 3%, 15%), worst row <= 5x the bound, finite. Master's judge metric on the same draw is printed too.
  - Then the two-state `_alt_check` replay, and 3 rounds of interleaved cold-L2 graph replays (route off vs on, empty graph subtracted). A row routes only if the ratio is <= 0.95 in every round.
  - A one-shot diag follows (start rule 500 s): MODE 1/0/2/3 kernel-only us, with implied TB/s at v16/v32.
- Deviations / risks: `kernel_fn` still allocates its q and output with `torch.empty`, as master does; persistent buffers are never allocated in kernel_fn. The kernel was never compiled here (this node has no nvcc), so the first build is the judge's. Node validation was skipped: subagent H100 runs outside a user-approved command are refused. The background nvcc runs alongside master's sequential builds, so if CPU contention delays them, master's own phase gates could shift.

## Rules check

- [x] CUDA-graph capturable: `_fvg_forward` makes Triton + one ext launch on the current stream; no `.item()/.cpu()/synchronize()` in kernel_fn's call graph (lint: none); branches only on config (`_fvg_gate`) and shapes (`_fvg_shape_ok`); lens/page table read on device; capture verified in prepare by `_alt_check` and the race graphs.
- [x] Nothing derived from `data` is cached across calls: `_FVG_BUF` holds partials/counters sized from G and batch only; `_FVG_ROUTE` frozen at import from synthetic races.
- [x] Weights: n/a: attention problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (read as e4m3, untouched format); queries rounded to FP8 E4M3 (as FA3); P f16 and fp32 O/softmax (finer than FA3); output bf16.
- [x] Attention sees every key: stream-K covers every 64-key tile of every request; only causal / length masking on the last tiles.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import (own fp64 reference), subprocesses (a `threading.Thread` around load_inline, as master's builds) or network.
- [x] All work happens inside the timed call: pre-attention, attention, combine and epilogue all run per call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
warn  host-sync  ... (outside kernel_fn's call graph; fine in prepare())   [x ~110, master's prepare + [fvg] prepare lines]
warn  dtype      L7646: q = torch.full(..., dtype=torch.uint8 ...).view(FP8)   [master, pre-existing]
OK with warnings: .../muse_attention_decode/submission.py
```
`python3 -I -m py_compile`: OK (__pycache__ removed).

## Result

- job `06c051723643462f` (check), status done, passed True, tests 16/16
- speedup 1.6154x, 95% CI [1.5911, 1.6396], score 2331.5390433187868 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/BEST.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 10.82 | 11.38 | 0.9726 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 15.62 | 15.47 | 0.9925 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.38 | 15.77 | 0.9872 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 104.77 | 105.4 | 0.9963 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 16.81 | 17.06 | 0.9996 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 177.91 | 190.81 | 0.9264 | lost |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.9 | 13.08 | 1.0625 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.09 | 29.53 | 0.9995 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 19.51 | 20.02 | 0.9593 | lost |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.74 | 163.82 | 0.9989 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 27.1 | 27.58 | 0.9816 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.09 | 302.29 | 1.0013 | flat |

gained on sliding verify_1; lost on full decode_32; sliding verify_16. Expected hit: 0/4. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-verify-paged-gather-gate-kernel`) that adds a further change in the same area is the only way to learn more.

