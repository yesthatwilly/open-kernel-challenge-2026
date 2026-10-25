---
problem: muse_bf16_proj
slug: bf16-lossless-exp4-decode-only-throughput-probe
created: 2026-10-25T011408Z
base: master
area: lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)
change: >
  A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn routes
  NOTHING new (byte-identical master dispatch, so the headline should read master within noise).
  Built on master (blob a8bb42a) plus ONLY the codec and decoder from branch
  hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2 (job f1024212dd2f49f3: the
  prepare() exp4 encoder, the load_inline build with its compile-seconds print and
  TORCH_CUDA_ARCH_LIST=9.0a, and the device decode routine). The stage-3 review (held) found stage
  2 decode/issue-bound (pooler m=1 30.8 us vs 19.0 incumbent; warm vs cold L2 barely differ; qkv's
  L2-resident compressed stream only ~1.5 TB/s compressed) and asked the deciding question: can
  the exp4 decoder alone reach >= 2.7 TB/s compressed (tie) / 3.3 TB/s (M3)? Change: add one new
  __global__ decode_only_kernel templated on WARPS_PER_CTA in {4, 8, 16} and CTAS_PER_SM in {1,
  2}: each CTA streams its contiguous span of the compressed pooler weight with the same cp.async
  ring as stage 2, decodes every element to bf16 with the stage-2 routine into registers,
  XOR-folds the decoded bits into one 32-bit word per thread, and writes that word to a dummy
  [grid*threads] int32 buffer (so nothing is dead-code eliminated; no mma, no x, no output tile).
  In prepare(), after the encoder, for each of the 6 configs: one warm-up launch, then time 5
  launches with cudaEvents twice: (a) L2-resident: the compressed qkv_gate weight (28.7 MB)
  re-read, and (b) cold: flush a 256 MB buffer before each event pair, pooler (51.1 MB
  compressed). Print '[decode-probe] cfg=... warm X TB/s cold Y TB/s (raw-equivalent Z)' and the
  best config, plus the same for a copy-only variant (same ring, no decode: XOR raw words) as the
  memory ceiling. Verify once that the XOR of decoded bf16 bits equals the XOR of the original
  bf16 weight bits computed with torch (bitwise_xor reduction) so the probe provably decoded every
  element. Wrap all of it in try/except; any failure prints and kernel_fn is unaffected. DECISION
  (record in FINDINGS at harvest): best cold decode-only < 2.7 TB/s compressed -> the lossless
  line is DEAD (move done/ hold to do-not-retry); 2.7-3.3 -> dead for mma.sync, note it; >= 3.3 ->
  the stage-2 loss is the MMA/x path, and a stage 3 may be minted against that number.
expected:
  - shapes: "*"
    direction: flat
    why: "diagnostic only: kernel_fn is master's; the readout is the [decode-probe] log lines, not the rows"
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T011846Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T011847Z
base_sha: 41b1d0ab801c3b80cbb8f4ac832daa8d0cabfc3a
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-lossless-exp4-decode-only-throughput-probe
worktree: .worktrees/muse_bf16_proj/bf16-lossless-exp4-decode-only-throughput-probe
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T012237Z
impl_commit: b356752bfbee360cc99c4a914d83a5999c52db5a
submission_sha256: 6212769b2b06c04fd5a532ebf23d401b5b3d4d2dcc92736e6269e77efc71c660
norm_sha256: a9f9ec775d9c456c02d65a2558aa383ba13d59eaa0837fe6c151271a79be2ea5
submission_blob: 11c9d817fdded87c2ece585de4a74a40df07155d
lint_warnings: 13
attempt_file: 2026-10-25T012237Z-check-bf16-lossless-exp4-decode-only-throughput-probe.md
prepared_at: 2026-10-25T012237Z
job: 9148142cee4441ad
dispatched_at: 2026-10-25T012245Z
---

# bf16-lossless-exp4-decode-only-throughput-probe

## Change

A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn routes NOTHING new (byte-identical master dispatch, so the headline should read master within noise). Built on master (blob a8bb42a) plus ONLY the codec and decoder from branch hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2 (job f1024212dd2f49f3: the prepare() exp4 encoder, the load_inline build with its compile-seconds print and TORCH_CUDA_ARCH_LIST=9.0a, and the device decode routine). The stage-3 review (held) found stage 2 decode/issue-bound (pooler m=1 30.8 us vs 19.0 incumbent; warm vs cold L2 barely differ; qkv's L2-resident compressed stream only ~1.5 TB/s compressed) and asked the deciding question: can the exp4 decoder alone reach >= 2.7 TB/s compressed (tie) / 3.3 TB/s (M3)? Change: add one new __global__ decode_only_kernel templated on WARPS_PER_CTA in {4, 8, 16} and CTAS_PER_SM in {1, 2}: each CTA streams its contiguous span of the compressed pooler weight with the same cp.async ring as stage 2, decodes every element to bf16 with the stage-2 routine into registers, XOR-folds the decoded bits into one 32-bit word per thread, and writes that word to a dummy [grid*threads] int32 buffer (so nothing is dead-code eliminated; no mma, no x, no output tile). In prepare(), after the encoder, for each of the 6 configs: one warm-up launch, then time 5 launches with cudaEvents twice: (a) L2-resident: the compressed qkv_gate weight (28.7 MB) re-read, and (b) cold: flush a 256 MB buffer before each event pair, pooler (51.1 MB compressed). Print '[decode-probe] cfg=... warm X TB/s cold Y TB/s (raw-equivalent Z)' and the best config, plus the same for a copy-only variant (same ring, no decode: XOR raw words) as the memory ceiling. Verify once that the XOR of decoded bf16 bits equals the XOR of the original bf16 weight bits computed with torch (bitwise_xor reduction) so the probe provably decoded every element. Wrap all of it in try/except; any failure prints and kernel_fn is unaffected. DECISION (record in FINDINGS at harvest): best cold decode-only < 2.7 TB/s compressed -> the lossless line is DEAD (move done/ hold to do-not-retry); 2.7-3.3 -> dead for mma.sync, note it; >= 3.3 -> the stage-2 loss is the MMA/x path, and a stage 3 may be minted against that number.

## Expected

- `*` flat: diagnostic only: kernel_fn is master's; the readout is the [decode-probe] log lines, not the rows

## Risk

Low for the score (routes nothing), costs one check. Must not exceed the 10-minute budget: stage 2's build took ~52 s and the encoder runs untimed; the probe adds ~1 s of launches. Rules: prepare-time timing is print-only and selects nothing. Cheap and decisive per the stage-3 review; if the user or churnsubs prefers to spend no more on the lossless line, close it without this run.

## Adversarial review

Reviewed 2026-10-25T011846Z by an adversarial subagent (model opus).

**Verdict: proceed.** This costs one check and routes nothing, so its value depends on what it replaces. It is worth running because bf16-lossless-exp4-warpspec-persistent-stage3 is still queued at p1 even though LESSONS (2026-10-25) already calls the line dead. The probe is the cheapest way to settle that conflict for good. Two conditions: stage 3 must be blocked on it, and its pass/fail thresholds must use a yardstick measured in the same harness, because prepare-time event timings overstate throughput by 20-25%.

### Justification
The question is the right one, but the fixed thresholds are miscalibrated. Stage 2 streamed compressed weights at 1.4-1.6 TB/s with the MMA attached (pooler 40.3 vs 19.0 us incumbent). The tie point is real: master's pooler m=1 swap-AB kernel takes 18.6 us in the profile for 67.1 MB, which is 3.6 TB/s raw. At ratio 0.761 (51.1 MB compressed) the decoder must deliver about 2.7 TB/s compressed just to tie, before any MMA or x work. That is borderline, not hopeless, on paper: 3.6 TB/s raw is about 1.8e12 bf16/s, or about 7 elements per SM-cycle over 132 SMs, which leaves a budget of roughly 15-18 lane-instructions per element for exp4 decode. So a measurement is needed. WEAKEST LINK: the DECISION compares absolute prepare-measured TB/s to 2.7/3.3, but LESSONS 2026-10-25 (harvest) says prepare CUDA-event timings undercount the judge profile by 20-25%. A prepare reading of 3.0 TB/s is therefore about 2.4-2.5 TB/s in profile terms, which is a kill, while the rule as written would call it a tie. FIX: in the same prepare harness, with the same cold-L2 flush and event method, also time master's own pooler m=1 _bf16_swapab_kernel path (or _bf16_gemm_dispatch_impl at m=1) as the YARDSTICK. Decide on ratios: decode-only cold time / yardstick time <= ~0.85 means revive (it leaves room for MMA+x), 0.85-1.0 means dead for mma.sync, > 1.0 means DEAD. Keep the copy-only variant: if copy-only itself is slower than the yardstick, the ring, not the decoder, is the limiter, and that is also a kill for this design. The warm/L2-resident qkv number is diagnostic only: every timed replay starts with a cold L2 (TASK.md), so it must not enter the decision.

### Confounders on the current kernel
- None. kernel_fn is byte-identical to master and no landed path is touched. Any headline or row movement is run-to-run noise, and the harvest must say so (expected flat everywhere).

### Interference and A/B needs
No companion. The readout is the [decode-probe] lines, not the rows. The yardstick timing inside the same run replaces any control. Without the yardstick, one run would be ambiguous because of the 20-25% prepare undercount. With it, the run is decisive.

### Open questions
- Before dispatch, set bf16-lossless-exp4-warpspec-persistent-stage3 to depends_on this probe (or hold it), so the line cannot spend a stage-3 check before the probe decides. If the user prefers to spend nothing more on the line, close both now. LESSONS already supports that.
- At harvest, record in FINDINGS and LESSONS the decode-only/yardstick ratio, the copy-only/yardstick ratio, and the regs/spills of the best config. On a kill, move the lossless line to Do-not-retry with that number.


## Implementation notes

Only `muse_bf16_proj/submission.py` in the worktree changed: +540 lines, 0 deleted. `kernel_fn` and every master function are byte-identical; the only edits to existing code are `import os`/`import time` and one call, `_lossless_probe(state)`, at the end of `prepare()` (adds no state keys).

Added section (6) before `prepare()`:
- `_LP_CUDA` (built with `load_inline` inside prepare, `TORCH_CUDA_ARCH_LIST=9.0a` set and restored, `-O3 -lineinfo`, flags printed, compile seconds printed). It holds stage 2's `Grp`/`load_grp`/`decode_step` verbatim and `probe_kernel<WARPS,CTAS,DEC>` with the stage-2 cp.async.cg ring and next-stage register prefetch. Each CTA owns 64 weight rows and a K-stage range; WARPS 4/8/16 = 1/2/4 groups per stage; ring sized to ~160 KB (1 CTA/SM) or ~96 KB (2 CTAs/SM). DEC=true decodes all 4 k16 steps and XOR-folds the bf16 words to one word per thread into a dummy int32 buffer; DEC=false is the copy-only twin (same ring, XOR of raw smem words). No mma, no x, no output tile, no mbarriers, no spins.
- `_lp_encode`: stage 2's exp4 encoder verbatim, run on pooler and qkv_gate.
- `_lossless_probe`: yardstick, 6 configs (W4/W8/W16 x C1/C2), verdict.

Review fixes applied (the review wins over the spec):
1. Thresholds are ratios, not 2.7/3.3 TB/s. Master's pooler m=1 kernel (`_pooler_forward`) is timed first with the same cold-L2 flush plus cudaEvent method in the same function. Verdict: best verified decode-only cold time / yardstick <= 0.85 and best copy-only <= 1.0x yardstick gives REVIVE; otherwise DEAD, with a reason (0.85-1.0 = dead for mma.sync; >1.0 = decode slower; copy-only >1.0 = the ring is the limiter). Each config line prints compressed TB/s, raw-equivalent TB/s, time/yardstick and its inverse, regs, local (spill) bytes, smem, stages, occupancy (the review's harvest ask).
2. All probe buffers are locals of `_lossless_probe_body`; the wrapper then runs `gc.collect()`, `synchronize`, `torch.cuda.empty_cache()` and prints the still-allocated MB.
3. Verdict line is `[lossless-probe] REVIVE:` or `DEAD:` with the numbers. If the probe cannot run (build failure, no verified config) it prints `NO VERDICT (...)` instead; it never raises (inner `except Exception`, outer `except BaseException`).
- The qkv_gate warm L2-resident number is printed as a diagnostic only and is not in the verdict (review).

Deviations: (a) cold timing uses the median of 15 flushed launches instead of 5, and warm is the median of 3 rounds of 5, for stability. (b) The decoder source was copied from the stage-2 branch into a smaller standalone module (probe kernel only), not the full stage-2 file. (c) The XOR verification runs on the pooler decode of every config (XOR of decoded bf16 words vs XOR of the weight's int32 words, halving fold in torch); a mismatching config is excluded from the verdict. (d) Split per config = round(132 x CTAS / n-tiles), so grids are ~132/264 CTAs.
Not done by the build agent (no GPU here): nothing was compiled or run. Biggest risk: C2 with 16 warps forces <= 64 regs (launch bounds 512,2), so spills are likely and are reported honestly; that config may read slow for a reason that is not the decoder.

## Rules check

- [x] CUDA-graph capturable: `kernel_fn` is byte-identical to master (`git diff` shows 0 deleted lines and no edit inside it); the probe runs only in `prepare()`, where syncs are fine.
- [x] Nothing derived from `data` is cached across calls: the probe uses only the weights and synthetic x, and adds no state keys.
- [x] Weights: BF16 stays BF16; the encoder works on a copy inside the probe and frees it; the weights in state are untouched and never routed.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8: n/a: kernel_fn unchanged; the probe's synthetic x is bf16.
- [x] Attention sees every key: n/a: no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no subprocess/fork/multiprocessing written; `load_inline` is the same in-process build stage 2 used; the only env edit (`TORCH_CUDA_ARCH_LIST`) is restored.
- [x] All work happens inside the timed call; no output caching or replay: n/a: the probe is untimed and prints only; kernel_fn is master's.
- [x] Brief: self-check launches announced with flushed prints before and synchronize plus print after; plain cp.async.cg; `-lineinfo`; no mbarrier/spin waits; any failure prints and never raises; no wgmma/setmaxnreg (arch list still set to 9.0a and printed).

okc_lint (`python3 -I okc_lint.py submission.py --problem muse_bf16_proj`): "OK with warnings": 3 dtype warns (uint8 casts in the encoder, prepare-only), 10 host-sync warns (all in prepare, "fine in prepare()"). `python3 -I -m py_compile` passed; `__pycache__` removed.
