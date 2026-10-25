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
review: required
status: queued
---

# bf16-lossless-exp4-decode-only-throughput-probe

## Change

A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn routes NOTHING new (byte-identical master dispatch, so the headline should read master within noise). Built on master (blob a8bb42a) plus ONLY the codec and decoder from branch hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2 (job f1024212dd2f49f3: the prepare() exp4 encoder, the load_inline build with its compile-seconds print and TORCH_CUDA_ARCH_LIST=9.0a, and the device decode routine). The stage-3 review (held) found stage 2 decode/issue-bound (pooler m=1 30.8 us vs 19.0 incumbent; warm vs cold L2 barely differ; qkv's L2-resident compressed stream only ~1.5 TB/s compressed) and asked the deciding question: can the exp4 decoder alone reach >= 2.7 TB/s compressed (tie) / 3.3 TB/s (M3)? Change: add one new __global__ decode_only_kernel templated on WARPS_PER_CTA in {4, 8, 16} and CTAS_PER_SM in {1, 2}: each CTA streams its contiguous span of the compressed pooler weight with the same cp.async ring as stage 2, decodes every element to bf16 with the stage-2 routine into registers, XOR-folds the decoded bits into one 32-bit word per thread, and writes that word to a dummy [grid*threads] int32 buffer (so nothing is dead-code eliminated; no mma, no x, no output tile). In prepare(), after the encoder, for each of the 6 configs: one warm-up launch, then time 5 launches with cudaEvents twice: (a) L2-resident: the compressed qkv_gate weight (28.7 MB) re-read, and (b) cold: flush a 256 MB buffer before each event pair, pooler (51.1 MB compressed). Print '[decode-probe] cfg=... warm X TB/s cold Y TB/s (raw-equivalent Z)' and the best config, plus the same for a copy-only variant (same ring, no decode: XOR raw words) as the memory ceiling. Verify once that the XOR of decoded bf16 bits equals the XOR of the original bf16 weight bits computed with torch (bitwise_xor reduction) so the probe provably decoded every element. Wrap all of it in try/except; any failure prints and kernel_fn is unaffected. DECISION (record in FINDINGS at harvest): best cold decode-only < 2.7 TB/s compressed -> the lossless line is DEAD (move done/ hold to do-not-retry); 2.7-3.3 -> dead for mma.sync, note it; >= 3.3 -> the stage-2 loss is the MMA/x path, and a stage 3 may be minted against that number.

## Expected

- `*` flat: diagnostic only: kernel_fn is master's; the readout is the [decode-probe] log lines, not the rows

## Risk

Low for the score (routes nothing), costs one check. Must not exceed the 10-minute budget: stage 2's build took ~52 s and the encoder runs untimed; the probe adds ~1 s of launches. Rules: prepare-time timing is print-only and selects nothing. Cheap and decisive per the stage-3 review; if the user or churnsubs prefers to spend no more on the lossless line, close it without this run.
