# muse_attention_decode · 2026-10-30T124932Z · check · ladder-at5b-v1f-distributed-finalize

- **Job:** `08386aa5c887477a`
- **File:** muse_attention_decode/submission.py at git `ladder/at5b` f2c27d84 (= ladder/at5 1abef7eb + AT5b; master 413655b6, board 1.760), 612,574 B

## Goal
Ladder rung AT5 (campaign 2026-10-30-public-ladder, user-requested). Target is full verify B=1 (v1f): ours ~30.1 us on
the judge (FA3 + FlashAttnFwdCombine + metadata), public 14.2 us.
- Pre-registered: -11.0 us e2e on H100 (judge-scaled ~-8 us, v1f ~22 us), headline ~+1.3%.
- H100 measured: -10.5..-11.0 us e2e (41.4 -> 30.5, x1.35), in-import race 0.775-0.801, KFN 0 bad.

## AT5b (this commit, on top of AT5 1abef7eb)
- Pre-registered -2.5 us vs AT5 (finalize 3.5 -> 1.0); measured -1.30 / -1.46 us in-run (seeds 3000/4000, identity
  within 0.08 us), e2e v1f 28.6 us vs FA3 41.2 us, in-import race 0.731-0.736, KFN 0 bad, mode 13 local 0 B (AT5 16 B).
- Change: the ticket elects the last 8 arriving CTAs as finalizers (row group of 16 rows each); each spins
  (ld.acquire.gpu, bounded, error bit 1) until all G CTAs arrived, then finalizes with 16 threads/row (float4), re-zeroes
  its rows; the last finalizer (acq_rel counter cnt[1]) resets ticket, cnt[1], fallback count. Host: grid >= 8 check.
- Extra risk: finalizers spin on at most 7 not-yet-arrived CTAs while 256 arrived CTAs exit (no co-residency needed;
  bounded spin -> error word, which the import self-check would catch).

## What changed, and where (AT5, unchanged by AT5b except the finalize)
- fvg CUDA (`_FVG_CUDA`, module `muse_fvg_gate_v1`): one new instantiation `fvg_kernel<4, PV8>` (launch mode 13).
  - Main loop: fvg's FP8-QK / FP8-PV loop, unchanged (AT7/AT7c/AT7e code paths).
  - `finalize` (MODE 4): per-row FP32 red.global.add (v4 for o, scalar for l) at fixed reference R=16 into a workspace
    past the partial slots; R11 guard rows store their exact (o, m, l) to the CTA's own slot and bump cnt[3].
  - After the loop (MODE 4): fence.acq_rel.gpu + one atom.acq_rel ticket per CTA; the last CTA LSE-merges any fallback
    slots (out-of-line `at5_fallback`, only when cnt[3] > 0), applies the fused RMSNorm x sigmoid(gate) epilogue
    (hardware rcp/rsqrt/ex2, ~2 ulp fp32), writes bf16, then re-zeroes the workspace, slots, ticket and count.
  - CTAs 0..15 L2-prefetch the gate rows. Host: mode & 8 selects the kernel, workspace size check, attr slot 8.
  - Modes 1/5 (routed at v16/v32): every added line is MODE 4 only (regs 128/127, local 0, v16/v32 race 0.79-0.78 of master as before; SASS identity not checked).
- Python: `_fvg_gate` admits batch 1; `_fvg_agentic` handles batch 1 (80000 / tails); `_FVG_BUF[1]` (zeroed workspace,
  slot m = -inf); the v1 row joins fvg's race with arm 13 at bar 1/1.03; `_fvg_forward` launches mode 13 non-cooperative;
  `_fvg_cnt_clean` checks cnt[0..3]; PATH CHOSEN line for B=1.

## Rules check (rules/technical-rules.llms.txt)
- [x] CUDA-graph capturable: one Triton pre-attn launch + one kernel launch, fixed grid 264; no host sync, no
      value-dependent allocation or control flow; workspace allocated once at import (two-state replay check passes).
- [x] Nothing derived from `data` is cached across calls: workspace, slots, ticket and fallback count return to 0 / -inf
      at the end of every launch (self-reset; checked after every import launch and in the probe after timing).
- [x] Weights: none (attention).
- [x] FP8 KV cache stays FP8 (read as cached bytes); queries E4M3 as FA3 (unchanged pre-attn); P for PV in E4M3 as FA3
      (unchanged AT7 path); softmax denominator in FP32 from unrounded P (unchanged AT7c); output BF16.
- [x] Every key of the whole context is attended exactly once (stream-K tiles cover [0, n + 8) with the causal mask in the
      last tile); the softmax and PV run over every key. The R11 guard never drops mass (exact fallback slot).
- [x] Merge in FP32 (red.global.add.f32, fixed reference; fallback LSE merge in FP32).
- [x] No timing patches, harness access, reference import, subprocesses or network.

## Risks
- The finalize runs on one SM (3.3-3.5 us on H100); H200 clock/latency may differ. The race re-decides on the judge.
- Import clock: the v1 race runs after v16/v32 inside fvg's 470 s row rule (node: starts ~311 s, import 308.7-313.7 s).
- Mode 13 has 0 B local with AT5b (gate allows <= 64 B).
- The forced-guard path is exact but ~1.8 ms; it never runs on judge data (R11: row max <= 65.4 < guard ~105).
File: muse_attention_decode/submission.py at git f2c27d84 (worktree .worktrees/ladder-at5b-ck, branch ladder/at5b-ck; AT5 + AT5b)

## Result
- job `08386aa5c887477a` (check), passed, **1.791x [1.764, 1.818]** vs board 1.760 (best run on the board: the stray c63f5b4a 1.785): v1f 23.7 us.
