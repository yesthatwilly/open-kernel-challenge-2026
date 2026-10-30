# muse_attention_decode · 2026-10-30T082410Z · submit · ladder-at12-window-uniform-mask-branch

- **Job:** `ab23aba5fa6d4616` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `d7f6f36c` (ladder/at12 = master 4a6bc079 + AT10 9d727c04 + AT12), 560,273 B

## Check-readiness: READY — `ladder/at12` d7f6f36c (on master 4a6bc079). Drafted attempt text:
> Goal: AT10 + AT12 (ladder rungs) — sliding verify B=2..32: the s2b window kernel at K/V pipeline depth 3 (AT10,
> 'pdl3') and with the score mask on a CTA-uniform branch (AT12, 'pdl3u': interior tiles run no per-element mask
> compare/select; SASS common path ~714 -> ~573 instr/thread/tile), both bit-identical to the routed pdl pair, raced
> in prepare after the incumbent gate (AT10 then AT12, route at <= 0.99 in both interleaved rounds, IQR <= 2x
> max(inc, 0.25 us)); everything else master 4a6bc079. Parts touched: muse_sliding_verify_wgmma_v2b CUDA (kernel
> template NS, UB; host modes 5-8; attrs), _SW_MERGES 'pdl3'/'pdl3u', _sw_case bit-identity branches,
> _sw_at10_race(fam, rung), _sw_setup hooks. Rules (technical-rules): FP8 KV unchanged (FP8 values, wgmma e4m3);
> queries E4M3 as FA3 (unchanged); every window key attended (same loop bounds; the mask is applied on exactly the
> tiles that have edge || elo, identical values); softmax denominator FP32 from unrounded P (unchanged); BF16 output;
> no host sync / allocation-by-value in kernel_fn (route frozen at import); nothing cached from inputs across calls;
> no reference import; file 560,273 B < 1 MB; prepare: +2 kernel instantiations, ~0.2 s of races, H100 import 264.3 s.
> Risk: import time 264 s on H100 (judge H200 prepare budget 300 s; AT10 alone was 266.7 s, so AT12 adds ~0);
> B=2..16 AT10 race flips around 0.99 (AT12 races against whatever it left, so no loss either way).

## Proposed CLAIMS line
- [lead] AT12 window-kernel score mask on a CTA-uniform branch (ladder-at12-window-uniform-mask-branch: kernel<SPLIT, NS=3, UB>, interior tiles skip the per-element compare/select; SASS common path ~714 -> ~573 instr/thread/tile, hot INT 252 -> 143) | rows: layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: H100 gpunode06 GPU 1, takemyjob hopper/ladder-at12 c1f900d, run 20261030-005825 (2 seeds x 7 rounds x 20 reps): pdl3u vs pdl3 -0.57/-0.49 us B=16, -1.10/-1.09 us B=32 (pre-registered -0.5 / -1.1), kernel alone -0.6 / -1.26 us, identity arms <= 0.17 us, bit-identical (CHECK 0 bad); judge-style import of d7f6f36c 20261030-010711 routes pdl3u on both bands (0.973/0.976, 0.947/0.963), 264.3 s, KFN 0 bad | board: 1.697 | blob: - | reopen: judge check of ladder/at12 d7f6f36c | 2026-10-30 campaign

## Next rung
**AT13: pin the V^T transposer's loop-invariant shared addresses** (~85 INT + 8 SEL of the remaining ~573/tile,
-15%), the AT7e mechanism on this kernel. Registers are the constraint (125/128 now; AT12 freed 3): restructure the
offsets so w2/w3 and the 4 stores ride immediates off 2 base registers per transposer unit (the odd-row store order
is a per-thread +/-64 that can be folded into the base), then `asm volatile("" : "+r")` pin at the loop top. Predict
from AT12's measured rate (-141 instr -> -1.1 us B=32, i.e. ~7.8 ns per instr/tile at B=32, ~4 ns at B=16): ~-0.7 /
-0.35 us. After that, the remaining gap to the public (14.8 us v32s) is the ~7 us startup chain, not the loop.


## Why a scored run
Check 8ea881e2 1.728x beats the board 1.697 (+1.8%), same bytes d7f6f36c. Standing rule: scored submit + promotion when a ladder check beats the board.

## Result (after the run)
- job `ab23aba5fa6d4616` (submit), passed, **1.715x [1.690, 1.741]** vs board 1.697 -> landed on master (bytes d7f6f36c).
