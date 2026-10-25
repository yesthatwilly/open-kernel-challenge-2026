# Rewrite read: muse_prefill_layer, 2026-10-24 kind:rewrite runs

These are judged on the replaced kernels' own time per shape from the profile top_kernels. They are not judged on the headline. Numbers are GPU µs per layer call. The incumbent is `attribution.base_result` unless a row says otherwise. Floors come from each hypothesis's `## Change` or `## Expected`.

**Noise calibration.** The wgmma run below never ran its kernel, so it measured master's code. Its first_turn `_moe_nvfp4_gemm_kernel` rows differ from the base run by -3.5% to +5.4%. The new master (stack-tonight-five-prefill, c4d14d62) shows 8083-9174 µs for the same code. A difference of up to about ±6% on this kernel row, and on the Marlin rows (2065-2343 µs at turn_4 across runs that did not touch them), is therefore noise.

**New master (stack-tonight-five-prefill).** It runs 768.6/735.9 µs of Marlin at turn_1 and 2148.2/2115.3 µs at turn_4. On the Marlin path, the parallel align (`_align_parallel_kernel`, sorted_ids hold local routes only) and `_silu_mul_sorted_kernel` replace count_and_sort plus act_and_mul. At first_turn the align runs over 32 local experts and the w2 zero-store is OFF (`[w2 zero-store] OFF`). Attention is unchanged.

## Summary

| rewrite | path that actually ran | replaced kernel vs incumbent | verdict |
|---|---|---|---|
| experts-cuda-wgmma-mixed-input-grouped-gemm | **build FAILED** (wgmma targeted sm_90, not sm_90a), so it fell back to master's Triton | not measured | PROMISING (unmeasured; fix the build) |
| sliding-attn-block-triton-gqa-fused-normgate | Triton at turn_1/turn_4; FA3 at first_turn (timing gate) | 1.26-1.39x slower at turn_1, 2.2-2.6x slower at turn_4, 1.87x at first_turn (gated off) | DEAD |
| attn-full-prefill-cuda-fp8-exp-offload | wgmma **build FAILED** (same sm_90 bug); the mma.sync fallback ran ON | 3.07-3.21x slower than FA3 on all 5 full shapes | mma.sync backend DEAD; the wgmma design is PROMISING only behind a hard gate |
| experts-triton-nvfp4-packed-asm-dequant | packed dequant ON, Triton at first_turn | 0.96-1.055x, mean +1.4%: no change | DEAD |
| experts-triton-w4a16-first-turn-blockt256 | Triton at every T (stage-3 lineage) | first_turn 0.82-0.95x of stage 2 (-13%) | DEAD (absorbed into master) |
| experts-triton-w4a16-swapab-reglhs | Triton at every T (stage-2 lineage) | 2.5x faster than stage 1; parity with Marlin at turn_4, +9-21% at turn_1, 0.8-0.9x of Marlin at first_turn | DEAD (absorbed into master) |

---

## 1. experts-cuda-wgmma-mixed-input-grouped-gemm (job 1821bb0e1f7246e2, neutral 1.064)

**Path.** `[cuda expert gemm] build FAILED after 23.2s`: ptxas rejected `wgmma.mma_async`, `wgmma.fence`, `commit_group` and `wait_group` because they were compiled `on .target 'sm_90'` (in `..._cuda.compute_90.ptx`). The flags were `extra_cuda_cflags=["-O3","-arch=sm_90a",...]`, but torch's cpp_extension still adds its own `-gencode` from `TORCH_CUDA_ARCH_LIST` or the detected 9.0 capability. That plain compute_90 pass cannot assemble wgmma. The self-check logged `OFF: build failed`, and the path lines read `PATH CHOSEN T=16384: triton` and `marlin` at every T ≤ 4096. The profile has no `_nvfp4_wgmma_grouped_kernel` row. **The run measured master.** The 1.064 is master plus noise (best at harvest was 1.059).

| shape (first_turn, the stage-1 target) | incumbent Triton w13+w2 (base fb1e52) | this run (still Triton) | floor (math) |
|---|---|---|---|
| full ctx0 | 8873.3 | 8565.6 | 1670 |
| sliding ctx32768 (w 138) | 8323.7 | 8294.6 | 1670 |
| full ctx32768 | 8338.7 | 8701.4 | 1670 |
| full ctx65536 | 8149.4 | 8591.5 | 1670 |
| turn_1 sliding/full (Marlin, not targeted) | 752.4 / 653.9 | 704.8 / 701.0 | 189 |
| turn_4 sliding/full (Marlin, not targeted) | 2217.1 / 2131.0 | 2065.4 / 2343.4 | 305 |

**Where the time went.** Nothing new ran. Build plus fallback cost 23 s of prepare.

**Verdict: PROMISING (unmeasured, not failed).** Milestone M0 failed on a toolchain flag, not on the design. The incumbent is still about 5x its floor at first_turn, and that is the biggest gap in the layer. Re-run stage 1 rather than design a new stage 2. That run must fix:
- **Build.** Set `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"` before `load_inline`. Also wrap the wgmma and setmaxnreg asm in `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`, with an `#else` that sets the error word or traps. A stray plain sm_90 pass then still compiles, and the self-check turns the path OFF if that image is the one loaded. Log the arch flags nvcc received.
- **Rebase onto stack-tonight-five-prefill.** At first_turn, master now aligns over the 32 local experts (`moe_align_block_size(..., LOCAL_EXPERTS)`, so fewer padded tiles) and has turned the w2 zero-store OFF. The kernel's "w2 zero-stores -1 items" must follow `state['zero_nonlocal_w2']` instead of always storing, or it adds back about 940 MB of stores that master removed. Its NaN-prefilled non-local w2 comparison already tolerates either setting.
- **Later stages (turn_4/turn_1, replacing Marlin).** These must consume master's new parallel-align `sorted_ids` (local routes only) and feed `_silu_mul_sorted_kernel`, not the old count_and_sort plus act_and_mul contract.

## 2. sliding-attn-block-triton-gqa-fused-normgate (job 78d8d7bdec55491c, neutral 1.051)

**Path.** All 7 sliding self-checks passed, at 10-15x margin to tolerance and an fp32-error ratio of 0.99. The CUDA-graph, cold-L2 timing in prepare: turn_1 38.8 (incumbent) vs 53.8 (Triton, S=4), ratio 1.389; turn_4 74.9 vs 168.4 (S=1), ratio 2.247; first_turn 346.2 vs 647.9, ratio 1.871. Only first_turn was gated, so it fell back to FA3. turn_1 and turn_4 **ran Triton despite their own timing showing it slower.** The timing gate applies only above T = 4096. The eager numbers were misleading: Triton 89 vs 101 at turn_1, because launch overhead dominates eager.

| shape | incumbent (q cast + FA3 + combine + norm-gate) | new (attn + combine) | ratio | floor |
|---|---|---|---|---|
| sliding turn_1 | 25.7 + 7.2 in profile (38.8 graph-timed with cast and norm-gate) | 42.1 + 6.7 = 48.8 (53.8 graph-timed) | 1.26-1.39x slower | ~6.4 |
| sliding turn_4 | 61.4 in profile (74.9 graph-timed) | 162.5 (168.4) | 2.25-2.6x slower | ~25.6 |
| sliding first_turn | 304.4 (346.2) | FA3 kept: 307.2 | gated off (Triton 1.87x) | ~139 |
| full layers (control) | FA3 415.2 / 1634.0 / 892.2 / 4471.1 / 8006.7 | 424.3 / 1614.9 / 897.9 / 4466.8 / 8019.6 | flat | — |

**Where the time went.** A program covers one KV head, 16 tokens and 8 q heads (M = 128). Every program walks the whole 2048 + 16 key window of 64-key page-size-1 gathers, about 33 serial tiles. K/V traffic per token therefore scales with 2048 / BLOCK_TOK = 128x re-reads, against FA3's 128-token-per-head tiles. At S=1 (turn_4) nothing parallelises that serial walk: 384 programs on 132 SMs, each latency-bound on the gather. The FLOPs are trivial (about 13 GFLOP at turn_4).

**Verdict: DEAD.** It is slower than the incumbent on every target shape, and the whole prize is small. Sliding attention at turn_1 and turn_4 is about 39 and 75 µs out of 1080 and 3335 µs. Even at the floor that is about 1% of the turn loads. At first_turn, FA3 is 2.2x its floor on a 14.3 ms shape (2%). Competing needs a 64-128 token tile with split-KV at every T, which means rebuilding FA3 in Triton for at most about 1% headline. Do not mint stage 2. If revisited, fold the norm-gate into whatever replaces FA3 on full layers (#3) instead.

## 3. attn-full-prefill-cuda-fp8-exp-offload (job f5c30b15fd284c15, regression 0.881)

**Path.** `[cuda full attn] wgmma: load_inline FAILED after 23.1s`, the **same `not supported on .target 'sm_90'` bug** as #1, this time for `wgmma.mma_async with FP8 types`. The mma.sync backend built (41.4 s; 128 regs, 2 CTAs/SM, 36.9 KB smem, no spills). It passed the exp2_sw unit test (max relative error 7.5e-5) and all 6 FA3 comparisons, and was switched ON: `CUDA mma kernel (desc_swap=0, exp offload 2/8 on FMA)`. Its own eager timing already showed it 3x slower (turn_1: FA3 421.8 µs vs 1264.0 (sw=0) / 1306.6 (sw=2); turn_4: 1640.8 vs 5069.5 / 5236.2). That timing was "print only", so nothing stopped it.

| full shape | incumbent FA3 (+combine), base fb1e52 | new attn_fwd_kernel<2> (+combine) | ratio | exp floor | new / floor |
|---|---|---|---|---|---|
| turn_1 | 415.2 + 7.0 = 422.2 | 1280.9 + 19.7 = 1300.6 | 3.08x | 250 | 5.2x |
| turn_4 | 1634.0 | 5214.3 + 30.3 = 5244.6 | 3.21x | 1000 | 5.2x |
| first_turn ctx0 | 892.2 | 2741.4 | 3.07x | 555 | 4.9x |
| first_turn ctx32768 | 4471.1 | 13954.6 | 3.12x | 2770 | 5.0x |
| first_turn ctx65536 | 8006.7 | 25152.5 | 3.14x | 4990 | 5.0x |
| sliding shapes (control) | 25.7 / 61.4 / 304.4 | FA3 kept | flat | — | — |

**Where the time went.** The mma.sync core is uniformly about 3.1x FA3 on every shape. That points to core-loop throughput, not split, launch or causal-balance effects. The exp offload made it slightly worse (sw=2/8 is 3% slower than sw=0 in eager). On this core, FP32 issue slots bind before MUFU does. That is a warning against the hypothesis's premise that FA3 is MUFU-bound: offload helps only once the MMA is wgmma and the issue slots have room.

**Verdict.** The mma.sync backend is DEAD. The wgmma design is **PROMISING but unmeasured, and riskier than #1.** It has a real target: FA3 is 1.6x the exp floor and is the second-largest kernel on full shapes, at about 4-5% headline once the experts drop. Stage 2 (re-run stage 1 with fixes) must:
- **Build.** Apply the same sm_90a fix as #1 (`TORCH_CUDA_ARCH_LIST=9.0a` plus the `__CUDA_ARCH_FEAT_SM90_ALL` guard).
- **Gate.** Delete the mma.sync backend, or never route to it. Replace the print-only timing with a hard CUDA-graph, cold-L2 gate per shape class that turns the path ON only at ≤ 0.92x FA3. Missing the gate is what produced the 0.881.
- **Offload.** Start with exp offload 0/8, the measured-better setting, and sweep the fraction only after wgmma parity is shown.

It does not interact with the new master: attention code is unchanged there.

## 4. experts-triton-nvfp4-packed-asm-dequant (job 49ff8248b96547de, neutral 1.056; stage 5 of the Triton expert line)

**Path.** `[expert gemm] packed dequant ON`. Triton ran at first_turn and Marlin at T ≤ 4096, unchanged. The compile report shows the asm version *added* a shared-memory buffer: `local_alloc` went from 4 to 5 and smem from 140 KB to 157 KB at BLOCK_T 256. The w13 BLOCK_T 256 variant still spills 4 and runs at 255 regs. So the asm output was converted through smem before the dot. The packed tile did not stay in the register-LHS layout.

| first_turn shape | incumbent Triton w13+w2 (base fb1e52) | packed-asm | ratio | floor | new / floor |
|---|---|---|---|---|---|
| full ctx0 | 8873.3 | 8880.5 | 1.001 | 1670 | 5.3x |
| sliding ctx32768 (w 138) | 8323.7 | 8778.7 | 1.055 | 1670 | 5.3x |
| full ctx32768 | 8338.7 | 8025.1 | 0.962 | 1670 | 4.8x |
| full ctx65536 | 8149.4 | 8460.0 | 1.038 | 1670 | 5.1x |
| turn_1 / turn_4 (Marlin, untouched) | 752.4/653.9, 2217.1/2131.0 | 734.3/754.1, 2234.2/2188.7 | drift | 189 / 305 | — |

**Where the time went.** Cutting dequant from about 10-12 to about 3 instructions per element moved nothing (mean +1.4%, inside the ±6% kernel noise). The per-element dequant ALU is not the binding cost. The binding cost is how Triton lowers `tl.dot`: synchronous wgmma after a full in-register dequant, no producer/consumer split, plus the extra layout conversion this change introduced. That matches the wgmma rewrite's own rationale.

**Verdict: DEAD.** This ends the Triton dequant-tuning line. Milestone 2 (beat the incumbent) was missed, and the kernel is still about 5x its floor. Further gains on this kernel need the CUDA wgmma engine (#1), not another Triton stage. It rebases trivially onto the new master (the asm is orthogonal to the align and zero-store changes), but there is nothing to stack.

## 5. experts-triton-w4a16-first-turn-blockt256 (job b33b87b5083b4d1c, check, neutral 1.023; stage 3)

**Path.** No path lines. The compile report shows the BLOCK_T 256 config built (w13: 254 regs, 4 spills, 140 KB smem). Triton ran at every T, since this predates the hybrid Marlin gate. Its base is stage 2 (swapab).

| shape | incumbent: stage 2 swapab | blockt256 | ratio | floor |
|---|---|---|---|---|
| first_turn full ctx0 | 10009.5 | 8573.7 | 0.857 | 1670 |
| first_turn sliding ctx32768 | 9908.9 | 8535.2 | 0.861 | 1670 |
| first_turn full ctx32768 | 9746.7 | 7959.7 | 0.817 | 1670 |
| first_turn full ctx65536 | 9146.3 | 8721.8 | 0.954 | 1670 |
| turn_1 sliding/full (config unchanged) | 796.3 / 822.6 | 847.4 / 728.5 | drift | 189 |
| turn_4 sliding/full (config unchanged) | 2157.8 / 2100.3 | 2274.6 / 2317.7 | drift (code identical) | 305 |

**Where the time went.** Halving redundant dequant passes (about 4 token tiles per weight tile down to 2) took about 13% off first_turn. It reached 7.96-8.72 ms against a milestone of ≤ 7.5 ms, so it is still about 5x the floor. The turn_4 "loss" in the attribution is noise: the T ≤ 4096 config is byte-identical.

**Verdict: DEAD as a line of its own, because it was absorbed.** This config is master's first_turn path today (experts-hybrid-marlin-turn-triton-first-turn = this plus Marlin at T ≤ 4096), and stage 5 (#4) already followed it. Nothing remains to mint from it.

## 6. experts-triton-w4a16-swapab-reglhs (job 00148fbedfb6414f, check, neutral 1.007; stage 2)

**Path.** There is no log in the archive. The profile shows `_moe_nvfp4_gemm_kernel` at every T, so Triton replaced Marlin everywhere. Its base is stage 1 (experts-triton-w4a16-inkernel-fp4, job 218515e2). The Marlin figures are the incumbent numbers quoted in the hypothesis for that era.

| shape | stage 1 inkernel-fp4 | Marlin of that era | swapab | vs Marlin | floor |
|---|---|---|---|---|---|
| turn_1 sliding | 2026.3 | ~730 | 796.3 | 1.09x slower | 189 |
| turn_1 full | 2008.0 | ~681 | 822.6 | 1.21x slower | 189 |
| turn_4 sliding | 3340.6 | ~2248 | 2157.8 | 0.96x | 305 |
| turn_4 full | 3456.6 | ~2201 | 2100.3 | 0.95x | 305 |
| first_turn full ctx0 | 14406.1 | 11.1-12.0 ms | 10009.5 | ~0.85x | 1670 |
| first_turn sliding ctx32768 | 13524.3 | 11.1-12.0 ms | 9908.9 | ~0.85x | 1670 |
| first_turn full ctx32768 | 14314.5 | 11.1-12.0 ms | 9746.7 | ~0.85x | 1670 |
| first_turn full ctx65536 | 13280.9 | 11.1-12.0 ms | 9146.3 | ~0.8x | 1670 |

**Where the time went.** Swap-AB with the register LHS removed stage 1's interleave/transpose/smem round trip: 1.3-2.5x faster than stage 1 on every kernel row. It met the turn_4 (≤ 2250) and turn_1 (≤ 1000) milestones but missed first_turn ≤ 9 ms. turn_1 stays dequant/latency-bound behind Marlin, and first_turn stays about 6x its floor.

**Verdict: DEAD as a line of its own, because it was absorbed.** Its kernel is master's Triton engine. At turn_4 it ties the current master's Marlin (2158/2100 vs 2148/2115), so swapping that gate back buys nothing. Further expert gains need #1.

---

## What the new master (stack-tonight-five-prefill) changes for any stage 2

- **#1 wgmma experts.** This is the one rewrite master directly affects. Follow `state['zero_nonlocal_w2']` (OFF at first_turn now) and the 32-local-expert align at first_turn. For the turn_4/turn_1 stages, consume the parallel-align `sorted_ids` and `_silu_mul_sorted_kernel` layout on the Marlin path.
- **#3 full attention and #2 sliding attention.** Attention is untouched on master, so neither is affected.
- **#4-#6.** These are dead or absorbed. #4 would rebase cleanly but has nothing to stack.

The common lesson: both CUDA rewrites died on the same `load_inline` arch bug (an sm_90 gencode alongside `-arch=sm_90a`). Every future `load_inline` wgmma or setmaxnreg kernel should carry the `TORCH_CUDA_ARCH_LIST=9.0a` fix plus the arch-feature guard, and needs a hard timing gate before any non-preferred backend is routed.

---

## 01:04Z harvest (2026-10-25)

Master for comparison is BEST `stack-tonight-five-prefill`:
- check c4d14d62f11e4aaf: 1.0705x [1.054, 1.087]
- submit 502f2eab95164781: 1.0744x

The master expert GEMM at first_turn is Triton `_moe_nvfp4_gemm_kernel` at 8.08-9.17 ms. turn_1/turn_4 run Marlin.

### experts-triton-packed-dequant-turn4 — job ce3cc1a22a244ec7, neutral 1.052x [1.036, 1.068]: RAN (Triton, not CUDA). Flat to a loss. DEAD.

**What ran (log):**
- Triton expert GEMM configs. Packed=False: 176-255 regs, local_alloc 4. Packed=True: 207-255 regs, local_alloc 5. BLOCK_T=256 spills 4 in both.
- `packed dequant ON`. e2e vs Marlin rel L2 5.86e-3 at T=8192 and T=3012.
- No gate line. The base is stale (3c444875, pre-`stack-tonight-five`), so turn_4 (T=3012) runs the Triton GEMM where master runs Marlin.

**Per-shape against master:**
- turn_4: Triton 2149 µs vs Marlin 2148 µs sliding (tie), and 2306 vs 2115 µs full (+9%). The shape went from 4849 to 4965 µs (+2.4%, z 2.1).
- first_turn: expert kernel 8.65-8.98 ms vs master's 8.08-9.17 ms. The shapes were +2.1% to +8.3% slower (z 2.5-9.6). The paired speedups were 1.096-1.167 vs master's 1.136-1.209.
- turn_1: flat.

**Verdict: DEAD.** This is the second run confirming that the packed asm dequant does not pay (see experts-triton-nvfp4-packed-asm-dequant above). It also confirms that Marlin should keep turn_4.

### attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc — job 306cbde58aff4fa4, neutral 1.068x [1.052, 1.084]: built, passed the canary and self-check, gated OFF everywhere. NEVER RAN in the judged path.

**What ran (log):**
- Build: `TORCH_CUDA_ARCH_LIST=9.0a, nvcc arch flags ['-gencode=arch=compute_90a,code=sm_90a']`, `build OK: wgmma module compiled in 31.0s`.
- Kernels:
  - `attn_fwd_kernel`: 116 regs, 0 B local, 2 CTAs/SM, 36 KB dynamic smem
  - `attn_combine_kernel`: 72 regs, 3 CTAs/SM
  - binaryVersion 90
- Canary: desc_swap 0/1, nsplit 1/2 and combine all OK and finite, error word 0.
- Self-check: all 5 cases PASS. vs fp64, rel L2 is 3.6e-3 to 9.3e-3, equal to or better than FA3's 6.2e-3 to 1.25e-2.
- GATE, prepare timing:

| shape | FA3 µs | new µs | new / FA3 |
|---|---|---|---|
| turn_1 | 425.3 | 675.3 | 1.588 |
| turn_4 | 1632.7 | 2725.2 | 1.669 |
| first_turn ctx0 | 906.0 | 1383.2 | 1.527 |
| first_turn ctx32768 | 4496.3 | 6990.1 | 1.555 |
| first_turn ctx65536 | 8076.0 | 12602.5 | 1.560 |

  The threshold is 0.92x, so the result is `PATH CHOSEN ... FA3` on every shape.
- **Per-shape:** this is master's path. The FA3 rows are identical (e.g. 415→420, 1620→1655, 4467→4456, 8017→8012 µs). Every shape is within ±2.6% of master. The two |z|>2 rows (first_turn ctx0 −2.6%, sliding turn_4 +3.7%) come from untouched Marlin and Triton expert rows, so they are run drift.

**Verdict: DEAD as designed.** The FP8 wgmma kernel is correct but 1.53-1.67x slower than FA3 on every full shape, so it needs a new design (warp specialisation and a TMA pipeline at FA3's level) before another check. The hard gate worked as intended: a slow kernel cost nothing.

### experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc — job 6d047cdeefd94357, regression 0.940x [0.926, 0.954]: RAN at T=16384 only, LOST ~1.8x. DEAD as is.

**What ran (log):**
- Build: `TORCH_CUDA_ARCH_LIST=9.0a ... ['-gencode=arch=compute_90a,code=sm_90a'] (sm_90a only)`, `load_inline compiled in 31.4s`.
- `_nvfp4_wgmma_grouped_kernel`: regs 168, **local_bytes 680 (spill)**, dynamic smem 190.6 KB, 1 CTA/SM, grid 132.
- Canary w13+w2: ok, error word 0. One-hot w13/w2: ok.
- e2e at T=16384: w13 and w2 vs Triton rel L2 **0.000e+00**; full experts vs Marlin 5.9e-3. Bit-identical to the Triton GEMM is surprising for a different MMA order. It is worth checking that the e2e really compared the CUDA output.
- `PATH CHOSEN T=300/753/1506/600/4096/3012: marlin; T=16384: cuda`.

**Per-shape against master.** Kernel ms are the wgmma grouped kernel vs master's Triton `_moe_nvfp4_gemm_kernel`:

| shape | new µs | master µs | Δ | kernel |
|---|---|---|---|---|
| first_turn ctx0 full | 22028.7 | 14705.3 | +49.8% (z 77) | 15.24 vs 8.08 ms |
| first_turn ctx32768 sliding (w 138) | 21471.3 | 14264.6 | +50.5% (z 72) | 15.67 vs 8.21 ms |
| first_turn ctx32768 full | 25236.5 | 17746.7 | +42.2% (z 20) | 16.22 vs 9.17 ms |
| first_turn ctx65536 full | 29244.7 | 21733.7 | +34.6% (z 15) | 14.66 vs 8.11 ms |
| turn_1 / turn_4 (Marlin, control) | — | — | −1.5% to +1.7% | flat |

**Verdict: DEAD as is.** The sm_90a build is now proven to run, so the wgmma path really executed. It is 1.8-1.9x slower than the Triton GEMM, with a 680 B spill at 1 CTA/SM. Only a redesign is worth another check: fix the register pressure (setmaxnreg producer/consumer split) and use 2 consumer warpgroups.

## 03:41Z harvest

Reference: master `stack-tonight-five-prefill`, check c4d14d62f11e4aaf, 1.0705x. (BEST.json's submit 502f2eab gave 1.0744.) z is Δ / √(err²+err²).

**Drift warning:** turn_1 rows on the untouched Marlin path moved −7% to −10% in the persistent run (Marlin profile 644 vs 769 µs, same kernel). turn_1 and first_turn rows must be confirmed by a kernel change in the log, never by z alone.

### experts-triton-persistent-grid-first-turn — job 5d9ddbbaaffd4599, check 1.0726x [1.057, 1.089]: RAN the A/B, GATED OFF. DEAD (persistent is slower).

- Persistent kernels compiled at grid=132: 249/210/255 regs at BLOCK_T 64/128/256, 0 spills. The non-persistent BLOCK_T=256 has 4 spills.
- **[persistent A/B] ratio lines** (T=16384, BLOCK_T=256, 74 live blocks, 7 alternations):
  - w13: non-persistent 3059.2 µs vs persistent 3148.7 → **ratio 1.0292 (persistent 2.9% slower)**
  - w2: 1579.7 vs 1654.0 → **ratio 1.0470 (4.7% slower)**
  - `PATH CHOSEN: w13 non-persistent, w2 non-persistent (gate ratio <= 0.98)`
- So the scored path is master's. The headline 1.0726 is master (1.0705) re-measured.
- The shape moves vs master are drift on unchanged code: sliding turn_1 −7.3% (z−5.4), full turn_1 −10.3%, first_turn ctx0 −3.5%, sliding first_turn −2.5%. The `_moe_nvfp4_gemm_kernel` first-turn row is 8082 vs 8209 µs, the same kernel.
- **Not a landing candidate.** With 74 live blocks on 132 SMs a persistent grid has no tail to recover. Grid-stride scheduling only adds loop overhead at this occupancy.

### experts-triton-turn4-persistent-local-align-control-no-persi — job 374efc72c4d14856, check 1.0557x: RAN (Triton at turn 4 with local align). DEAD.

- `[turn4-triton] ON`. The local-parallel align cut n_padded from 25216 to 4096 (bsm 128) and from 29440 to 8192 (bsm 256). align ok with 0/12 mismatches; e2e rel L2 5.9e-3.
- turn_4 still loses to Marlin even with the padding gone. Sliding: 3376 vs master 3319 µs (+1.7%), Triton 2193 vs Marlin 2148. Full: 4967 vs 4849 (+2.4%, z+2.6), 2254 vs 2115.
- The first_turn rows (+1.4-1.8%) are on the same path as master: drift.

### experts-triton-turn1-blockt32-stream — job 8d1a2efec67b4758, 0.9425x: RAN, LOST at turn 1. DEAD.

- BLOCK_T=32 kernels: 168 regs (199 packed), 0 spills. e2e rel L2 5.9e-3. The run took the Triton path at turn_1.
- turn_1: sliding 1590 vs master 1101 µs (+44%, z+31) with `_moe_nvfp4_gemm_kernel` 1206 vs Marlin 769. Full 1915 vs 1530 (+25%). The turn 1 load fell to 0.713.
- At T=753 the Triton NVFP4 GEMM is 1.6x slower than Marlin even at BLOCK_T=32. Keep Marlin at turn 1.

## 04:36Z harvest

Master: `stack-tonight-five-prefill` (submit 502f2eab, 1.0744; check c4d14d62, 1.0705). Both runs below turned their own paths OFF, so every row is master re-measured. Their turn_1 rows (−6% to −8.5%) and first_turn rows (−3.8% to +3.1%) are the known prefill drift (LESSONS 03:41Z: turn_1 drifts −7% to −10% without any kernel change).

### norm-boundaries-persistent-multirow: job 46afe489e2754c83, check 1.0716. DEAD.

- `[boundary] OFF: no variant beat master by 2%; PATH CHOSEN per T class: {753: 'master', 3012: 'master', 16384: 'master'}`. The coefficient probes matched the formula (attn 0.0, ffn max diff 1.9e-3).
- No row moved because of the change. Turn_1 sliding −6.0% and full −8.5% are drift on unchanged Marlin and FA3 kernels; Marlin itself fell 734 → 693 µs. SGLang's fused boundary kernels are already within 2% of a persistent multi-row variant at prefill sizes.

### experts-triton-first-turn-unmasked-loads-l2-hints: job aeda8b5a523642b1, check 1.0677. DEAD.

- The A/B over 7 alternations at T=16384 measured:
  - w13: master 3297 µs, unmasked 3867 µs (1.173x slower), unmasked plus hints 3886 µs (1.179x slower).
  - w2: master 1725 µs, unmasked 2041 µs (1.183x slower), unmasked plus hints 2043 µs (1.185x slower).
  - Every variant had 254 regs and 4 spills.
  - The gate (≤0.98x) printed `PATH CHOSEN = w13 master, w2 master`.
- Removing the load masks made the Triton NVFP4 expert GEMM 17-18% slower. The L2 eviction hints added nothing. The masks are not the bottleneck, and the unmasked form probably changed the pipelining or vectorisation Triton emits at 254 regs. Close the masks/hints knob.
