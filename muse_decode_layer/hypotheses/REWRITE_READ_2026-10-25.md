# Rewrite read — muse_decode_layer, 2026-10-24 kind:rewrite runs (H200)

Each rewrite is judged on the replaced kernels' own time per target shape. That time is compared with the incumbent profile (the `attribution.base_result` job) and with the floor in the rewrite's `## Milestones`. The headline is not the measure.

Sources:
- `done/<outcome>/<slug>.result.json`
- `archive/<slug>/<job>.{log,okc.json.gz,profile.json.gz}`
- the base job's `profile.json.gz`

Caveats:
- The profiles list only the top ~12 kernel names per shape. Rows marked "+hidden" mean that some replaced small kernels fall into the "... N more" tail and cannot be counted. All µs figures are single-profile kernel times. Shape µs figures are the benchmark medians from attribution.
- Marlin moves by ±20-40 µs between single profiles, even on byte-identical master paths. For example, at full verify_16 in the dense-chain run Marlin went 284→326 µs on the T>32 master path, and in the oversub run it went 109→74 µs. The Marlin rows are noise unless a pattern repeats.

| slug | outcome | verdict |
|---|---|---|
| small-t-launch-minimal-dense-chain-pdl | gain (master) | **PROMISING**: stage 1 met M1; it missed M2 on R and on P at T≥16, and missed M3 by 2.7 µs |
| full-decode-cuda-splitkv-port | neutral (vs a moved best) | **PROMISING**: −45 µs on the replaced kernels at both target shapes, about 1.8-2.0x the floor |
| triton-flash-decode-full-paired-kv | neutral | **DEAD**: slower than FA3 (1.63-1.66 TB/s), superseded by the CUDA split-KV kernel |
| triton-flash-decode-full-oversub-kv-prefetch | regression | **DEAD**: kernel 1 got 21-22% slower (1.34-1.37 TB/s) |
| moe-chain-cuda-swapab-expert-once-large-t | failed (0/0 tests) | **FIXABLE bug**: a sticky CUDA illegal instruction in the prepare self-check; the design was never measured |

---

## 1. small-t-launch-minimal-dense-chain-pdl (gain, now master) — job 81ee6714b7614a99, base 9dcaec316b3045a7

**Path (log):**
- `load_inline compiled in 41.5s`, P occupancy [3,2,2], grid 216 CTAs.
- PDL probe: eager overlap False; graph overlap True, so pdl=1.
- Self-check max errors: ei 1.3e-4, logit 3.0e-6, others 0.
- `PATH CHOSEN = DENSE-CHAIN at T <= 32, MASTER above`. There was no fallback. The verify_16/32 shapes (T=128/256) ran master and are the in-run control.
- P timing from the log, with CUDA events in a graph:

| T | pdl=0 | pdl=1 |
|---|---|---|
| 1 | 21.0 µs (3.60 TB/s) | 20.7 µs (3.64 TB/s) |
| 8 | 22.1 µs | 21.9 µs (3.45 TB/s) |
| 32 | 26.7 µs | 26.4 µs (2.85 TB/s) |

**Replaced kernels per target shape (µs).**

Incumbent:
- the pooler nvjet row (the 2-call row minus the remaining qkv call, or the separate 128xN splitK row)
- the router: cuBLAS `dot_kernel` at T=1, or the `tss` hi/lo pair at T≥8
- the pooler's splitKreduce share
- `vectorized_layer_norm` (pre-expert F.rms_norm)
- +hidden: FusedMuseSparkRouting, the expert_map gather, casts and moe_align, roughly 2 µs each

New: `dl_pooler_router_kernel<NF>` (P) and `dl_route_norm_align_kernel` (R).

| shape (T) | incumbent visible | new P + R | Δ shape (bench) | shape floor | M2 bar (P / R) |
|---|---|---|---|---|---|
| sliding decode_1 (1) | 18.8 + 5.0 + 1.6 + 6.2 = 31.6 +hidden | 21.9 + 7.6 = 29.5 | 145.8→132.7 (−13.1) | 40.0 | P ≤20: 20.7 (event) — miss by 0.7; R ≤4: 7.6 — miss |
| full decode_1 (1) | 20.3 + 5.0 + 1.9 + 5.6 = 32.8 +hidden | 21.2 + 7.5 = 28.7 | 157.0→142.6 (−14.4) | 44.2 | same |
| sliding verify_1 (8) | 19.9 + 9.2 + 2.1 + 5.2 = 36.4 +hidden | 22.5 + 7.4 = 29.9 | 200.8→186.5 (−14.3) | 62.3 | P 21.9 (event) OK at ≤22; R miss |
| full verify_1 (8) | 19.3 + 9.0 + 5.6 + 5.8 = 39.7 +hidden | 22.9 + 7.5 = 30.4 | 218.1→195.9 (−22.1) | 66.4 | same |
| sliding decode_16 (16) | 20.0 + 9.7 + 7.2 + 5.7 = 42.6 +hidden | 23.9 + 8.0 = 31.9 | 225.8→217.1 (−8.7) | 76.7 | P ~ OK; R miss |
| full decode_16 (16) | 20.1 + 9.3 + 3.7+ + 5.6 = 38.7 +hidden | 24.3 + 7.7 = 32.0 | 364.9→347.5 (−17.4) | 137.5 | same |
| sliding decode_32 (32) | 20.5 + 9.3 + 7.8 + 5.5 = 43.1 +hidden | 28.3 + 7.6 = 35.9 | 251.9→247.0 (−4.9) | 84.5 | P ≤22: 26.4 (event) — miss |
| full decode_32 (32) | 21.0 + 9.2 + ? + 5.8 = 36.0+ +hidden | 27.6 + 7.5 = 35.1 | 535.8→535.0 (flat) | 208.1 | same |

**Where the time went:**
- M1 is met everywhere: P is at or below master's pooler+router+reduce rows.
- P streams 75.5 MB at 3.6 TB/s at T=1, which is ~1.3x the 4.8 TB/s byte floor (~15.7 µs) and right at the 3.8 TB/s M2 target. At T=32 it drops to 2.85 TB/s; the implementation notes already flagged the per-CTA L2 re-read of x (~37.7 MB vs 75.5 MB of weight) as the cause.
- **R is 7.4-8.0 µs on every shape, ~2x its 4 µs bar.** It is T-independent, so it is a serial tail: the last block builds the whole align. It is not per-token work.
- **PDL bought only 0.3 µs.** In the P timing, pdl=1 is only 0.3 µs faster than pdl=0. P's predecessor is the SGLang attn_boundary JIT kernel, which never executes `launch_dependents`, so the weight prefetch overlaps nothing until the predecessors are our own kernels.
- At decode_1 sliding the rows left after stage 1 are:
  - Marlin 40.2 µs (30%)
  - qkv nvjet 13.1 µs, 37.7 MB, which is ~2.9 TB/s against ~9.9 µs at 3.8 TB/s
  - o/unpooler nvjet 7.0 + 5.3 µs, plus splitKreduce 3.4 µs (2 calls)
  - norm_residual 3.6 µs × 2
  - FA3 10.4 + 4.0 µs
- Marlin check: at verify_1 (T=8), Marlin read 76→96 µs (sliding) and 87→107 µs (full) in this profile. The benchmarked verify_1 gain is still positive, and Marlin moves ±40 µs on untouched master shapes, so this is probably noise. It is still worth one A/B on the Marlin row with R's align vs master's align at T=8, because R produces a different sorted_ids layout.

**Verdict: PROMISING (already on master).** Stage 2 should target, in order:
1. **Attention-side chain (the planned stage 2).** One qkv split-K GEMV, which is 13 µs now against a ~10 µs bar, with QKNorm, RoPE, the FP8 casts and the cache store in the last CTA. Plus an o GEMV with RMSNorm×sigmoid(gate) in its prologue. This is also what makes PDL pay, because the predecessors become our kernels that can trigger `launch_dependents`.
2. **R to ≤4 µs:** a parallel align (per-expert warps, or a warp top-k), not a single last block.
3. **P at T=16/32:** a wider token tile (BLOCK_N 128) to halve the x re-read, from 26.4 to ≤22 µs.

Items 1-3 are already queued as `dense-chain-stage2-attn-side-pdl`, `dl-route-warp-topk-pdl` and `dl-p-wide-tile-t16-t32`. M3 (decode_1 sliding ≤130 µs) needs about 3 more µs. Fixing R alone covers that.

---

## 2. full-decode-cuda-splitkv-port (neutral) — job e789032128df4291, base 9dcaec316b3045a7

**Path (log):**
- `[splitkv] load_inline compiled in 49.6s`.
- Self-check rel L2 1.63-1.69e-3 on all 4 cases, including B=33.
- `self-check passed: CUDA split-KV path on full decode, 1 < batch <= 64`. There was no fallback.
- Only full decode_16/32 take the path. The sliding, decode_1 and verify shapes ran master and serve as controls; they are all flat within ±1.5% (sliding verify_16 is +2.7%, which is noise).

**Replaced kernels (µs).**

Incumbent: paired FA3 main + FlashAttnFwdCombine + pad/unpad elementwise + hidden (FA3 prepare, FusedRMSNormGate, zero-fill).

New: `splitkv_attn_kernel` + `splitkv_combine_kernel`.

| shape | incumbent | new A + B | saving | Δ shape (bench) | KV floor | new / floor | KV BW A vs FA3 |
|---|---|---|---|---|---|---|---|
| full decode_16 | 146.8 + 4.3 + 4.4 (+hidden) ≈ 155-165 | 112.0 + 8.1 = 120.1 | ~ −35 to −45 | 364.9→329.1 (−35.8, 1.128x) | ~63 | 1.9x | 2.9 vs 2.2 TB/s |
| full decode_32 | 289.0 + 4.8 (+combine, +hidden) ≈ 300-305 | 249.5 + 9.4 = 258.9 | ~ −41 to −46 | 535.8→490.2 (−45.6, 1.103x) | ~126 | 2.05x | 2.6 vs 2.3 TB/s |

**Milestones:**
- M1 (correct, 13/13): met.
- M2 (at or below the incumbent): met on both shapes.
- M3 (≤1.5x floor, i.e. ≤95 / ≤189 µs): not met.

The headline was 1.186x against its base's 1.166x. It was filed "neutral" only because best moved to 1.228x (dense-chain landed) during the run. Dense-chain changes disjoint kernels, so the two stack.

**Where the time went:**
- Kernel A is 93-96% of the replaced time.
- It reaches 2.6-2.9 TB/s on the row gather, which is better than FA3's ~2.2 but still ~2x the floor. Bandwidth drops at B=32: a persistent 2×SM grid over 1024-token chunks, with agentic spread imbalance at the tail.
- The combine (8-9 µs) is twice FA3's combine plus the gate's cost. It does the LSE merge + RMSNorm + gate and could absorb more.

**Verdict: PROMISING.** Stage 2:
1. Re-stack onto the current master. This is running now as `full-decode-splitkv-restack-on-pdl`. It is the cheapest way to bank the ~36-46 µs at full decode_16/32.
2. Raise kernel A's bytes in flight: a deeper ring, or a smaller chunk at B=32 so the tail drains, to get it to ≥3.5 TB/s, about 95 / 190 µs (M3).
3. Extend it to full verify (queued `full-verify-cuda-splitkv-stage2`), where paired FA3 is still 153/296 µs.

---

## 3. triton-flash-decode-full-paired-kv (neutral) — job e82b11f9d9f64b23, base 0730135cdcac43b7

**Path:**
- The log tail has only profiler USDT lines. This path has no self-check or fallback: Triton runs unconditionally on full decode (q_len=1).
- 13/13 tests passed. `_full_decode_split_kernel` appears in every full-decode profile, so the path ran.

**Replaced kernels (µs).** Incumbent: FA3 main + FlashAttnFwdCombine. New: `_full_decode_split_kernel` + `_full_decode_combine_kernel` (the combine is in the hidden tail, a few µs).

| shape | FA3 main + combine | new k1 (+k2 hidden) | stage-1 bar | Δ shape (bench) | KV floor | k1 BW |
|---|---|---|---|---|---|---|
| full decode_1 | 18.4 + 6.8 = 25.2 | 21.2 + ~3 | ≤25 | 183.8→183.0 flat | 4.3 | — |
| full decode_16 | 188.8 + ~4 | 200.8 + ~4 | ≤150 | 433.0→442.4 (−1.3%) | ~68 | 1.63 TB/s (FA3 1.74) |
| full decode_32 | 374.4 + ~4 | 394.8 + ~4 | ≤300 | 601.7→637.4 (−1.3%) | ~137 | 1.66 TB/s (FA3 1.75) |

**Where the time went:** kernel 1 is gather-latency bound. It has one 16 KB K+V tile in flight per CTA at ~2 CTAs/SM (272-288 CTAs), so it reaches the same ~1.65 TB/s as FA3 even with the 128-byte paired-row reads. The paired-row idea itself was not the lever.

**Verdict: DEAD.** M2 was missed on both targets, and it is ~2.9x the floor. The CUDA split-KV kernel (rewrite 2) is the better line: on the same KV it runs 112.0 / 249.5 µs against this kernel's 200.8 / 394.8 µs. Do not mint a stage 3 of the Triton line.

---

## 4. triton-flash-decode-full-oversub-kv-prefetch (regression) — job e55908a05eec4000, base = rewrite 3 (e82b11f9d9f64b23)

**Path:**
- No self-check or fallback lines; same as stage 1.
- 13/13 passed. Kernel 1 + the visible combine are in the profiles, so the path ran.

| shape | stage-1 k1 | stage-2 k1 + k2 | bar | Δ shape (bench) | k1 BW |
|---|---|---|---|---|---|
| full decode_1 | 21.2 | 26.8 + hidden | ≤25 | 183.0→187.7 (−1.7%) | — |
| full decode_16 | 200.8 | 245.3 + 4.6 = 249.9 | ≤150 | 442.4→485.5 (0.886x) | 1.34 TB/s |
| full decode_32 | 394.8 | 478.2 + 3.8 = 482.0 | ≤300 | 637.4→718.8 (0.864x) | 1.37 TB/s |

**Where the time went:** kernel 1 is 21-22% slower at every batch size, including B=1, where S is unchanged at 128. That rules out the oversubscribed grid as the main cause. The common change is the explicit one-tile K/V prefetch with `num_stages=1`:
- The loop carries K and V tiles for both the current and the next tile (uint8 [64,128] each, about 32 regs/thread per tile pair at 128 threads), which most likely cut residency.
- `num_stages=1` removed whatever pipelining Triton's `num_stages=3` was doing on the stage-1 loads.

Combine cost at S=64/32 is small (3.8-4.6 µs) and is not the problem. There is no numRegs in the log to confirm the residency drop.

**Verdict: DEAD.** It is worse than stage 1, which was itself worse than FA3. The Triton flash-decode line should be closed in favour of the CUDA split-KV kernel.

---

## 5. moe-chain-cuda-swapab-expert-once-large-t (FAILED, 0/0 tests) — job 1bba1bab70da4f61

**Path (log):**
```
[moe-cuda-large-t] load_inline compiled in 41.6s
kernel moe_route_list: numRegs 22, local 0 · moe_w13_silu_swapab / moe_w2_swapab: numRegs 148, local 0, dyn smem 221184 B (attrs ok)
retiled 32 experts losslessly · grid 132 persistent CTAs x 256 threads, chunk 64 routes
OFF: compile 41.6s, max row rel nan, setup raised AcceleratorError("CUDA error: an illegal instruction was encountered ... 715 CUDA_ERROR_ILLEGAL_INSTRUCTION")
PATH CHOSEN = MARLIN ...
Traceback ... prepare() -> torch.cuda.synchronize() -> AcceleratorError: illegal instruction (Sticky error detected)
```

**Failing test:** none ran (`tests 0/0`, returncode 112). The run died in `prepare()`.

The order of events:
1. The first self-check case (T=40) ran `_marlin_experts`, then `_lt_experts` twice, then `torch.cuda.synchronize()`.
2. No `self-check T=40` line was printed, so the fault came from the first batch of new-kernel launches.
3. The Python `except` set the path OFF.
4. The error is sticky (715), so prepare's final `torch.cuda.synchronize()` re-raised it and the worker died.

The fallback design cannot catch this class of fault. The hypothesis' own Uncertainties had flagged that risk.

**Likely cause:**
- The same swap-AB mainloop was ported verbatim from moe `cuda-nvfp4-swapab-mma-persistent-small-m`. That sibling died the same way: `muse_moe_w4a16/hypotheses/archive/cuda-nvfp4-swapab-mma-persistent-small-m/8f198afc3bf14f6f.log`, "CUDA path setup raised: ... illegal instruction". The sibling had no `moe_route_list`, so the fault is in the shared GEMM body/helpers. It is not in the route list.
- These PTX forms in it have passed on the judge in other TUs:
  - `cp.async.cg/ca`, `ldmatrix.x4` and `mma.sync m16n8k16 bf16`, all in dense-chain
  - `cvt.rn.f16x2.e4m3x2`, in splitkv
- These forms are unique to the failing pair and have never run on the judge:
  - `createpolicy.fractional.L2::evict_first` + `cp.async.cg...L2::cache_hint` with the 64-bit policy operand
  - the inline `fma.rn.bf16x2` dequant (`mul_bf16x2` with a −0 addend)
  - 221 KB dynamic smem at 256 threads
- An illegal-instruction trap (rather than an illegal-address one) points to an instruction or encoding problem, or to a `.sync.aligned` collective executed by a non-converged warp. The `f < me_nf` / item-end branches look CTA-uniform on paper.

The exact opcode cannot be pinned from this log.

**Verdict: FIXABLE (a bug, not a design verdict).** No timing exists, so expert-once swap-AB at T=128/256 is still untested against Marlin's 254-443 µs. Stage-2 fix, in order:
1. **Make the self-check fault-safe.** Run the first launch of each new kernel in a `torch.multiprocessing` spawn child (or a subprocess with the same TU cache) and return OK/err, so a sticky fault falls back to Marlin instead of killing the run. The dense-chain and splitkv TUs could also have died this way.
2. **Name the kernel.** In the self-check, call `torch.cuda.synchronize()` and print after each of `moe_route_list`, w13 and w2.
3. **Swap the two never-judge-proven forms for proven ones:** plain `cp.async.cg.shared.global [d],[s],16` (drop createpolicy/cache_hint), and `__hmul2` on `__nv_bfloat162` (or `fma.rn.bf16x2` with a +0 addend via the intrinsic) in `dequant8`. Optionally add a 1-CTA micro-kernel self-test of `dequant8` + one mma tile before the full launch.

The same fix applies to the moe sibling.

---

## Addendum: the load_inline gencode check (coordinator request)

The concern: `cpp_extension` adds a plain `compute_90` gencode next to `-arch=sm_90a`, so ptxas rejects wgmma or setmaxnreg for `.target sm_90` and the run silently measures the fallback. I checked each decode CUDA build for this.

| build | log build line | ptxas / gencode errors | wgmma / setmaxnreg in patch | did the new kernel run? |
|---|---|---|---|---|
| small-t-launch-minimal-dense-chain-pdl | `load_inline compiled in 41.5s` | none | 0 | **Yes.** `PATH CHOSEN = DENSE-CHAIN at T <= 32`, and `dl_pooler_router_kernel` / `dl_route_norm_align_kernel` appear in every T≤32 profile. |
| full-decode-cuda-splitkv-port | `load_inline compiled in 49.6s` | none | 0 | **Yes.** `self-check passed`, and `splitkv_attn_kernel` / `splitkv_combine_kernel` appear in the full decode_16/32 profiles. |
| moe-chain-cuda-swapab-expert-once-large-t | `load_inline compiled in 41.6s` | none | 0 | **It built and launched, then faulted at runtime.** All three kernels report attributes (numRegs 22/148/148). The run died on a 715 illegal instruction during the first self-check launch, so nothing was measured. |

None of the three uses wgmma or setmaxnreg; they are mma.sync / ldmatrix / cp.async code, which is valid for plain sm_90. So none is "UNMEASURED (build)", and the gencode issue does not explain the moe fault.

The prefill fix (`TORCH_CUDA_ARCH_LIST=9.0a` + an `__CUDA_ARCH_FEAT_SM90_ALL` guard) is still worth adding to the moe stage-2 TU as hygiene. It ensures only one sm_90a cubin is built and loaded. Any later wgmma stage (for example `bf16-m32-swapab-wgmma-port`) needs it.

---

## 01:04Z harvest (2026-10-25)

Master for comparison is BEST `stack-align-par-sidestream-sliding1`:
- submit 39bcf05afc4e4dd3: 1.2521x
- check cc92097b64784c0f: 1.2495x [1.231, 1.268]. This is the result file named in BEST.json. `BEST.result.json` holds a stale baseline job (3c3181b1fce24b9a).

Per-shape figures are benchmark medians ± err. z is Δ / √(err_new² + err_ref²). The paired per-shape speedup is quoted alongside because raw medians drift between runs.

### full-decode-splitkv-restack-on-pdl — job c243a29f26dc4725, neutral 1.268x [1.249, 1.287]: RAN, WON on its two target shapes. LAND CANDIDATE.

**What ran (log):**
- Both TUs built: `[small-t-dense-chain] load_inline compiled in 50.0s` and `[splitkv] load_inline compiled in 49.0s`. There is no arch line; this is mma.sync/cp.async code with no wgmma, so plain sm_90 is fine.
- Dense chain: `PATH CHOSEN = DENSE-CHAIN at T <= 32, MASTER above`, pdl=1. P timing: 20.6 µs (3.67 TB/s) at T=1 and 25.9 µs (2.92 TB/s) at T=32. Align-parallel ON with 0/18 mismatches.
- Split-KV self-check: kernel A and kernel B ok on all 4 cases (B=5 and B=33, kv_scale 1.0 and 0.25). rel L2 1.63-1.69e-3, finite.
- `[splitkv] ON: self-check passed: CUDA split-KV path on full decode, 1 < batch <= 64`. No fault. This build prints no timing or gate lines.

**Per-shape against master's best run (check cc92097b):**

| shape | new µs | master µs | Δ | z | paired speedup new / master |
|---|---|---|---|---|---|
| **full decode_16** | 306.7±4.0 | 349.5±3.4 | **−42.7 (−12.2%)** | **−8.1** | **1.407 / 1.229** |
| **full decode_32** | 479.6±3.1 | 529.7±9.5 | **−50.1 (−9.5%)** | **−5.0** | **1.301 / 1.178** |
| full decode_1 (B=1, not routed) | 142.7 | 142.9 | −0.2% | −0.2 | 1.271 / 1.266 |
| sliding decode_1/16/32 | 126.1 / 209.8 / 242.3 | 127.5 / 205.0 / 245.8 | −1.0 / +2.4 / −1.4% | ≤1.3 | flat |
| verify, all 6 shapes (not routed) | — | — | −2.1% to +4.7% | ≤1.8 | flat (±0.03) |

- Kernels in the profile:
  - decode_16: `splitkv_attn_kernel` 112.7 µs + `splitkv_combine_kernel` 8.4 µs, replacing FA3 main 151.0 µs + combine
  - decode_32: 248.5 + 8.6 µs, replacing FA3 287.7 µs
- These match the stage-1 port (112.0 / 249.5 µs). The kernel time carried over unchanged onto the PDL master.
- **Only the two full-decode B>1 shapes beat master, and both are well beyond noise (z −8.1 and −5.0).** No other shape moved outside noise. The worst control was full verify_1 at +4.7%, z 1.7; that path is untouched.
- The headline gain (+0.016 to +0.019) is inside the overlapping CIs only because the two winning shapes carry weight 16 each.

**Verdict: LAND CANDIDATE** (a scored submit, which needs the user's approval). It is the cleanest way to bank 43-50 µs per full decode_16/32 call. Kernel A is still ~1.8-2.0x the KV floor, so further stages build on this.

### full-decode-splitkv-kv-bw-3p5 — job 6af3b0e3311943ee, neutral 1.266x [1.247, 1.285]: RAN. The variant sweep did not raise bandwidth, and the static chooser picked a slower variant at B=16. DEAD as a knob.

**What ran (log):**
- Both TUs built: 50.2s and 51.2s. The dense chain and align are the same as above.
- Six kernel-A variants were self-checked. All of them pass all 4 cases with the same rel L2 1.63-1.69e-3, and none faulted.
- Variants: stages 4/5, chunk 1024/512, ahead 1/2. smem ranges from 89 to 115 KB, and **every variant runs at 2 CTAs/SM**.
- `PATH CHOSEN = full decode batch 2..23: S5/C512/A2; batch 24..64: S5/C1024/A2`. This choice is a **hard-coded `_SPLITKV_SEL`**; the timing pass "ranks the variants for the log, it does not choose them".
- Prepare-time kernel-A timing (CUDA events, warm L2, synthetic agentic lengths, ~74K-token mean):

| B | S4/C1024/A1 (stage 1) | S5/C1024/A1 | S5/C1024/A2 | S4/C512/A1 | S5/C512/A1 | S5/C512/A2 | combine C1024 / C512 |
|---|---|---|---|---|---|---|---|
| 16 | 94.0 µs, 3.04 TB/s | 93.9, 3.04 | 93.8, 3.04 | 104.0, 2.74 | 104.2, 2.74 | **104.5, 2.73 ← CHOSEN** | 11.2 / 11.1 |
| 32 | 215.1, 2.81 | 214.9, 2.81 | **215.2, 2.80 ← CHOSEN** | 239.4, 2.52 | 238.8, 2.53 | 239.0, 2.53 | 11.3 / 11.2 |

- A 5th stage and 2-ahead prefetch changed nothing (±0.3 µs). Chunk 512 cost 10-11%. **No variant reached 3.5 TB/s.** The ceiling is ~3.0 TB/s at B=16 and ~2.8 at B=32, at 2 CTAs/SM.
- So the ring depth is not the limiter. Occupancy and page-gather latency are the likely limits, along with the B=32 tail.

**Per-shape against master (check cc92097b), and against the restack:**

| shape | new µs | master | Δ vs master (z) | restack | kernel A (profile) |
|---|---|---|---|---|---|
| full decode_16 | 334.9±3.5 | 349.5 | −14.6 (−4.2%, z −3.0) | 306.7 | **131.1 µs (C512) vs 112.7 restack: +18 µs lost to the chooser** |
| full decode_32 | 473.3±2.9 | 529.7 | −56.4 (−10.6%, z −5.7) | 479.6 | 250.6 vs 248.5: flat |
| sliding verify_32 (untouched) | 634.8 | 592.5 | +7.1% (z 6.7) | 600.8 | paired 1.175 vs 1.176: run drift, not the change |

**Verdict: DEAD** as a config knob. Do not land it, because it is worse than the restack at decode_16. A further kernel-A stage needs a structural change, such as more warps or bytes in flight per SM at higher occupancy, or a different gather. It also needs a timing-driven or corrected variant choice, keeping C1024 at every batch.

---

## 01:36Z harvest

Master for comparison is BEST `stack-align-par-sidestream-sliding1`, check cc92097b64784c0f, 1.2495x, which is the result file named in BEST.json. z is Δ / √(err_new² + err_ref²).

Both runs below are on **stale bases** that lack master's dense chain. Their profiles have no `dl_pooler_` kernel and instead show an `nvjet 128x8` at ~32 µs. So their decode_1 and verify_1 rows are +4% to +15% worse than master for reasons unrelated to the change. Only the kernel rows of the changed path count.

### full-verify-cuda-splitkv-stage2 — job e9d5eeffddef4f70, regression 1.1559x [1.139, 1.173]: RAN, LOST on its targets. DEAD.

**What ran (log):**
- Build flags were `-O3 -arch=sm_90a -lineinfo` with TORCH_CUDA_ARCH_LIST=None. The code is mma.sync and cp.async with no wgmma, so this build is fine. It compiled in 41.6 s.
- Stage-1 decode self-checks passed at B=5 and B=33 (rel L2 1.63-1.69e-3), and the run reported `PATH CHOSEN: full decode CUDA split-KV (stage 1)`.
- The verify kernel uses 124 regs, 0 local, 90640 B dyn smem and 256 threads. Its 4 verify self-checks passed (B=5 and B=33, rel L2 1.64-1.67e-3), and the run reported `PATH CHOSEN: full verify CUDA split-KV (stage 2)`. No fault.
- There were no timing or gate lines in prepare, so the path was chosen without being timed.

**Per-shape (target rows) against BEST:**

| shape | master µs | new µs | Δ% | z | kernels (profile) |
|---|---|---|---|---|---|
| **full verify_16** | 563.7 | 682.7 | **+21.1%** | +20.9 | FA3 153.2 + combine 5.8 = 159.0 → verify A 251.7 + combine 19.3 = 271.0 (**1.70x slower**) |
| **full verify_32** | 861.3 | 1126.7 | **+30.8%** | +38.1 | 289.4 + 7.7 = 297.1 → 506.0 + 26.8 = 532.8 (**1.79x slower**) |
| full decode_16 / 32 (stage 1) | 349.5 / 529.7 | 327.7 / 509.4 | −6.2 / −3.8% | −5.4 / −2.0 | splitkv 111.3 / 248.3 vs FA3 151.0 / 287.7 (this is the restack land candidate already read at 01:04Z) |
| full verify_1 (B=1, not routed) | 192.4 | 216.1 | +12.3% | +4.8 | stale base (nvjet 33.5 vs pooler) |

**Verdict: DEAD.** With q_len 8 and GQA 8, the verify tile is 64 rows per kv head, and FA3's wgmma beats the mma.sync split-KV design by 1.7-1.8x. The decode split-KV kernel is gather-bound, but verify is MMA-bound. Do not mint a verify stage 3 on mma.sync.

### moe-chain-cuda-swapab-expert-once-fix-inproc — job 4ebd055adab34ecd, regression 1.1651x [1.148, 1.183]: BUILT and RAN every kernel with NO FAULT, but the self-check missed its own bound, so it ran Marlin everywhere. UNMEASURED. NEXT STAGE (bound fix).

**What ran (log):**
- The build used TORCH_CUDA_ARCH_LIST=9.0a with `-gencode=arch=compute_90a,code=sm_90a`, `-O3 -lineinfo`, and compiled in 30.8 s. torch was 2.14.1+cu130, capability (9, 0).
- Every kernel reported binaryVersion and ptxVersion 90. `moe_w13_silu_swapab` and `moe_w2_swapab` use 148 regs, 0 local and 221184 B dyn smem (the opt-in max is 232448).
- The arch probe reported `__CUDA_ARCH__ 900, __CUDA_ARCH_FEAT_SM90_ALL 1`.
- These canaries ran ok:
  - cvt e4m3x2
  - dequant8 (__hmul2)
  - ldmatrix.x4 + mma.sync, with tile rel 3.0e-8
  - cp.async at 221 KB × 256 threads
- **The 715 did NOT recur.** Every launch printed `launching` and then `ok`:
  - `moe_route_list`, `moe_w13_silu_swapab`, `moe_w2_swapab` and `post_expert_norm_weighted_sum` at the T=33 canary (62 local routes, finite)
  - the Marlin reference and two full chains at T=40
- The old run (1bba1bab) faulted in the first prepare self-check launch but never named the kernel, because it had no per-launch prints. This run changed three things at once:
  - plain cp.async.cg instead of the createpolicy cache-hint form
  - __hmul2 instead of fma.rn.bf16x2 with a −0 addend
  - an sm_90a-only build

  So the fix cannot be attributed to one of them. The cache-hint cp.async is the likeliest, since the reviewer called __hmul2 an implausible cause.
- **Self-check at T=40:** act row rel 8.79e-3, **y row rel 1.07e-2 (bound 1e-2)**, out global rel 6.00e-3, finite, ids equal, repeat identical. The y row missed the bound by 7%, giving `OFF ... max row rel 1.075e-02` and `PATH CHOSEN = MARLIN for tokens > 32; MARLIN at tokens <= 32`. The T=128/256 kernel timings were never printed.

**Per-shape:** this is Marlin on a very stale base. It has no dense chain, and full decode still uses FA3 (149.5 µs). Every shape is +3% to +14% vs master: decode_1 +13.6% (z 16), verify_16 full +4.6% (z 4.1), verify_32 full +4.5% (z 5.3). This is base staleness; no kernel of the change was timed. The Marlin rows at verify_16/32 (295.6-305.7 / 425.4-475.2 µs) are the untouched control.

**Verdict: FIXABLE, next stage.** The design is still untested against Marlin's 254-443 µs at verify_16/32. The 1e-2 row bound is self-imposed and compares against Marlin, which is itself approximate. The judge allows 3-15% on the layer output, with rows up to 5× that.

Stage 3:
1. Rebase onto current master (`stack-align-par-sidestream-sliding1`).
2. Gate against an fp32-dequant reference rather than Marlin, as the moe bs1 kernel does: fused 2.9e-3 vs master 4.9e-3. Use a bound derived from Marlin's own error against that reference (e.g. ≤ 1.5× Marlin's).
3. Print the T=128/256 CUDA-event timings before the decision.
4. Keep `okc check` only.

## 03:41Z harvest

References:
- BEST `stack-align-par-sidestream-sliding1`. The result file is check cc92097b64784c0f, 1.2495x; the submit 39bcf05afc4e4dd3 gave 1.2521x.
- Control: `dl-p-wide-tile-t16-t32`. It leaves T≤8 on master P (PATH CHOSEN), so its decode_1 rows show this batch's drift.
- decode err is 0.5-8 µs, so only the decode_1 rows (err ~0.6) resolve 2% moves.

decode_1 rows across today's runs (µs):

| run | sliding decode_1 | full decode_1 |
|---|---|---|
| BEST submit 39bcf05 / check cc92097 | 127.1 / 127.5 | 145.9 / 142.9 |
| p-wide-tile (control: T=1 untouched) | 126.5 | 144.1 |
| **route-warp-topk-pdl** | **124.1** | 141.2 |
| **boundaries-unpooler-pdl** | **123.9** | 140.5 |
| dense-chain-stage2 | 125.8 | **139.3** |

### dl-route-warp-topk-pdl — job b471a950805c4053, check 1.2599x [1.241, 1.279]: RAN. Small real gain at T=1; NOT a candidate alone; STACK.

- Self-checks passed: P/R at T=1/4/8/9/16/32, R alone on no-local / one-hot-local / near-tie, and R variants 0/1/2. PDL probe → pdl=1. `PATH CHOSEN = DENSE-CHAIN at T <= 32, MASTER above`.
- Prepare: R at T=1 went from 7.01 to 5.79 µs. The P→R pair with pdl went from 27.87 to 26.73 µs (−1.1). T=8 and T=32 were similar (−1.0 to −1.5 µs).
- Shapes vs BEST: sliding decode_1 −3.4 µs (−2.6%, z−3.3) and full decode_1 −1.8 (−1.2%, z−2.0). Against the control that is −2.4 and −2.9 µs. Every other shape is within err (z ≤ 2.7). full verify_1 at +6.8% (z+2.7) has err 3.3 µs; its Marlin row is lower (74.6 vs 86.2), so that is noise.
- Substituting the decode_1 rows into BEST gives 1.2548 (+0.005), below one check's resolution.

### boundaries-unpooler-pdl-small-t — job 1dfe83fdbeea458a, check 1.2559x [1.237, 1.275]: RAN. Small gain at T=1; the selector shipped a slower kernel; STACK the good half.

- Build: `-arch=sm_90a`; the log notes griddepcontrol, mma.sync and cp.async only. Residual-coefficient probes matched the reference formula exactly. Post-norm rounding probe: attn bf16, ffn fp32. Max errors were 1e-5 to 2.6e-3, and the chain graph matched.
- `PATH CHOSEN = boundaries CUSTOM at T <= 32; unpooler CUSTOM per NF {1: True, 2: False, 4: False}`.
- **The boundary kernel lost its own timing and was chosen anyway.** `dl_boundary_kernel` ran 3.91/4.09/4.20 µs vs SGLang `FusedPostNormResidualPreNorm` 2.79/2.92/3.07 µs at T=1/8/32. The unpool kernel wins only at NF=1: 6.5 vs 9.2 µs.
- Shapes vs BEST: sliding decode_1 −3.6 µs (z−3.6) and full decode_1 −2.5 (z−2.8). sliding verify_1 −9.2 µs (−4.8%, z−2.2; err 2.8). Others within err.
- Substituted headline 1.2542 (+0.005).
- Next: keep `dl_unpool_kernel` at NF=1 only, and put the boundaries back on SGLang's fused kernel. Each boundary costs +1.1 µs and there are two per layer.

**Stack proposal (one check):** route-topk (R) + unpool NF=1 only, on BEST. The prepare figures add up to −1.1 (R) −2.7 (unpool) ≈ −3.8 µs per T=1 shape. The two runs measured −2.4 to −3.6 µs each against the control. Expect ≈ +0.008-0.010 headline. That needs a submit-grade A/B, not a check headline.

### dense-chain-stage2-attn-side-pdl — job b31b8453208a471c, check 1.2253x: RAN. Won at T=1 full, LOST at T=16/32. Gate to T=1 or DEAD.

- FP8 encode uses `__nv_cvt_float2_to_fp8x2` SATFINITE through bf16. Max errs: q 1.2e-2, k 1.3e-2 (FP8), o 3e-4. Graph replay matched.
- `PATH CHOSEN`: sliding decode_1 = Q raw + `_sliding_forward` + O raw. Every other T≤32 shape = Q + FA3 + O. Above T=32, master.
- Prepare: Q at T=1 8.4 µs (4.5 TB/s) and O at T=1 6.4 µs. **O at T=32 takes 40.8 µs (0.41 TB/s).**
- Shapes vs BEST: full decode_1 −3.7 µs (−2.6%, z−3.9) is a gain. Losses: sliding decode_16 **+9.2%** (z+4.4; `dl_o_kernel` 26.8 µs), sliding decode_32 **+13.5%** (z+6.5; `dl_o_kernel` 46.0 µs) and full decode_32 +6.0% (z+2.6; `dl_o_kernel` 53.1).
- The O kernel does not scale past T=1. Next: gate stage 2 to T=1 (full only; sliding decode_1 was flat at −1.4%), or drop it. Do not spend more on the T>1 O kernel without a new tile design.

### dl-p-wide-tile-t16-t32 — job 9ad8e2cf869c4b4f, check 1.2457x: RAN, flat. DEAD.

- Wide P: 80/106 regs at NF=2/4, 0 spill, 3/2 CTAs/SM (more warps resident). The full self-check passed.
- **The selector picked WIDE at T=9..32 although its own CUDA-event timing said it was slower**: T=16 old 23.4 vs wide 24.9 µs, T=32 old 26.4 vs wide 28.9.
- Shapes at T=16/32 are within err: sliding decode_16 +1.2%, sliding decode_32 −2.2% (z−1.0), full rows ≤0.3%.
- More resident warps did not raise P's bandwidth: 3.0 TB/s at T=16 vs 3.2. Dead.

## 04:36Z harvest

References: new master `full-decode-splitkv-restack-on-pdl` (check c243a29f26dc4725, 1.2680). Four of the six runs were built on the OLD base `stack-align-par-sidestream-sliding1`, which lacks the CUDA split-KV. Their full decode_16/32 rows read +7-15% against master, and their headlines (1.267-1.275) are inflated by SGLang-baseline drift: verify_32 paired ratios run 1.02-1.07 on untouched rows. Ignore those headlines. The decode_1 rows (err 0.4-0.7 µs) are the only rows that resolve a 2% move. Batched rows (err 3-8 µs) need more than 4% and a kernel change before I read them.

decode_1 rows (µs; master = 126.15 sliding / 142.70 full):

| run | base | sliding decode_1 | full decode_1 |
|---|---|---|---|
| master c243a29 | – | 126.15 | 142.70 |
| sliding-batched-triton-b16-band (control: T=1 untouched) | master | 126.22 | 143.06 |
| **w2-marlin-atomic-add-t1-port** | master | **123.88 (−1.8%, z −2.5)** | **139.18 (−2.5%, z −4.0)** |
| **stack-route-topk-pdl-unpool-nf1** | old | **122.44 (−2.9%, z −4.3)** | **136.93 (−4.0%, z −7.3)** |
| dense-chain-stage2-attn-side-pdl-t1-gate | old | 125.21 (−0.7%, flat by design) | **137.98 (−3.3%, z −5.8)** |
| fused-preattention-paired-pad-restack | old | 125.43 (−0.6%) | **138.37 (−3.0%, z −6.5)** |
| moe-chain s3 | old | 127.64 (+1.2%) | 142.74 (0.0%) |

### w2-marlin-atomic-add-t1-port: job 7a9f5de5aecc4aba, check 1.2856 [1.266, 1.305]. LAND-CANDIDATE.

- `[w2-atomic] PATH CHOSEN = ATOMIC w2 at T=1 (worst rel-L2 vs master 0.00556); use_atomic_add=True use_fp32_reduce=True memset in path: False; T>1 MASTER`.
- Built on master, and both decode_1 rows moved: −2.3 and −3.5 µs. In `runs_ns` the medians were 130.8 vs 134.1 (sliding) and 144.3 vs 150.0 (full). Both distributions are unimodal; the count of runs above median×1.08 was 3 vs 4 and 2 vs 2. Unlike moe, this is a shift of the whole distribution, not a removed slow mode.
- Marlin's summed kernel time on the full decode_1 row went from 40.3 to 35.8 µs. On sliding decode_1 the profile reads 42.3 vs 40.9 (+1.4, profile noise). The shape median still fell.
- Every other row is within err. Sliding verify_32 has pr 1.054 at a flat median, which is baseline drift.
- The headline CI overlaps master's [1.249, 1.287]. Substituting only the decode_1 rows gives about 1.272-1.276. Promotion should rest on the two decode_1 rows; a scored submit is fine because the change has no failure mode beyond the moe-proven flag.

### stack-route-topk-pdl-unpool-nf1: job 87211e1362b64c4c, check 1.2671 (old base). STACK-INTO-NEXT (rebase onto master).

- R selector: `new R pdl=1 87.31 us vs master R 92.96 us ... WINNER = NEW R`.
- Unpool selector: `custom 13.8 us vs master 19.8 us (T=1+8): WINNER = CUSTOM dl_unpool_kernel`. The resulting path: `CUSTOM dl_unpool_kernel at T <= 8 (NF=1) ... boundaries stay SGLang`. Both selectors now assert the timed winner, so the 03:41Z bad-selector lesson was applied.
- Both decode_1 rows had the largest moves of the batch: sliding −3.7 µs and full −5.8 µs against master. Sliding verify_1 −5.9% is within err (2.5 µs).
- The run lacks the split-KV, so full decode_16/32 read +15% and +10%. It cannot land as is.

### dense-chain-stage2-attn-side-pdl-t1-gate: job c6cb53d8ebc2478a, check 1.2717 (old base). STACK-INTO-NEXT, full decode_1 only.

- `[dense-chain-s2] compare T=1 full: custom Q+O 18.6 us vs master attention side (sans FA3) 27.3 us -> custom wins`. Q 8.3 µs (4.54 TB/s) and O 6.5 µs with pdl.
- PATH CHOSEN: full decode_1 runs Q, then FA3 over every key, then O. Sliding decode_1 stays on master (the drift control), and T>1 stays on master.
- The judged full decode_1 row moved 142.70 → 137.98 (−3.3%, z −5.8), and the sliding control is flat (−0.7%). That is a clean attribution. The T>1 losses from the 03:41Z run are gone: full decode_32 −6.5% vs its base, and every T>1 row is within err of master except full decode_16/32, which lack the split-KV.
- **Conflict:** master's full decode_1 path is FA3, and so is this one, so there is no clash with the split-KV gate (B>1). It does overlap fused-preattention below: both replace pre-attention at T=1. Pick dense-chain-s2: it has the larger measured margin (−8.7 µs in the side-by-side), and fused-preattention moved no other row beyond err.

### fused-preattention-paired-pad-restack: job 5e9b5464d1444708, check 1.2745 (old base). DEAD as a separate line; superseded by dense-chain-s2 at T=1.

- `PATH CHOSEN = DENSE-CHAIN at T <= 32, MASTER above`. The archived log tail does not contain the pre-attention path line.
- full decode_1 −3.0% (z −6.5) is the only row that moved and resolves. Sliding decode_32 −5.2% (z −3.2; err 3.2 µs) is inside the batched rows' resolution and could be drift. The batched paired rows (the reason for the paired pad) did not move beyond err: sliding decode_16 +0.7% and the full rows are confounded by the missing split-KV.
- Keep its patch on file. If a later stage needs the paired pad for full verify, rebase it there.

### moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound: job 51d047cbc5ff450a, check 1.064, regression. DEAD (design).

- The fp32-reference bound fix worked: `worst out error ratio new/marlin vs fp32 0.82 (bound 1.5)`, all checks passed, `PATH CHOSEN = CUDA for tokens > 32; MARLIN at tokens <= 32`.
- **Why it was slow:** the path printed its own timing as "print only" and shipped anyway.
  - T=128: `moe_w13_silu_swapab 421.8 + moe_w2_swapab 240.0 ... route+w13+w2 670.3 us; marlin path total 319.5 us`.
  - T=256: 868.9 vs 442.4 µs.
  - The CUDA path was 2.0-2.1x slower than Marlin and was chosen on correctness alone.
- On the judge, verify_16 and verify_32 regressed +37% to +77% against master (sliding verify_16 743.6 vs 420.4 µs; `moe_w13_silu_swap` alone 405-590 µs). The T≤32 rows equal master because Marlin ran there.
- Together with the 01:36Z small-T result (1.7-2x slower than Marlin), the mma.sync swap-AB expert kernel loses at every T. Close the line.

### sliding-batched-triton-b16-band: job 02bf417257e14a3d, check 1.2743, on master. DEAD.

- Built on master. `_sliding_fused_kernel` ran on sliding decode_16 at 14.6 µs, against master FA3's 13.7 µs.
- The targeted row went 209.82 → 215.34 µs (+2.6%, z +1.4). It did not improve, and the kernel itself is 0.9 µs slower.
- Full decode_32 −7.0% (446.0 vs 479.6) is drift: its `splitkv_attn_kernel` time is identical (248.3 vs 248.1 µs) and the change does not touch that row. The verify rows at +3-5% are drift too.
- Sliding decode_16 should stay on FA3. The attention problem's C=8 cluster at B=16 also lost (+7%, 01:36Z), so no tested kernel beats FA3 in the B=2..16 sliding band.

**Recommended next decode stack (one check):** master + w2-atomic T=1 + route-topk R + unpool NF=1 + dense-chain-s2 at full decode_1. The changes touch disjoint kernels: Marlin w2 flags, the route/align kernel, the unpool GEMM and the T=1 full attention side. Stack w2-atomic first, because it alone is a land candidate today. Expect sliding decode_1 at about 120-122 µs and full decode_1 at about 131-134 µs. Judge on the two decode_1 rows against this control (126.2 / 143.1).
