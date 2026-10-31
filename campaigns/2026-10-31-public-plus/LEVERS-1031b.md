# Lever round 2, 10-31: judge ranking (workflow wf_c7b20f1e-b31)

Three effort-max finders worked on the H100 (decode on GPU 1, moe on GPUs 3/4, attention on GPU 6), and one max judge ranked their levers. The judge used no H100 time. Nothing in musespark was edited, and git is clean at 991a2787. Judge evidence comes from read-only `okc status --json` on 8 unchanged decode reads and 9 moe reads.

## Verdict

| # | Lever | Problem | Value if it works | p | Value x p | Build | Compliance | Call |
|---|---|---|---|---|---|---|---|---|
| 1 | **ET3**: early PDL triggers on three sk2 producer edges | decode | +0.45% (range +0.33 to +0.75) | 0.65 | **~0.29%** | knob: 1 CUDA line + ~6 Python lines | clean (scheduling only) | **RUN**: audit, then 2 decode checks today. Default: rider. |
| 2 | **B4**: the public decode's B4FIN final as moe's own d32 build | moe | +0.13% | 0.7 | ~0.09% | small: ~70 copied lines, 1 extra compile (+~45 s prepare) | clean (FP32 order only) | **PARK**: rider only. It supersedes round-1 SHT32. |
| none | attention | attention | none | none | none | none | none | No lever. I agree with the finder. |

- No gray area was found in either lever, and nothing was rejected for compliance.
- Neither lever clears the /hypothesize bar, so ET3 runs as a direct rung, as in round 1.
- **Round 2 produced no phase-A-sized decode rewrite.** ET3 reaches tomorrow's phase A only as landed base or as the default rider.

## What I verified

- **Master 991a2787 decode sites:**
  - `SK2_DIAG` and the compiled-out `dgv` (:7727, :7741).
  - `P` (:7802).
  - The thread-0 pre-issue wait (:7828) and the all-thread wait (:7855, ET3's anchor).
  - The late trigger after the main loop (:8042).
  - `bfp__run_sk2` (:9409, `ext.sk2` at :9415) and `_bfp_gemm` (:9831).
  - `_de_source` (:12944) and the narrow override (:12980/:12985).
  - `_gle_source` (:13243). Its anchor at :13273 is a prefix of :7854-7855, so it still matches exactly once after the edit.
  - `gle_o` passes dg=0 (:13521).
  - `PDL_MAX_T 32` (:89), so v16/v32 run no PDL.
  - `PIN_PICK` (:103) pins (8,o), (32,o) and (8,un). The pairs (8,qkv), (16,qkv), (16,o) and (16,un) are raced in prepare (:10747-10790). They resolved to bfp sk2 in 8/8 judge reads.
- **ETDG_FIN patch** (`levdec.py` PATCHES): all 4 old strings occur exactly once in master, at :7855, :9831, :9415 and :12985.
- **Dependents' pre-wait code (the safety argument):**
  - Attention (:1585-1606) issues only page-index cp.async before `pdl_wait()`. The code's comment there: "nothing above reads or writes it".
  - b1/b2 `glue_cluster_boundary` (:12491-12528, b2c POST launcher :11542-11551) loads stream, alpha, beta, alpha2, pre and post. It then runs cluster.sync, then the wait.
  - b2's stream is s1, written by b1 four kernels earlier and complete before the unpooler's own wait returns.
  - `_graft_row_boundary_pre` (:11810-11821) loads only weights before `_gdc_wait`.
  - The only pre-wait global store in this family is the SLICED variant (:11450/:11496). No SLICED=true instance is ever launched.
  - The trigger fires only once every producer CTA has executed it, so every producer CTA is already resident. Cooperative or cluster dependents cannot deadlock it.
  - `griddepcontrol.wait` still waits for the whole producer grid and its writes. Master's own comment at :6253-6254 says so.
- **moe sites for B4:**
  - :150 `MID_MAX_M = 16`; :220 `OKC_WGSFINSS`; :343 `EMAX = 3072`; :369 `dbg`.
  - :2098-2111 phase-B y stores (2106-2107); :2650 the final; :2837 `p.ybuf`.
  - :3335-3338 compiles serially (one `_ext` per build); :3382-3385 routes builds.
  - `sub_B4.py` differs from master by 88 diff lines.
- **H100 runs re-read:**
  - lev8 (`runs/20261031-052400-gpunode06`): v1s x0.9879, v1f x0.9792, d16s x0.9815, d16f x0.9851, d32s x1.0008, d32f x0.9973. Controls: d1s x1.0035, d1f x1.0022. All 12 checks and 13 tests passed.
  - B4 (`20261031-045216`/`044348`): R67 d32 x1.3752/1.3777 against x1.3647/1.3671 (+0.77% / +0.78%); cold x1.0080 / x1.0091; v16 and v32 at x0.9996-1.0009.
- **Public sources:**
  - NEG decode #3 is the same mechanism ("move the PDL trigger of the projection kernels after the weight issue"). It gained at most 0.736 us per real-kernel pair on their H200, a positive but small gain, within their T<=8 scope.
  - Facts §2: a `launch_dependents` at the end of a kernel costs 1.1 us in their synthetic bench.
- **CLAIMS:**
  - ET3 is not on the do-not-re-propose list.
  - The old decode line "[refuted] trigger-first boundary PDL at T=1" (10-27, our old base) does not cover it: ET3 has no T=1 edge, and the finder's T=1 nulls agree.
  - B4 targets the same rows as round-1 SHT32 (parked at 0.06%) and replaces it.

### Corrections to the finder reports

- **ET3, narrow source:** the narrow T<=8 qkv source does set the bit, at T=8, through the :12985 edit. That is the v1 qkv edge. Only the GLE launches never set it.
- **ET3, audit:** the proposed "1000 replays identical to a no-ET build" cannot hold. Master itself is order-nondeterministic in two places: multi-split attention merges and the skinny MoE ss atomics.
- **Elasticities:** headline ≈ weighted geometric mean of row speedups, weights 46 (sliding) / 16 (full) out of 372. So e_i = 100·w_i/(372·t_i) %/us. My reconstruction reads 0.3-0.5% below the reported score, so this is an approximation.
  - Per row: d1s 0.182, v1s 0.116, d16s 0.096, d32s 0.083, d1f 0.057, v16s 0.051, v32s 0.040, v1f 0.037, d16f 0.021, d32f 0.014, v16f 0.013, v32f 0.009.
  - The finder's d16s/d32s/d16f/d32f values were 15-60% off.
  - lev8 at 1:1 transfer: +0.75%.
- **B4, decode lines:** the cited ranges are about 9 lines stale after DPP landed. Correct ranges: defines :3840-3845, Params `sspart` :3997 (unguarded in decode), epilogue partials :5738-5752, final :6293-6340, okc_run :6532-6537.

## Instrument note: the judge profile pass is one eager call (this changes every PDL gate)

**Evidence:**
- In all 8 unchanged decode reads, d1s and d1f show no kernel overlap at all (no "GPU busy" line).
- v1s, v1f, d16s and d32s show only 1.4-3.5 us of total overlap; d16f and d32f show 5.2-10.8 us.
- The H100 graph replay of the same chain has 14.2-15.3 us of PDL wait per row (lev8 nsys).
- The same o kernel reads 7.9-8.3 us at d16f against 6.0-6.6 us at d16s, and 8.1-8.9 us at d32f against 6.6-7.6 us at d32s. It launches early only after the 86-175 us flat attention.

This fits the team's 10-28 note ("torch profiler, one call each"): a dependent overlaps its producer only when Python has already enqueued it, which happens after long kernels.

**Consequences:**
1. **Early-layer PDL edges are invisible in the judge profile.** These are the qkv→attention edge and both v1 edges. The finder's gates ("attn<8,F,15> at v1s ≥ 12 us", "v1s/v1f non-MoE busy") cannot fire, and as written they would KILL a working ET3.
2. **Late edges are visible and give the only H200 speed reading.** These are o→b1 at d16f, o→row boundary at d32f, and unpooler→b2 after the MoE.
3. **d1 per-kernel times carry no PDL wait.** That bears on the parked cgemv question: the d1 qkv (10.0-10.9 us) and gle_pr (17.7-20.6 us) times are real kernel time. The qkv also carries the unexplained 1.5-3.5 us first-kernel penalty.

**Suggested LESSONS line:** "Judge profile = one eager call. Gate PDL levers on late edges and on GPU busy − moe − attn, never on early-edge kernel times or 'total'."

## 1. ET3 (muse_decode_layer): RUN

**Change.** The early trigger goes on three sk2 producer edges:
- the qkv at T=8/16 (dependent: attention);
- the o at T=8/16/32 (dependent: b1, or the row boundary at T=32);
- the unpooler at T=16 (dependent: b2).

| Rows | Edges |
|---|---|
| v1s, v1f | qkv, o |
| d16s, d16f | qkv, o, unpooler |
| d32s, d32f | o |

d1, v16 and v32 run no ET edge. The T=1 sk2 instantiations change only by one never-taken predicated branch.

**Transfer.** This is the launch/latency scheduling class that transferred today (PP and DPP). It is not the bandwidth/L2/plan-race class that did not (DL2H, S6, BFR). The public measured a positive gain of at most 0.736 us per pair on their H200 for the same mechanism.

I estimate the H200 gives 60-75% of the H100's microseconds:

| Row | H100 (lev8, us) | H200 estimate (us) |
|---|---|---|
| v1s | -1.93 | ~-1.2 |
| v1f | -3.22 | -1.5 to -2 |
| d16s | -3.29 | ~-2.0 |
| d16f | -4.29 | -2.2 to -2.5 |
| d32f | -1.15 | ~-0.7 |
| d32s | +0.16 | ~0 |

That gives **+0.45%**. At 0.5 us per edge it is +0.33%; at the public's per-pair cap, +0.49%; at 1:1, +0.75%. With **p 0.65**, value x p ≈ **0.29%**.

**Compliance: in.**
- Only the position of `launch_dependents` moves.
- Every dependent still waits before its first read of producer output.
- No arithmetic, precision, routing, KV or caching change. Every key is still attended, and the FP32 router at T<=8 is untouched.
- No new buffers, so no NEG #11 exposure.
- Still graph-capturable and load_inline. Prepare stays at ~121 s; the file grows ~0.6 KB from 841,361 B.
- Outputs differ from D0 only by the existing atomic-order jitter in the 4th digit.

**First build step.** Use a worktree off 991a2787 with this master edit, not the probe's module global.
1. In bfp_SK2_SRC after :7727, add `#define SK2_ET 1`.
2. After :7855, add:
   ```
   #if SK2_ET
   if (pdl > 0 && (dg & 0x40000000)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
   #endif
   ```
   Put the safety reason in a comment: dependents read only inputs, weights or buffers finished two kernels back before their wait.
3. In `_gle_source`, add `s = s.replace("#define SK2_ET 1", "#define SK2_ET 0", 1)` with `assert count == 1`. This keeps the T<=8 GLE o and pooler-router SASS identical to master. The narrow source keeps ET on, because it carries the v1 qkv edge.
4. Add a module helper:
   ```
   def _sk2_et(layer, m):
       return (layer == "qkv_gate" and m in (8, 16)) or (layer == "o" and m in (8, 16, 32)) or (layer == "unpooler" and m == 16)
   ```
   At :9415 and :12985, pass `dg | (0x40000000 if _sk2_et(layer, m) else 0)`.
   - This avoids `_ETDG`, which leaves a stale value for sk2 calls that bypass `_bfp_gemm` (the prepare paths at :9609/:9747).
5. Drop QET816. It adds a second mechanism (all ring stages before the wait), which public NEG #3 and the finder's UPW dead end both show losing.

**H100 gate (independent audit).** Use one idle gpunode06 GPU from 1/3/4/6; check `nvidia-smi --query-compute-apps` first. Work node-local, through hopper-probe, and clean caches afterwards.
- **Correctness:**
  - 12 rows x 2 seeds and the 13 TASK tests pass.
  - Every error stays within D0's own run-to-run envelope (no bit-identity claim).
  - Stale-read stress: at least 2,000 graph replays each at v1s, v1f, d16s, d16f and d32s, with fresh stream, KV and page-table inputs every replay. Check every replay against the reference.
- **Build and prepare:**
  - GLE SASS byte-identical to master.
  - The T=1 sk2 instantiations differ only by the added predicated instructions, at the same register count.
  - Prepare picks (mm_pick, moe_pick, gluev2, attention plans) identical to master on the same node.
- **Speed:** 2 runs x 8 states x 20 rounds against master.
  - **GO** if pooled v1s ≤ -1.2, v1f ≤ -1.5, d16s ≤ -2.0 and d16f ≤ -2.0 us, with d1s and d1f within ±0.6 us.
  - **KILL** if v1s + v1f + d16s + d16f is above -4 us, or on any failure.

**Judge checks: 2.**

*Bases.* Eight reads share identical T<=32 code (except the d1 MoE build in the 2 DL2H reads): 4b2adae2, de38cec7, 201db0b8, 5b0f8dae, cc512a50, 9d726d84, f078d673, eb4e07ad. The headline base is DPP master: 2.2220 / 2.2367 / 2.1826, mean **2.2138**.

ET3 inflates 'total' through the dependents' waits, so read the "GPU busy" line, never 'total'. Match kernels by template prefix, because the md5 name tags change.

**(1) Live, required in both reads.**
- d16f b1 `glue_cluster_boundary<4, 32, false, false, false, …>` must be **≥ 6.5 us**. Base 3.4-3.6; expected 7.5-9.
- Supporting signals: d32f `_graft_row_boundary_pre` (2 calls) ≥ 11 us (base 7.3-8.0), and d16f/d16s b2 `glue_cluster_boundary<4, 32, true, …>` ≥ 8 us (base 3.6-3.8 / 3.3-3.7).

**(2) Transfer (the one H200 speed reading).**
- Z = Δ[d16f busy − moe − attn] + Δ[d16s busy − moe − attn], using 2-read means.
- Bases: d16f **54.76** (sd 0.58, range 54.2-55.8); d16s **57.18** (sd 0.46, range 56.6-58.0).
- A row counts in a read only if its attention is not inflated in that read: d16f attention ≤ 93 us (base 86.0-92.3), d16s attention ≤ 8.5 us (base 7.7-8.2).
- d16s also needs b2 ≥ 8 us in that read, which shows its edge ran early.
- Expected -1.5 to -2.5 us; SE ≈ 0.6 us.

**(3) Rows.**
- ΔI = 0.116·Δv1s + 0.037·Δv1f + 0.096·Δd16s + 0.021·Δd16f + 0.083·Δd32s + 0.014·Δd32f, in % headline.
- Δ is the 2-read mean minus the 8-read mean. Bases: v1s 106.87, v1f 115.19, d16s 128.98, d16f 205.55, d32s 149.24, d32f 298.79 us.
- Per-read sd 0.68%, so the 2-read SE is 0.54%. Expected -0.45%. The gain itself cannot be confirmed by rows in 2 checks.

**Decision rule:**
- **GO** (scored submit; land if the scored read is ≥ 2.2220). All of these must hold:
  - (1) live in both reads;
  - Z ≤ -1.0 us (≤ -0.7 if only d16f counts);
  - ΔI ≤ 0;
  - headline 2-read mean ≥ 2.2138;
  - d1s/d1f 2-read means ≤ 69.1 / 77.0 us (the 8-read maxima);
  - 13/13 tests.
- **RIDER:** (1) holds, Z ≤ -0.5, no KILL flag, but GO is not met. Carry it in tomorrow's phase-A decode candidate and disclose it in the attempt file.
- **EXTEND:** -0.5 < Z < -0.2. Spend 2 more checks in phase C and pool all 4 (SE 0.41%).
- **KILL** if any of these:
  - (1) fails in both reads (the bit is inert; if it fails in one read only, inspect before deciding);
  - Z ≥ -0.2, meaning the visible edges gain nothing on the H200 (the DL2H/S6 pattern);
  - ΔI ≥ +0.8%;
  - headline 2-read mean < 2.19;
  - d1s or d1f above its maximum;
  - any test failure.

**Phase.** Not a phase-A item. Audit now, then 2 checks today (decode has 13 left). Tomorrow it is either landed base or the default rider.

## 2. B4 (muse_moe_w4a16): PARK, rider only

- **Change.** Port the decode's B4FIN pieces into moe:
  - defines (from decode :3840-3845) after moe :220;
  - the `sspart` Params field after :369, **guarded by OKC_B4FIN** so the 4 existing builds keep their SASS;
  - the epilogue partials (from :5738-5752) after moe :2108;
  - the final (from :6293-6340) at :2650 as `#if OKC_WGS && OKC_B4FIN … #elif OKC_WGS && OKC_WGSFINSS`;
  - the okc_run setup (from :6532-6537) after :2837, also guarded;
  - Python: `B4_MAX_M = 32`, a prepare compile when MAIN is WGS (:3335-3338), and routing of 16 < M ≤ 32 to the new build (:3382-3385).
- **Starting file:** `sub_B4.py`; add the Params guard.
- **Memory:** `sspart` uses ybuf rows 3008-3071 (EMAX 3072). M ≤ 32 uses fewer than 256 rows, so there is no new allocation.
- **Value x p:** d32 -1.0 us (R67: -0.72 and -1.35 us) is +0.13% headline (0.1325 %/us). p 0.7 gives **0.09%**.
  - It is a latency mechanism (one L2 pass, fewer barriers, 128 CTAs instead of 32), and the code already runs on the judge inside the public decode.
  - Decode gains nothing, since it already ships B4FIN.
- **Compliance: in.** Only the FP32 summation order of each route's sum of squares changes, to a fixed tree. Output is BF16, as in SGLang. The build is chosen by shape only.
- **H100 gate:**
  - R67 d32 ≥ +0.5% in 2 runs with v16/v32 within ±0.3%. Already met: +0.77/+0.78%, controls -0.04 to +0.09.
  - At full bytes: check_implementation at bs 17/24/32 and the 5 TASK tests, rel L2 ≈ 2.5e-3 (as master).
  - TINY/TINY2/MID/MAIN SASS identical to master.
  - nodejudge prepare ≤ 240 s. The judge's M1a prepare is 165.6/168.2 s; this adds one serial compile.
  - KILL if R67 d32 < +0.3% or any other row moves more than 0.3%.
- **Judge:** no standalone checks.
  - The 1.0 us gain sits inside the M1a pair's own d32 spread (127.90 / 123.73), and resolving it would take more than 60 reads per arm.
  - As a rider: d32 ≤ 127.9 us in every carrier read and prepare_s ≤ 240 s. The other rows are untouched by construction.
- **Phase:** none. It waits for a moe carrier.

## 3. muse_attention_decode: no lever (agreed)

- The PP family does not carry to the per-request q8 kernel. Decoupling the warpgroups costs more than an exp overlap can save; that ceiling is at most 1-5%.
- The flat q1 kernel sits at the public's 3.8-4.1 TB/s ceiling.
- The per-request q8 epilogue is limited by red volume, and every compliant variant of it measured neutral or worse.
- The finder was right to decline BF16/FP16 split partials, because FA3 merges in FP32.
- Nothing is left to port into ATT7; DPP already landed.

## Finder dead ends: accepted

No dead end is reopened. All agree with CLAIMS, the public NEG files and the run records.

- **decode:** UPW, AP0F/AP0, KVPRE (matches vjeux's retired `OKC_PDLKV` at :169), the T=1 and T=32-qkv edges, glue-launch removal, T=32 pooler+router fusion, MoE pre-wait setup, Triton PDL at T=128/256, sk2 m16/m32 (public NEG #8), the first-kernel penalty, and cgemv (parked; see the instrument note).
- **moe:** NW16, NS4/NS5, FACQ, LLA, LAST1, HF in R67, PAIR, PBLL, FIN2/F1, the WGS A→B ablation, and B4FIN at v16/v32.
- **attention:** NFPP, nfpv, pipeline depth, the merge/finalize knobs, flat q1, and the ATT7 port.
- **R67 proxy:** a credible new gating regime for HBM-bound moe kernels.
  - It reproduces judge row times within 1% at d16/d32/v16/v32 (d1 +6%, v1 +11%) and predicted S6's non-transfer.
  - That is one validation point. Use it as the moe speed gate, and keep cold mode for correctness.

## Plan

1. **Now (phase B):** one agent builds ET3 and a second agent audits it, on one idle GPU. Then 2 decode checks under the rule above, pre-registered in the attempt file with the rules checklist.
2. **Phase C (18-24Z):** spend the 2 extra ET3 checks only in the EXTEND zone. No B4 checks.
3. **Tomorrow, phase A (decode-first):** round 2 adds no rewrite. ET3 enters only as landed base or as the default rider on the phase-A decode candidate. B4 and attention contribute nothing to phase A.

## Artifacts

The judge records I fetched (17 runs, `<job>.json`) and the parsers `jparse.py` and `dstats.py` are in my scratchpad:
- /tmp/claude-1000/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/scratchpad/judge/
- /tmp/claude-1000/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/scratchpad/tools/jparse.py
- /tmp/claude-1000/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/scratchpad/tools/dstats.py

Probe sources:
- /home/user/git/.takemyjob-wt/lev-decode/prototypes/hopper-probe/probes/levdec/levdec.py (PATCHES `ETDG_FIN`)
- /home/user/git/.takemyjob-wt/lev-moe/prototypes/hopper-probe/probes/levmoe/sub_B4.py