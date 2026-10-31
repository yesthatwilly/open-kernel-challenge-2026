# Foundation levers, 10-31 (workflow wf_d49ebbab-bc8: 5 effort-max finders on the H100 and a max judge)

Generated from the workflow's return value. The full JSON (row breakdowns, dead ends, run paths) is in research/levers-1031.json.

## Judge ranking

**Judge ranking of the levers on the public foundation (10-31)**

How I scored: expected value is the headline gain if the lever works, times p. That is after my discounts for H200 transfer, judge noise, build cost and compliance. All 8 levers are compliant: none changes precision and none is a gray area. S6 is compliant only with the conditions listed under it.

I read and confirmed every cited master line. Nothing in musespark was touched, and nothing on the H100 was run by me.

**1. muse_attention_decode: pp/ppstsm, softmax ping-pong between the two KV-head warpgroups in the flat q8 kernel**
- **Value:** about +0.9–1.0% on attention if it works, p 0.65, so about 0.62%. The same A0 kernel runs inside decode, so it ports there too: v16f −4.5 us and v32f −8.5 us, about +0.17% decode at p 0.6.
- **Why credible:**
  - Three pooled H100 runs in the L2-resident regime (where the kernel is bound by its compute warps, as on the judge): v16f 0.954 and v32f 0.953, with CIs within ±0.1%.
  - That regime predicted the judge's A0-versus-public compliance cost at about 0.9 in microseconds, so the H100-to-H200 transfer is calibrated.
  - The same edit on the public file is 2.2% slower. That is consistent with the public's NEG attn #2. It wins only on A0, because each turn now covers A0's exact-max, rescale and FP32-sum phase.
  - It changes scheduling only, and outputs are bit-identical.
- **First step:**
  - Port the two arms into two master candidates. The arms are /home/user/git/.takemyjob-wt/lev-attn/prototypes/hopper-probe/probes/levattn/sub_pp.py and sub_ppstsm.py. Edit only /home/user/git/musespark/muse_attention_decode/submission.py at :1812, :1855, :1953-1955 and :2021-2024.
  - Confirm with cuobjdump that the SASS of every non-flat instantiation is identical to A0. The r11k arm showed that a change in one instantiation can alter the code of others.
  - Audit the barrier 5/6 arrival counts, and check that nothing else uses IDs 5 or 6.
- **H100 gate:**
  - GO if the L2-resident v16f/v32f ratio to A0 is ≤0.965 for ppstsm (≤0.970 for pp), pooled over 2 runs.
  - The cold non-flat rows must stay within ±0.3%.
  - 12 rows × 2 seeds must be bit-identical, and the 16 TASK tests and the R11 force tests must pass.
  - KILL if the ratio is >0.975.
- **Judge checks:** 4 (2 ppstsm, 2 pp).
  - GO if, in both reads, the v16f kernel is ≤114.6 us and v32f ≤209.5 us. Today they read 117.2–118.0 and 213.6–214.5.
  - KILL if either read has v16f ≥116.5 or v32f ≥212.5, or if any non-flat row is 0.3 us slower on the mean.
  - On a tie, keep pp. The public's NEG attn #1 already saw an STSM stack lose.
- **Decode port:** its own rung, 2 checks. Decode's attention<8,true,0> must reach ≤112.5 us at v16f and ≤203.5 us at v32f (today 116.1 and 209.0).

**2. muse_moe_w4a16: S6a+S6b, a lossless 6-bit code for the E4M3 scale bytes, in all builds and in place; then ported to decode's MoE**
- **Value:**
  - About +1.2–1.5% on moe if it works, p 0.5, so about 0.7%.
  - Decode port: MoE is about 45% of decode, and 1% of MoE is worth 0.47% decode. That gives about +0.5% decode at p 0.45, so about 0.22%. This is the largest decode lever in the reports.
  - The value is already discounted for the large build.
- **Why credible:**
  - The byte-proxy run (every copy 1 KB short) was replicated: +1.8–2.4% on all five rows, with MID as a flat control.
  - The same proxy plus the real unpack work was equal or better. A synthetic test where the compute warps are the bottleneck puts the unpack's cost at ≤0.8%.
  - It differs from the public's NEG moe #6, which was an untimed estimate for an entropy code. This is a fixed-width code, and its offset folds into the existing PRMT constant.
  - H200 transfer: the skinny builds' compute warps have about 2x slack, so those rows should keep most of the gain. The WGS rows are partly limited by their compute warps on the H200, so they should keep about half.
- **Compliance: in, with conditions.** Rule 107 explicitly allows lossless compression, outputs are bit-identical, and it is the same class as the 12-bit BF16 codec already on master. Conditions:
  - Derive the offset and range from the data in prepare and pass them as a -D define; do not hard-code 0x50.
  - Fall back to the 8-bit layout when the local scale bytes span more than 64 codes. Test this with nvfp4.random_weights(wide_scales=True).
  - Add no new weight copy. Growing prepare memory cost 0.5–2.4 us on unchanged kernels (NEG decode #11). So build S6a and S6b together; TINY, TINY2 and MID share the unit-order copy, so MID should not get its own 8-bit copy.
  - Before the scored submit, tell the user: the gain depends on the generator's narrow scale range. A held-out rerun with wide scales would fall back to 8-bit, with no gain and no breakage.
- **First step:**
  - Packer and range check in prepare.
  - TMA copy sizes and strides: :690/692, :1974/1976, :2503/2505, and the helper0 string at :3544.
  - Unpack in skinny_ring_ws :1630 and the WGS stage :2207; the PRMT constant at :669 and :2229 becomes data-derived.
  - Unit-test the unpack on all 64 codes against the existing lookup table.
  - Then time the real build with /home/user/git/.takemyjob-wt/lev-moe/prototypes/hopper-probe/probes/levmoe/mkvar.py.
- **H100 gate:**
  - GO if the real build averages ≥+1.6% over the 6 rows with no row below +0.8%. Use judge pairing, cold L2 and at least 120 reps.
  - Outputs must be bit-identical, and the forced fallback must pass.
  - KILL if the mean is below +1.0% or any row is slower.
- **Judge checks:** 3, against a same-day control.
  - GO if d32 is ≥1.0% faster in at least 2 of 3 reads (d32 is the low-noise row, A/A 0.6%) and the 3-read headline mean is ≥+0.7%.
  - KILL if d32 and v32 are within ±0.5% in all 3 reads.
- **Decode port:** its own rung, 2 checks. It must also cover decode's own B4FIN WGS build at T=32.
  - GO if the summed MoE kernel time at d32s, v1s and d16s (base 180.9 us) drops by ≥2.0 us on the 2-read mean.
  - KILL if it drops by less than 0.5 us.

**3. muse_decode_layer: turn on L2 eviction hints in the TINY MoE build**
- **The change:** drop "-DOKC_L2H=0" at /home/user/git/musespark/muse_decode_layer/submission.py:6591.
- **Value:** +0.4–0.6% decode if it works, p 0.5, so about 0.25%. It is a one-token edit with no build.
- **Why credible:**
  - At d1s: x0.960, CI [0.954, 0.972], over 20 rounds and 8 input states.
  - A factor run shows the L2 hints account for TINY2's whole d1 advantage. nsys puts −4.0 us inside moe_fused, and the control rows are flat.
  - TINY2, MID and WGS already ship these hints and the judge accepted them. FINDINGS says neither side ever isolated TINY's L2H=0.
  - It targets different data than the public's refuted hints (NEG decode #4, page-table and KV rows).
  - Discount: d1f gained about 1% against d1s's 4% on the same MoE work, so the effect depends partly on what else is in L2. Expect 50–100% of the microseconds on the H200.
- **H100 gate:** already passed (12/12 rows, 13/13 TASK tests). Only confirm that the TINY SASS carries .L2::cache_hint.
- **Judge checks:** 2.
  - GO if moe_fused<false> at d1s is ≤21.6 us in both reads (base 22.9/23.3) and d1f's MoE is not above 29.2 us.
  - KILL if d1s is ≥22.6 us in either read.
- **Optional rider:** MOE_TINY2_MAX_M 8→4 changes only the v1 rows, so it can share the check (+0.06%). Keep it only if v1s and v1f are not slower.
- **Cross-problem:** A/B TINY's L2 hints at bs=1 in levmoe on the H100. Spend 3 moe checks only if d1 gains ≥2% with a CI that excludes 1.

**4. muse_prefill_layer: knob bundle, plus a pairwise-tree rider for the FP32 denominator**
- **The change:**
  - MOE_ORDER_SMALL 1→2 (:130), FINE_MIN_T 2048→512 (:136), BND_RC_MIN_T 8192→2048 (:146).
  - Rider: pairwise-tree sums in softmax_core_rr at :4315-4320.
- **Value:** +0.5% prefill (+0.1% from the rider), p 0.55, so about 0.3%.
- **Why credible:**
  - Four independent routing draws, with an A/A null of 0.1–0.2%: order 2 alone t1s −1.04%; order 2 plus fine tiles −1.59%; recompute alone t4s −0.48%.
  - Their NEG #3 does not cover it; they measured fine plans only at 2048 ≤ T < 4096.
  - I applied a 0.7x discount to the parts that trade HBM traffic.
  - Compliance: scheduling, tiling and recompute only, and the math stays FP32. The rider still sums unrounded P in FP32; only the order of the adds changes.
- **First step:** one lpf run with exactly these knobs plus the rider, on 2 draws and all 8 rows. The earlier all-row run also carried the T=16384 pooler pin, which is not part of this lever.
- **H100 gate:** GO if turns 1 and 4 together contribute ≥+0.45% and the first-turn rows stay within ±0.2%. KILL if below +0.35%.
- **Judge checks:** 3.
  - GO if, in at least 2 of 3 reads, the MoE kernel is ≤281 us at t1s (today 285.0) and ≤324.5 us at t1f (328.5), and the t4s boundary pair is ≤145 us (today 75+75).
  - First-turn rows must not be more than 0.5% worse on the mean.
  - KILL if t1s MoE is ≥284 us in 2 of 3.
  - Keep the rider only if t4f attention is ≤1455 us (today 1464.9) in at least 2 of 3.

**5. muse_bf16_proj: non-cooperative 64x64 nvjet plan for pooler m128, chosen by a timing race in prepare**
- **Riders, raced the same way:** klarge (18,4,2) for unpooler m128; olayer PIPE=1; qbig RED=4 or the 3-D tensor map.
- **Value:** +0.4–0.55% bf16, p 0.4, so about 0.2%. It is a knob, and the race bounds the downside.
- **Why credible:**
  - The proxy that isolates SM time reads 0.889–0.922 across 6 arms in 2 runs; the full cold problem reads 0.983–0.994.
  - That proxy reproduces the judge's 20.2 us, so this row is limited by SM time on the H200, not HBM.
  - Discount: the public chose the cooperative plan on the H200 for an unknown reason.
  - Compliance: same GEMM, same raw BF16 weights, FP32 accumulation. Confirm splitK=1 on the plan that wins.
- **First step:** add the arms to _iv_choose (:4400) for the pooler and unpooler m128 entries (:4012-4016). Time them under a read flush with enough reps to resolve a 1 us row.
- **H100 gate:** GO if the race picks the non-cooperative plan in at least 9 of 10 prepares and the cold v16 pooler is ≤0.99x base. KILL otherwise.
- **Judge checks:** 2–3.
  - GO if top_kernels shows the non-cooperative nvjet kernel for the v16 pooler at ≤19.2 us in at least 2 reads. Today it is 20.24 us; the B0 controls read 20.06–21.16.
  - If the judge's race picks the cooperative plan, the change is a free no-op.

**6. muse_moe_w4a16: HF, drop the helper warp's redundant end-of-kernel fence**
- **The change:** remove the __threadfence at :1778-1779 and its copy in the helper0 string at :3544.
- **Value:** +0.2% moe, p 0.5, so 0.1%. It adds a few hundredths of a percent to decode's skinny builds.
- **Why credible:** d1 x1.0077, CI [1.0043, 1.0111]; d16 x1.0025. It follows the cooperative-groups grid.sync release pattern, and it is outside the public's NEG #3, which covers the WGS fences.
- **First step:** audit it against the PTX memory model: helper stores, then bar 2, then CTA_SYNC, then thread 0's fence.gpu, then the barrier atomic. Then run a 10k-replay bit-exact stress test at d1, v1 and d16.
- **H100 gate:** the stress test must be clean and d1 must pool to ≥+0.5%. KILL on any mismatch.
- **Judge checks:** 2, as its own rung after S6, for non-regression only. The +0.2% is below moe's d1/d16 row noise of 3.9–6.3%.
- SHT32 (0.06%) is parked behind it.

**7. muse_decode_layer: the bf16 file's cgemv for the T=1 qkv**
- **Value:** +0.45% decode, p 0.2, so about 0.09%. Small build.
- **Why credible:**
  - The only evidence is on the judge: the bf16 board runs this shape in 7.0 us, against decode's sk2 at 10.1–10.9 us. The H100 shows parity and cannot tell them apart.
  - Extra discount: the +28 MB packed copy is exactly the NEG decode #11 risk (growing prepare memory).
- **First step:**
  - Port cg_compress (bf16 :3209-3240) and cgemv_kernel<2,512,8192,2> (:4355-4366), adding launch_dependents. Route only the T=1 qkv to it.
  - Put the packed copy into unused rows of an existing buffer if possible.
- **H100 gate:** in-layer d1s and d1f must not be more than 0.3 us slower. v1s, d16s, d32s and v32s must stay within ±0.3 us, as a check on the memory-layout effect.
- **Judge checks:** 2. GO if the qkv kernel is ≤9.0 us at d1s and d1f in both reads, and no other d1 or v1 kernel is 0.5 us slower.

**8. muse_attention_decode: compact R11 guard (park)**
- **Value:** about +0.3% attention, p 0.25, so about 0.08%. The +1.8% ceiling is real, but d1s swings +2–3% with code layout.
- **First step, only if a builder is idle:** study the SASS size of the run-once path. Never spend a judge check on A0nr, because it drops the LSE fallback.
- **H100 gate:** d1s must stay within ±0.3% across at least 3 cold runs, with ≥0.1 us gained on 4 of the other 5 non-flat rows.
- **Judge checks:** none now; 3 if the gate passes.

Ranks 1 and 2 are nearly tied, at about 0.7 risk-adjusted each. Attention ranks first on certainty and build size. None of these levers clears the /hypothesize bar on its own, so run them as a direct plan of rungs rather than minting hypotheses.

**Build now in phase A**
1. **moe S6 (rank 2).**
   - It is the only large build and the biggest decode lever, so it needs the most calendar time.
   - Design both layouts at once with the decode port in scope; decode embeds the same MOE_CUDA_SRC and _MA_ALT_SRCS.
   - Validate on moe first, where each row is one kernel, then port. Order: H100 gate, 3 moe checks, decode port, 2 decode checks. Aim to have the decode checks in by 11-02.
2. **attention pp/ppstsm (rank 1).**
   - The arms already exist, so it should land today.
   - Then port it into decode's attention as its own decode rung.
3. **decode L2 hints (rank 3).**
   - This needs no ultracode build. Dispatch the one-token edit now as decode's next 2 checks.
   - It goes ahead of the attention and S6 rungs in decode, one mechanism per rung.

**Leave for phase C (18–24Z)**
- **Cheap checks:**
  - Prefill bundle with the rider (3 checks).
  - bf16 race with its riders (2–3).
  - moe HF after the S6 rung (2).
  - moe TINY L2 hints, only after its H100 probe passes (3).
  - decode cgemv, only if a builder is idle (2).
- **Parked:** compact R11 and SHT32.
- **Not proposed:**
  - decode MID at T=8 on its own (a rider at most), and olayer/qbig on their own.
  - Any judge check of A0nr, and the finders' dead ends.

That is about 30 judge checks between 11-01 and 11-04, well inside 18–20 per problem per day. The number of agents, not checks, limits the plan.

## Finder levers (summary)

### muse_decode_layer (master = public-C D0, alygg77@6cae49a lineage; judge D0 check 4b2adae23a3346ff 2.2060, scored 201db0b8 2.1793)

- **L2 eviction hints on in the decode TINY MoE build (drop -DOKC_L2H=0; T=1 rows)**: 0.5% x p0.55, knob. Rows: d1s, d1f (only builds at T<=4 change; v1s and d16s controls unaffected)
- **bf16 file's cgemv (m=1 12-bit GEMV) for the decode T=1 qkv projection**: 0.45% x p0.25, small. Rows: d1s, d1f
- **Rider: MID MoE build at T=8 (MOE_TINY2_MAX_M 8 -> 4); below the bar on its own**: 0.06% x p0.35, knob. Rows: v1s, v1f

### muse_moe_w4a16

- **S6a: lossless 6-bit E4M3 block-scale code in the TINY/TINY2/WGS builds (36 KB -> 35 KB per unit and stage)**: 1.2% x p0.55, large. Rows: decode 1, verify 1, decode 32, verify 16, verify 32 (decode 16 via S6b)
- **S6b: port the 6-bit scale code into the MID build (ALT_SRCS helper0, decode 16)**: 0.3% x p0.5, medium. Rows: decode 16
- **HF: drop the helper warp's redundant end-of-kernel fence in the skinny (WS) builds**: 0.2% x p0.5, knob. Rows: decode 1, decode 16 (verify 1 unresolved)
- **SHT32: one-pass WGS final + producer warm-up of the final's code, M<64 only (decode 32)**: 0.13% x p0.45, medium. Rows: decode 32

### muse_bf16_proj

- **Re-pin the m128 cuBLASLt plans: non-cooperative 64x64 nvjet for pooler m128 (rider: klarge 128x64 plan for unpooler m128), raced in prepare**: 0.45% x p0.45, knob. Rows: v16 pooler m=128 (main); v16 unpooler m=128 (rider)
- **Warm-L2 config wins on the public m128 custom kernels as prepare-raced riders: olayer PIPE=1 (o m128), qbig RED=4 or 3-D tensor map (qkv m128)**: 0.15% x p0.35, knob. Rows: v16 o m=128; v16 qkv_gate m=128

### muse_attention_decode (master = public-C A0, scored 2.4337, BEST 0ef88b07). H100 probes on gpunode06 GPU 5, hopper-probe package levattn, branch/worktree hopper/lev-attn at /home/user/git/.takemyjob-wt/lev-attn. Generator: prototypes/hopper-probe/probes/levattn/mk_lev.py. Driver: levattn.py. Arms are sub_*.py. Runs 1-5 and 8 are under prototypes/hopper-probe/runs/20261031-0106*..0132*-gpunode06/out. Pooled stats: scratchpad lev/agg.py; headline calculator: lev/hl.py.

Method:
- Cold-L2 mode times CUDA-graph replays the way the judge does: 256 MiB flush, 2 input states, paired rotating rounds, bootstrap CI.
- L2-resident mode is a new regime. Every page table is remapped onto 64K cache rows (16 MB) and replayed with no flush. This makes the flat q8 kernel consumer-bound on the H100, as it already is on the judge's H200.
- Correctness used reference.check_implementation on 12 rows x 2 seeds, the 16 TASK tests, and an R11 forced-fallback test (_R11_FORCE=1, a second call, then normal routing).

- **L1 ppstsm: flat q8 softmax ping-pong between the two KV-head warpgroups + stmatrix V^T stores**: 1% x p0.65, small. Rows: v16f, v32f (attn_kernel<8,true,0>). No other kernel changes.
- **L2 (BELOW BAR as measured) compact R11 guard: the R11 code removed from the once-run path of the per-request kernels**: 0.3% x p0.3, medium. Rows: d1s, v1s, d16s, v16s, d32s, v32s (d1f, v1f marginal)

### muse_prefill_layer (master = public-C P2, submission.py e5558cfe, judge run 8b89517bd87544b6 at 1.7836). Probe package: /home/user/git/.takemyjob-wt/lev-prefill/prototypes/hopper-probe/probes/lpf/ (driver lpf.py, variant generator mkvar.py, run-lpf.sh). It is on branch hopper/lev-prefill and not committed. Runs are under /home/user/git/.takemyjob-wt/lev-prefill/prototypes/hopper-probe/runs/20261031-0104xx..0147xx-gpunode06: route, proft1, knob1, moet, attn, gemm, attnvar, moevar, ncu, bundle, seeds, merge. All runs were on gpunode06 GPU 6 only, as the operator, through hopper-probe, node-local, and every check_implementation passed. Inputs are judge-like: the judge's own case seeds, WEIGHTS_SEED 5353, and a router seed from reference.weights_fields with router_load targets backed out of each row's SOL (t1s 1358, t1f 1314, t4s 5543, t4f 5330, ft 28.9k-30.5k). Routing is very skewed: the top local expert takes about half the local routes (t1s 667 of 1356, fts 16108 of 30127), with 13-22 distinct local experts. Bottom line: no single compliant lever clears 0.5%. The only candidate near the bar is a knob bundle, about +0.65% on H100 and about +0.5% expected on H200.

- **MoE schedule + boundary knob bundle (MOE_ORDER_SMALL=2, FINE_MIN_T=512, BND_RC_MIN_T=2048)**: 0.5% x p0.55, knob. Rows: t1s, t1f (T=753), t4s, t4f (T=3012); first-turn rows unchanged
- **Pairwise-tree FP32 denominator in the RR softmax (variant 43, full attention)**: 0.1% x p0.7, small. Rows: t1f, t4f, ftf0, ftf32, ftf64 (full attention)
