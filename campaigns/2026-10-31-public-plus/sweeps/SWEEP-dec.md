# SWEEP-dec: decode sweep after DEC-DX2 (design only, read-only, 2026-11-01 ~20:40Z)

Base for every variant: master DSTK `muse_decode_layer/submission.py` sha256 1c653e3f… (860,349 B; `SUB:` = its line
numbers). Every edit is an append-only overlay at the end of the file (master's `_x_prior = f; def f(...)` style);
CUDA text changes are anchored `str.replace` calls on the source strings inside that overlay, with count asserts.
Line 2 gets the provenance clause and a nonce. CG-O, CG-U, O8 and CPUSH-DEC stay closed (DX2 84b5da1a).

## 0. Answer

- **(a) Moe-master ports: nothing to port into TINY or TINY2.**
  - **B4FIN at d32:** decode already runs it. Decode was B4X's donor: `_moe` routes 16 < M ≤ 32 to `-DOKC_B4FIN=1`
    (SUB:6732-6736, SUB:13156).
  - **D6 and FINPF:** already in decode's TINY and TINY2 builds (SUB:12991-12992).
  - **D1:** neutral on the moe judge (MD16F: S 0.9994, d1 0.9929, v1 1.0027). LOG 05:42Z: "no decode D1 rider".
  - **NOFENCE:** affects the WGS builds only (d32/v16/v32; the fence is at SUB:5839). alygg77 measured it on the decode
    layer as a loss: d32f 1.35 us slower in 4 of 4 pads (public NEGATIVE_RESULTS, moe #3).
  - **C1 engine:** the only moe lever with decode value (+0.5 to 0.9%), but it is about 600 lines and its M0 read
    BETWEEN. It is a build, not a knob.
- **(b) and (c): what is left is two DRAM ramps, each sitting right after a tail where HBM is idle.**
  - **gle_pr (pr 16.85 us):** its stream cannot start until o's CTAs exit. gle_pr needs 209 KB of smem and 416 × 128
    registers, which do not fit beside o's two 81 KB CTAs (DS1 chain §1.2). On the H100, o ends 4.78 us after ATT7
    (TF7 profile: ATT7 ends at +17.33, o ends at +22.11). By my estimate the last ~2 us of that, after o's last weight
    stage lands, is compute and epilogue with HBM idle.
  - **Unpooler slice (un 3.62 us at d1s, 3.12 at v1s):** its stream cannot start until moe's CTAs exit (119 KB vs
    210 KB). SKIPF showed that this stream, not moe's barrier and final, is the d1 critical path. In DX2, un's chain
    cost equals its solo cost.
  - **Fix for both:** the predecessor prefetches the dependent's weight bytes into L2 during its own idle tail, as a
    fire-and-forget `cp.async.bulk.prefetch.L2` with evict_last. This gives **V1 PRF** and **V2 UPF**.
  - **The router byte cut is dropped.** Lossless storage of full-mantissa FP32 weights needs about 28 bits, so it saves
    at most 1.05 MB, about 0.3 us. It is in the byte class, and the H100 cannot screen that class.
- **v1s:** the only H100 lead is TF7's post-wait trigger: v1s span -2.6 us and o end -1.09 us in one process, with a
  ±0.7 us placement control. That is **V4**. HSV1 is judge-proven on attention (v1s +10.27%, about -0.66 us), but on
  its own it is below the 1.5 us bar. That is **V5**, built only if V4 holds.
- **Ranking by expected judge gain × p:**

| rank | variant | touched rows | central H200 gain if it works | p(screen) × p(judge \| screen) | EV (headline) |
|---|---|---|---|---|---|
| 1 | **V1 PRF** | d1s d1f | d1s and d1f -1.2 us each; headline +0.28% | 0.40 × 0.55 = 0.22 | +0.06% |
| 2 | **V2 UPF + UNP4** | d1s d1f v1s v1f | d1 -0.8 us, v1 -0.7 us; +0.30% | 0.30 × 0.50 = 0.15 | +0.045% |
| 3 | **V3 = V1 + V2** | d1s d1f v1s v1f | +0.5% | built only if V1 and V2 both survive | (sum) |
| 4 | **V4 TF7-E1V** | v1s | v1s -0.6 us; +0.07% | 0.30 × 0.35 = 0.10 | +0.007% |
| 5 | **V5 HSV1-DEC + E1** | v1s | v1s -1.1 us; +0.13% | 0.25 × 0.45 = 0.11 (only if V4 holds) | +0.014% |

- **Rank 4 vs 5:** V5's EV is slightly above V4's, but V5 is gated on V4's mechanism and costs a port lane, so V4 is
  screened first.
- **Odds:**
  - P(at least one survivor) ≈ 0.6.
  - P(the decode board moves today) ≈ 2-4%. A move needs about +1.5% over the 4-read master level of 2.1892.
  - If nothing survives the screen, spend no decode checks and do not re-roll master's bytes.

## 1. Inputs

**H200 map (DX2, master's timed path).** Chain cost / solo cost in us:
- d1s: qkv 7.36, attn 3.74, o 5.28, pr 16.85 (solo 17.50), route 1.18, moe 27.50, un 3.62 (solo 3.65), fin 0.99.
- v1s: qkv 8.88, attn 6.53 (solo 6.53), o 5.28, b1 1.31, pr 17.38, route 1.17, moe 66.27, un 3.12, fin 1.23.

**Gate metric (stated for every gate below):**
- rho_row = (geomean over this variant's checks of that row's `paired_ratio`, read from the okc status JSON
  `result.records[]` with `event == "benchmark"`, field `paired_ratio`) / R0_row.
- It is **NOT** the median of the per-rep `ratios` list.
- R0 = the geomean of paired_ratio over the 4 reads of master's DSTK timed bytes: 5a84a880, 437f3be3, 9185f31c, and
  84b5da1a (the DX2 timed path is master byte for byte).
- Index I = Σ w_r·ln(rho_r) / Σ w_r over the touched rows, with w = 46 for sliding rows and 16 for full rows.
- The okc headline field is geomean over the 6 loads of Σ(w·baseline)/Σ(w·median). It is used only as a no-harm
  guard. Master's 4-read mean is 2.1892, so the guard is 2.1564 (−1.5%).

| row | R0 | 0.99·R0 (touched floor) | R0/1.03 (untouched floor) | us | sd per read |
|---|---|---|---|---|---|
| d1s | 2.4917 | 2.4668 | 2.4191 | 68.5 | 0.37% |
| d1f | 2.4731 | 2.4484 | 2.4011 | 73.1 | 0.65% |
| d16s | 2.0362 | 2.0158 | 1.9769 | 129.7 | 0.73% |
| d16f | 2.1008 | 2.0798 | 2.0396 | 206.8 | 0.44% |
| d32s | 1.7362 | 1.7189 | 1.6856 | 151.4 | 2.12% |
| d32f | 2.0140 | 1.9938 | 1.9553 | 306.4 | 0.79% |
| v1s | 2.4486 | 2.4241 | 2.3773 | 107.1 | 1.90% |
| v1f | 2.3925 | 2.3686 | 2.3228 | 115.9 | 1.32% |
| v16s | 2.1275 | 2.1063 | 2.0656 | 246.1 | 2.28% |
| v16f | 2.0739 | 2.0532 | 2.0135 | 338.7 | 1.53% |
| v32s | 2.2899 | 2.2670 | 2.2232 | 311.4 | 1.75% |
| v32f | 2.2101 | 2.1880 | 2.1458 | 484.2 | 1.15% |

**Resolution.**
- d1s moves about 0.37% per read, so 1 us at d1s (+1.46%) shows clearly in 2 checks.
- v1s moves about 1.9% per read, so a 1 us change at v1s (+0.93%) is under 1σ in 2 checks. v1s-only variants
  therefore pre-register 4 reads (§4).

## 2. Variants

### V1 PRF: o's idle tail stages gle_pr's first K pooler stages in L2 (T = 1 only; rows d1s, d1f)

**Knobs**
- `PRF_K` ∈ {2, 3, 4} pooler stages, = K × 3,145,728 B. It is set at runtime, so one build serves every arm. K = 0 is
  the same module with the prefetch off; it is the placement control.
- The ship value is K* from the screen.

**Mechanism (critical path).**
- The pooler stream is stage-major (`bfp_sk2_stream`, SUB:9388): stage st of CTA c sits at
  `Ws + c·WBY + st·128·WBY`, with WBY = 24,576. So "the first K stages of all 128 CTAs" is one contiguous range of
  the stream, `pstream[0 : K·128·WBY]`.
- gle_o's producer warp exits right after its last refill. Instead, it waits for o's last full barrier, then prefetches
  its 1/(256·32) slice of that range, plus gle_pr's escape table. The escape table matters because sk2 loads it only
  after `griddepcontrol.wait`.
- gle_pr's pre-wait stages and its first post-wait stages then hit L2.
- The prefetch is a hint only. No data, ordering or result depends on it. gle_pr's FUSE instantiation never compiles
  the block (`if constexpr (!FUSE)`), so its SASS at T = 1 and T = 8 is unchanged.

**Edit 1: `_gle_source` wrap (3 anchored replaces plus 1 appended host function; every count == 1).**
1. Append to the GleEpi struct `"... int fpr, ilv; };"`:
   - the fields ` const unsigned char *pf, *pft; unsigned long long pfb; unsigned pftb;`;
   - the host statics `static const unsigned char *g_pfp=nullptr,*g_pftp=nullptr; static unsigned long long g_pfb=0; static unsigned g_pftb=0;`.
2. Producer tail. The anchor is unique in bfp_SK2_SRC:
   `"    __syncwarp();\n    if constexpr (CL > 1) asm volatile(\"barrier.cluster.wait.acquire.aligned;\" ::: \"memory\");\n    return;\n  }\n  if (warp >= NW + 1) {"`.
   Insert this right after `__syncwarp();`:
```cuda
    if constexpr (!FUSE) {   // [PRF] o (T = 1): stage gle_pr's first pooler stages + escape table in L2
      if (ep.pfb) {
        const int ls = nstt - 1;
        mbar_wait(full0 + 8 * (ls % NSTAGE), (ls / NSTAGE) & 1);   // o's last weight stage has landed: HBM goes idle
        uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
        const unsigned long long nl = 32ull * gridDim.x, per = ((ep.pfb + nl - 1) / nl + 15ull) & ~15ull;
        const unsigned long long off = ((unsigned long long)blockIdx.x * 32ull + lane) * per;
        if (off < ep.pfb) {
          const unsigned n = (unsigned)((ep.pfb - off) < per ? (ep.pfb - off) : per);
          asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(ep.pf + off), "r"(n), "l"(pol) : "memory");
        }
        if (blockIdx.x == 0 && lane == 0 && ep.pftb)
          asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(ep.pft), "r"(ep.pftb), "l"(pol) : "memory");
      }
    }
```
3. In gle_o, after `part.data_ptr<float>(), uvf.data_ptr<float>()};` (count 1), add
   `g_gle.pf=g_pfp; g_gle.pfb=g_pfb; g_gle.pft=g_pftp; g_gle.pftb=g_pftb;`.
4. Append the host setter:
   `void gle_set_pf(at::Tensor pf, int64_t pfb, at::Tensor pft, int64_t pftb)`. It TORCH_CHECKs that pfb and pftb
   are ≤ the tensors' byte sizes and are multiples of 16, then sets the statics.

**Edit 2: `_gle_ext` redefinition.** Copy master's SUB:13591-13609 verbatim, plus:
- the C++ declaration of `gle_set_pf`;
- `"gle_set_pf"` in the `functions` list.

**Edit 3: prepare wrap.** After the prior prepare:
- read `pci, pstream, ptab_, *_ = st["bfp"]["sk2"]["pooler"]`;
- nb = K × (n_rows / R) × WBY, computed from the `"pooler"` entry of `bfp_SK2_CONFIGS`; assert nb ≤ the stream's
  bytes;
- tb = the table's bytes, rounded down to a multiple of 16;
- call `_gle_ext().gle_set_pf(pstream, nb, ptab_, tb)`;
- print `[prf] on K=… bytes=… tab=…`, or `[prf] off: <reason>` if any step fails (that leaves master's behavior).

**Size.** About 60 lines and 5 KB. No new memory. Prepare +1 s.

**STEP 0, before any timing.** All of:
- gle_o: registers ≤ master's 84, 0 spill, and consumer-loop SASS identical apart from the producer block;
- gle_pr and every other module SASS-identical to master's;
- layer outputs bit-identical to master on 64 inputs, at K=0 and at every K (the prefetch is a hint).

**Predicted effect.**
- H100: d1s and d1f -1.0 to -2.3 us (o's tail ≈ 2.5 us idle; pr's ramp ≈ 1 us).
- H200, taking about 0.6 × because the stream is shorter: d1s and d1f -0.6 to -1.5 us. Central -1.2 us, which is
  rho_d1s ≈ 1.018 and rho_d1f ≈ 1.016.
- All other rows: 0 ± placement.

**Floors.**
- d1s ≥ 2.4668, d1f ≥ 2.4484.
- Every other row ≥ R0/1.03. v1s and v1f also run gle_pr from the rebuilt module; their SASS is identical.

**Risks.**
- The prefetch slows o's epilogue. The barrier wait keeps it after o's own stream.
- The H200 tail has less idle time than the H100's.
- Public prior: the earlier L2 prefetches that lost were issued *during* attention (contention) or *during* moe
  (eviction). PRF is issued in neither window.

### V2 UPF + UNP4: moe's phase-B finishers stage the unpooler slice in L2; the unpooler issues all 4 stages pre-wait (rows d1s, d1f, v1s, v1f)

**Knobs**
- `UPF_ON` (runtime): the w-list extras below are present or absent.
- `UN_PDL_N` ∈ {2, 4} (runtime; 2 is master's).
- Arms on one build: off/2 (placement control), on/2, off/4, on/4.

**Mechanism.**
- Each moe CTA, once its phases A and B are done (after `WS_JOIN()`, before `launch_dependents`), prefetches its
  1/132 share of the unpooler slice stream (`slice8_r192`: 4 stages × 128 CTAs × 12,288 B = 6.29 MB) and its escape
  table. It does this while the slowest CTAs finish phase B.
- At T ≤ 8 moe is consumer-bound (DX23 r(d1) 1.026), so HBM has headroom in the end spread.
- evict_last protects the prefetched lines from the end-spread stream. The public loss ("unpooler prefetch during MoE
  ... evicted") was a prefetch issued before the stream.
- UNP4: P = min(4, NSTAGE = 4, nst = 4) = 4 (`bfp__sk2_geom`), so the whole slice is requested as each unpooler CTA
  becomes resident.

**Edit 1: anchored replaces on `_MA_ALT_SRCS["moe_helper0"]`.** TINY, TINY2 and MID compile this source. Each anchor
count was verified as 1.
1. Params: before `"  unsigned long long* dbg;\n};"`, add
   `#if OKC_UNPF` / `const unsigned char *upf, *upft; unsigned long long upfb; unsigned upftb;` / `#endif`.
2. moe_fused: after `"  WS_JOIN();  // the helper's last epilogue stores are visible\n#endif\n"`, add:
```cuda
#if OKC_UNPF
  if (tid == 0 && p.upfb) {   // [UPF] this CTA's phases are done: stage its share of the dependent unpooler's stream in L2
    uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const unsigned long long per = ((p.upfb + gridDim.x - 1) / gridDim.x + 15ull) & ~15ull, off = (unsigned long long)blockIdx.x * per;
    if (off < p.upfb) {
      const unsigned n = (unsigned)((p.upfb - off) < per ? (p.upfb - off) : per);
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p.upf + off), "r"(n), "l"(pol) : "memory");
    }
    if (blockIdx.x == 0 && p.upftb)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p.upft), "r"(p.upftb), "l"(pol) : "memory");
  }
#endif
```
3. okc_run: after `"  p.out = (__nv_bfloat16*)out.data_ptr();"`, add under `#if OKC_UNPF`:
   - when `w.size() > 15`: set upf / upft from `w[14]` / `w[15]`, with byte counts rounded down to multiples of 16;
   - otherwise: null pointers and 0 bytes.

**Edit 2: Python.**
- Add `"-DOKC_UNPF=1"` to `MOE_TINY_DEFINES` and `MOE_TINY2_DEFINES`. MID, B4 and MAIN preprocess unchanged.
- Prepare wrap:
  - assert `len(st["ma_w"]) == 14`;
  - take the stream and table from `st["bfp"]["sk2"][bfp__sk2_name("unpooler", 1)]`, which is `slice8_r192`;
  - set `st["ma_w"] = list(st["ma_w"]) + [stream, tab]`. Other builds read only up to w[13];
  - print `[upf] on bytes=… tab=… unp=…`.
- A `_bfp_gemm` wrap for `name == "un" and m <= 8 and _pdl(m)`:
  - call `_bfp_pdl(m)`, then set `bfp_SK2_PDL = UN_PDL_N`;
  - call `bfp_kernel_fn(...)` as master's `_bfp_gemm` does;
  - in a `finally`, call `_bfp_pdl(m)`.

**Size.** About 50 lines and 4 KB. No new memory.

**STEP 0 (hard KILL).** Required:
- TINY and TINY2 `moe_fused<false>` phase-A/B unit loops SASS-identical to master's;
- registers ≤ 145/142, 0 spill;
- MID, B4 and MAIN SASS-identical.

Code compiled into moe_fused has moved d1s before: FOLD0 +4.5 us and FOLDNI0 +0.86 us without ever running.

**Predicted effect.**
- H100: d1 rows -0.6 to -1.5 us, v1 rows -0.5 to -1.3 us.
- H200: d1 rows -0.4 to -1.0 us, v1 rows -0.3 to -0.9 us. Central rho: d1s 1.012, d1f 1.011, v1s 1.007, v1f 1.006.
- The other 8 rows: 0 ± placement.

**Floors.**
- Touched: d1s ≥ 2.4668, d1f ≥ 2.4484, v1s ≥ 2.4241, v1f ≥ 2.3686.
- d16, d32, v16, v32 ≥ R0/1.03.

**Risks.**
- Moe codegen perturbation (STEP 0 catches it).
- Eviction of the staged lines.
- Once the unpooler is fast, moe's barrier and final become critical again. SKIPF (barrier and final removed) showed
  only -0.25 to +0.05 us, so they are short.

### V3 = V1 + V2: the fresh-submission form (rows d1s, d1f, v1s, v1f)

- Built only if V1 and V2 each survive. It is both overlays at K* and UN_PDL_N*.
- They touch different edges, so the gains should add. The stack screen checks for interference.
- Central H200: d1s -2.0 us, d1f -2.0 us, v1s -0.7 us, v1f -0.7 us. That is I ≈ +1.8% over the 4 rows and a headline
  of about +0.5%.
- Same floors as V1 ∪ V2.

### V4 TF7-E1V: ATT7 post-wait trigger at v1s only (row v1s)

**Edit.** One line in `campaigns/2026-10-31-public-plus/files/decode/tf7/gen_tf7.py` (worktree
h100-dec-tf7-att7-trigger-first, 42cbd6f6):
- change `tf = 1 if var == "E0L1" else 3` to `tf = 1 if var == "E0L1" else 2 if var == "E1V" else 3`;
- E1V is not in the `entry` tuple, so the trigger goes right after `pdl_wait()`;
- run `python3 gen_tf7.py <master 1c653e3f> TF7-E1V.py E1V`.

That gives `_TF7 = 2`: d1s keeps master's instance 13, and v1s swaps 15 to 47 under master's coop residency guard.

**Mechanism.** It is a launch-order (scheduling) change, and the transfer evidence is weak: the H100 v1s gain was
bandwidth overlap, and ATT7's own 9.10 → 7.87 us is unexplained.

**Predicted effect.**
- H100, 4 copies: v1s -0.6 to -2.0 us.
- H200: -0.3 to -1.2 us.

**Floors.** v1s ≥ 2.4241. Every other row ≥ R0/1.03; d1s must not move (its instance is unchanged).

**Build.** Minutes. Screen it first, while V1 and V2 build.

### V5 HSV1-DEC + E1: attention master's v1s head split in decode ATT7, with the TF7 bit on its instance (row v1s)

- **Port:** as cpush-dec-port did with CPUSH. Port the HSV1 edits (attention HSV1-M1 81a56018, landed as 399eb35d:
  one CTA per (split, KV head) of the v1s one-tile speculative coop grid; it requires !cluster) into
  `ATT7_CUDA_SRC`. Route only q_len 8, B = 1, sliding.
- **Trigger:** add MODE bit 32 (post-wait trigger) to the HSV1 instance.
- **Why it can work:** attention's v1s gain transferred on the judge (pooled +10.27%). In decode at v1s, ATT7's chain
  cost equals its solo cost (6.53 us), so a kernel saving reaches the row.
- **Predicted effect:** HSV1 alone -0.6 to -0.8 us (under the 1.5 us bar by design); with E1, -1.0 to -2.2 us on the
  H100 and -0.6 to -1.6 us on the H200.
- **Condition:** build only if V4's v1s D ≤ -0.7 us. About 3 h of port work.
- **Floors:** as V4.

## 3. H100 screen (all variants; gpunode06 GPU 7 only; hopper-probe rules; mandatory protocol)

**Arms**
- 4 byte-identical master copies M, I, K, P and 4 copies of the variant module, each under its own extension name.
- Load order is shuffled and rotated per process. Every process is fresh, with fresh extension caches.
- Runtime knob arms (V1 K, V2 UPF and UNP) run on those same copies. The off arm separates module placement from the
  mechanism.

**Metrics**
- **F (primary):** a 256 MiB `ld.global.cg` read sweep, then one graph replay; 30 rounds × 8.
- **G (corroborating):** the host enqueued ahead with a GPU spin, then 16 back-to-back replays; 40 rounds.
- **D** = per-round mean(variant copies) - mean(master copies). Report the median and a 95% bootstrap CI, plus the
  copy-level D.
- **Rows:** d1s, d1f, v1s, v1f, with d16s and v32s as controls.

**Validity**
- A process is valid if the F-mode A/A |mean(M,I) - mean(K,P)| ≤ 0.20 us at both d1s and v1s.
- Otherwise the process is void. At most 2 void repeats.
- Two valid processes are required.

**Survival (the orchestrator's bar, pre-registered).** In **both** valid processes:
- d1s F D ≤ -0.80 us with CI upper < 0 (V1, V2, V3), or v1s F D ≤ -1.50 us with CI upper < 0 (V2 through V5);
- G D of the same sign, ≤ -0.5 (d1s) or ≤ -1.0 (v1s);
- no untouched row worse than +0.30 us F in both processes;
- STEP 0 clean;
- tests clean: 13 TASK tests and 12 rows × 2 seeds through `reference.check_implementation`; dense and attention
  outputs bitwise equal to master's, moe within master's own run-to-run spread; 16,384 stall-injected fresh-state
  replays at each touched row with 0 gross; for V4 and V5, the TF7 positive control (wait removed) must fail.

**KILL (stop at the first valid process)**
- d1s F D > -0.30 us for V1, V2 or V3;
- v1s F D > -0.60 us for V4 or V5;
- any correctness, stress or SASS-identity failure.

**Ship pick.** V1 takes K* = argmin of the pooled d1s D. For V2, UN_PDL_N = 4 ships only if on/4 - on/2 ≤ -0.2 us
pooled.

## 4. Judge gate (2 checks per survivor; never a re-roll of judged bytes)

**Metric:** as §1. rho is computed from `paired_ratio` in okc status `result.records[event=benchmark]`, **not** from
the median of `ratios`.

**VALID** requires all of:
- 13/13 tests and 12/12 rows in both checks;
- the variant's on-line printed in both logs (`[prf] on K=…`, `[upf] on …`, the TF7 route line);
- every untouched row with rho in [1/1.03, 1.03].

A VOID read gets at most one re-send of the same bytes.

| variant | GO | KILL | otherwise |
|---|---|---|---|
| V1 | I(d1s,d1f) ≥ +0.8%, both touched rho ≥ 0.990, mean headline ≥ 2.1564 | I < +0.3%, any touched rho < 0.985, or a test failure | HOLD: 2 more reads, decide on 4 |
| V2 / V3 | I(4 rows) ≥ +0.6% (V3: ≥ +0.9%), every touched rho ≥ 0.990, headline ≥ 2.1564 | I < +0.2%, any touched rho < 0.985, or a test failure | HOLD as V1 |
| V4 / V5 | pre-register **4 reads**: rho_v1s ≥ 1.010 on the 4-read geomean, headline ≥ 2.1564 | rho_v1s < 1.000, or a failure | shelf |

**On GO**
- One scored run of r3 (a fresh nonce, bytes otherwise identical), attempt file first.
- Land on master if it is valid and the 3-read pooled I is ≥ +0.5% with untouched rows ≥ R0/1.03.
- The board moves only above 2.2220.

## 5. Order and time (now ~20:40Z; checks expire 00:00Z)

1. **Now:** generate V4 (minutes) and screen it first on GPU 7 (~25 min). Builders write V1 and V2 in parallel on CPU,
   each with STEP 0.
2. **~21:45Z:** V1 screen (~35 min; K runs at runtime). **~22:30Z:** V2 screen (~40 min).
3. **Survivor:** independent audit (~30 min), then attempt files with this gate verbatim, then 2 checks.
   - The latest realistic dispatch is V1 at ~23:15-23:30Z.
   - V2, V3 and V5 most likely land in 11-02's quota.
4. **If nothing survives:** no decode checks today. The quota goes to phase C on the other boards.

## 6. Considered and dropped (do not rebuild)

| lever | why it is dropped |
|---|---|
| B4X / B4FIN to decode | already in decode at d32 |
| moe D1 | judge-neutral on moe |
| moe D6 and FINPF | already in decode |
| NOFENCE | WGS-only; alygg77 measured a decode loss |
| Router 24-bit "lossless" or hi+lo byte cut | not lossless at 24 bits; ≤ 0.3 us; byte class (H100-blind) |
| cgpr1 / pooler cgemv redesign | refuted on the H100 (+5.2 us) |
| GLE_ILV, GLE_RC, SL_* configs, BFP_PDL_N | tail-only (≤ 0.2 us) and already tuned by alygg77 to identical values |
| GLE at T=8 (drop b1) | gle_pr is TORCH_CHECKed to M ≤ 8; needs code, not a knob; ±0.5 us |
| Route fold / FFOLD / SKIPR | refuted (+7.5 us fold cost; -0.25 us ceiling) |
| Moe TINY / TINY2 / MID build swaps at T = 1 / 8 | moe-board-tuned, no evidence |
| Moe pre-wait init hoist | F1EARLY already overlaps it; ≤ 0.1 us |
| Unpooler P=4 without UPF | probably ≤ 0.3 us because the stream is bandwidth-bound from residency; kept only as V2's knob |
| C1 engine | the real moe lever (+0.5 to 0.9% for decode) but about 600 lines; 11-02 or later |
| CG-O, CG-U, O8, CPUSH-DEC, TF7 at d1s, DS0, DS1, S1, CGPR1 | closed (DX2, lanes, CLAIMS) |
