# C1 engine design, options (a) and (b): moe phase A Build 2 (2026-11-01)

Designer for C1 (a)/(b). Read-only: nothing edited, no GPU time, no check. Sources: GAPMAP-1031 §2.1, §2.2, §4 Q8, App. C; research/gapmap-1031/finder-moe.md §4 C1; LEVERS-1031b (moe dead ends, research/levers-1031b.json); muse_moe_w4a16/hypotheses/CLAIMS.md (to :143); master muse_moe_w4a16/submission.py (line numbers below are master's); muse_decode_layer/submission.py MOE_CUDA_SRC (:3829-6587); the public H200_MEASURED_FACTS (copy in campaigns/2026-10-30-public-ladder/research/public-alygg77, commit 83a351a6, cited FACTS:n); public-ladder research/mo1-1030.md (the pure-consume bench).

## 0. Verdict

1. **(a) is worth its M0.** It does not lower the dequant floor. It removes the skinny path's per-unit structure: the 8-way K split, the per-unit CTA barrier, the partial hand-off to the helper warp, and the x reload at each job switch. It spends the 64 freed x registers on warps that run independently. Instructions per weight word fall only 4-6% (16.8 to 15.8-16.1). A 1.25x GO therefore also needs issue efficiency U to rise from about 0.55 in situ to at least 0.66. The same instruction class already reaches U = 0.69 when nothing interrupts the loop (mo1-1030 r13 bench). Central M0 estimate at N=1/8 is 1.15-1.25x, p(GO) about 0.3. N=16 passes almost by construction: one pass replaces MID's split re-stream.
2. **(b) loses on paper at d1 and v1. Do not build it.** The red.add traffic alone fits under 1.6 us (0.3-1.1 us). The step as a whole does not:
   - W2 K-slices of 80 columns per CTA (d1) or 270 (v1) lose KPERM scale sharing and pay a per-row-tile red.add epilogue. That adds 2.4-2.9 instructions per word on W2.
   - y is complete only after the cross-CTA sum, so ss needs a post-reduction pass. The public's measured finisher costs (FACTS:41, :49, :102) apply.
   - Net at d1 is -0.9 to -3.2 us against a gain of at most 1.6 us (§6).
3. **Corrections for the judge.**
   - 18.44 instructions per word is the public's WGS RS-wgmma loop after KPERM3 (FACTS:114), not TINY/TINY2/MID. The skinny inner loop is 15.28 per word by source count, about 16.8 with its structure.
   - "Decode minimum ~10-11" can't be reached under the ruling. It drops the per-weight block-scale multiply (4 HMUL2 per word). Moving that multiply to the FP32 partials costs at least 4 FFMA per word for an n8 tile, and KPERM rules it out altogether, since one k16 step mixes four scale blocks. Removing it would need FP8 activations or a lossy encoding. The compliant inner-loop floor is about 15.0 per word (§1).
   - Realized row gains are capped by the stream headroom (§4): d1 about 10% with today's chain, v1 about 7%, d16 about 10-15% (phase B and the split re-stream; FACTS:107 puts d16 phase A at the size-matched ceiling already).

## 1. Baseline: what the skinny consumer executes today

TINY, TINY2 and MID all run `skinny_ring_ws` (:1598) plus `ws_helper` (:1706).
- **Structure.** 8 compute warps and 1 helper warp: 288 threads, 149-152 registers, 215 KB of smem. 178 KB of that smem is the fat-path term in `okc_smem_bytes` (:2794), which these builds never use.
- **Unit.** One unit is 16 rows x full K: 32 KB of codes plus 4 KB of scales, one TMA copy, NS=3 slots.
- **Per warp.** Each warp holds x for its 512-K slice in 64 registers (`load_x_regs`, reloaded at every job switch). It does 32 k16 steps per unit, stores a float4 partial, and joins a CTA barrier. The helper sums the 8 partials and runs the epilogue (:1743, :1774).

**Per weight word** (one lane, one k16 step, 8 weights), by source:

| term | count |
|---|---|
| LDS.128 codes, 1 per 4 words | 0.25 |
| SCDK x2 per chunk (PRMT + IMAD each, sg and sg8 broadcast apart) | 1.00 |
| dq4 DQ7 (:468): 7 ALU + 2 IMAD.SHL | 9 |
| HMUL2.BF16 | 4 |
| HMMA.16816.F32.BF16 | 1 |
| scale LDS.128, 1 per unit | 0.03 |
| **inner subtotal** | **15.28** |
| per unit (32 words per lane): 12 FADD merge, wsre wait, STS.128 partial, BAR, refill/arrive by tid 0, full wait, slot/parity/job bookkeeping, about 35 instructions | +1.1 |
| helper warp: 8 LDS.128 + 32 FADD + 2 mbar + epilogue, about 120 per 256 warp-words | +0.45 |
| **total** | **about 16.8** |

The M0 census replaces these with SASS counts. Pipes per word:
- ALU: about 8.1 (7 + 0.5 PRMT + structure).
- FMA pipe: about 7.0 (IMAD 2.5, HMUL2 4, FADD).
- LSU: about 0.35.

**In-situ rate.**
- The public H200 d16 runs 1.12 us per 36 KB unit, issue 51.9%, 0.82 eligible warps per scheduler (FACTS:107).
- Our R67 d1 stamps give about 1.25 us per unit (H100, 1830 MHz: phase A 9.7 us over 7.76 units).
- At about 1.75 GHz, 1.12 us is about 1960 clk, or 33.5 weights/clk/SM. That gives U = 33.5 x 16.8 / 1024 = 0.55, which agrees with the ncu figure.

**Pure-consume anchor** (mo1-1030 r13, H100, 8 warps/SM, records resident in smem, B by ldmatrix):
- E4XL: 16.97 per word, 1573.9 clk per 32-step warp record. That is 41.6 weights/clk/SM, U = 0.69.
- KPERM-4 E4: 15.38 per word, 1428.2 clk. That is 45.9 weights/clk/SM, U = 0.69.
- The same instruction class runs 1.25-1.37x faster per SM when nothing interrupts the loop. C1 targets that gap.

**Floor.** An exact BF16 mma.sync consumer needs at least the following per word:

| term | per word |
|---|---|
| DQ7 decode (CLAIMS:117: the proven minimum) | 9 |
| HMUL2 (block scale on A) | 4 |
| HMMA | 1 |
| scale decode, paired (one PRMT+IMAD per row pair per 64-K chunk) | 0.5 |
| code LDS | 0.25 |
| **floor** | **about 14.75-15.0** |

The reachable instruction cut is therefore about 1-1.8 per word, 6-11%.

## 2. C1(a): row-owning warps over K-streamed stages

**Kept from master unchanged:** `route()`, the FAST1 mask path (:2423), the doneA hand-off, the grid barrier, the FINCOL final, and the ss double buffer.

**What changes:**

- **Work item.** The skinny unit stays as it is: job j, 16-row block, full K, 36 KB. CTA c takes units [v0, v1) by master's own `muldiv` split, so CTA-level balance is identical to master.
  - There are no cross-CTA fixups. On the H200 a stream-K split share costs 2.2-2.6 us and a finisher 3.6-4.4 us (FACTS:102).

- **Inside the CTA.** The range's stage work is cut into NW contiguous warp ranges in (item, stage) order, balanced to one stage.
  - Each warp owns rows and streams their whole K. Its accumulators hold the item's full-K sum, so there is no per-unit cross-warp reduction.
  - Two item shapes:
    - NT=1: item = 1 unit; stage = 512-K slice.
    - NT=2: item = an aligned unit pair (32 rows); stage = 256 K. Odd units at a range edge run as NT=1 items.

- **Range edges.** At d1 nearly every unit straddles two warps, since a CTA has 5-10 units for 8 warps. Each contributor:
  1. stores its C fragment (STS.128) into a CTA fixup slot;
  2. does `atom.add.acq_rel.cta` on the item's smem counter.

  The contributor that completes the count sums the slots in contributor order, which is deterministic, then runs the epilogue. That is at most 2 fixups per warp per phase, cta scope only, no GPU fence.

- **Weight feed.** Each warp has a private ring of D slots, refilled by its own lane 0:
  1. `fence.proxy.async.shared::cta`;
  2. `mbarrier.arrive.expect_tx`;
  3. `cp.async.bulk`.

  There is no producer warp, no empty barrier and no helper warp. FAST1 pre-issues each warp's first D stages before routing, from the touched-expert mask. That works for TINY and TINY2, and for C1-MID too, because C1-MID has one job per expert and no SPLIT.
  - NT=1 reads the existing `wAu` bytes. A (unit, 512-K slice) is 4 KB of contiguous codes `[w][c][lane][16 B]` plus 512 B of contiguous scales `[w][lane][16 B]`: 2 copies, no new weight copy, the same KPERM-4 order and decode.
  - NT=2 reads a new stage-major copy `wAc`/`wBc`, laid out `[unit][kstage 16][2 KB codes | 256 B scales]`. It is a lossless byte permutation of `wAu`: +0.9 GB of HBM, about +4 s of prepare.

- **x feed (B operand) from smem.** x is never read by per-warp LDG: at N=8 that would read more L2 bytes than the weights.
  - **Phase A.** All M token rows (32 / 64 / 128 KB for M ≤ 4 / 8 / 16) are TMA-copied at kernel start. They don't depend on routing, so the copy overlaps `route()`.
    - Row pitch 8208 B makes lanes (g, t) hit banks 4g+8t, a conflict-free LDS.128 in 4 wavefronts.
    - Job row r maps to token `S.ent[st+r]>>3` through a per-lane row pointer. Lanes with g ≥ N are predicated off, so B = 0.
  - **Phase B.** One lane polls `doneA[j]` as today, then TMA-copies the job's N inter rows (8 KB each) into a job buffer. Warps wait on `xfull[buf]`.
    - This turns master's exposed per-warp x reload at a job switch (16 LDG.128, about 0.4 us) into a prefetch.

- **Consume loop, per 64-K chunk per tile.**
  - One LDS.128 of codes.
  - One PRMT takes bytes 2c and 2c+1 of the lane's scale word into halves 0 and 1. One IMAD (`x k16 + 0x5C005C00`) turns them into the bf16x2 pair (s_g, s_g8) x 2^64.
  - `dq4` per word.
  - HMUL2 with `.H0_H0`/`.H1_H1` operand selectors: `bmul2l`/`bmul2h` (:427-434), already used by the WGS consumer (:2230-2231). r0 and r2 take s_g; r1 and r3 take s_g8.
  - HMMA into 2 accumulator chains (N ≤ 8), or 2 chains x 2 n-tiles (N 9-16).
  - B fragments: 2 LDS.128 per chunk per n-tile (`x[row][512s+64c+16t ..+16]`), shared by the warp's NT tiles.

- **Epilogue, warp-local, once per finished item.**
  - Phase A: merge the chains, then gt/up x gsA·2^62, SiLU, store bf16 inter. This is the same expression as `ws_helper` (:1743).
  - Phase B: x gsB·2^62 x (1+w), store f32 ybuf, Σy² by shuffle-xor 4/8/16, atomicAdd ss. This is the same as :1755-1774.
- **Guards.** All new code is under `#if OKC_C1`: device code, the Params fields `wAc`/`wBc`, Smem fields, the `okc_smem_bytes` branch and the `okc_run` w[14]/w[15] handling.
  - TINY2, MID and MAIN therefore preprocess to master's text. That is the B4 lesson: a new unguarded Params field changes every build.
  - Knobs: `OKC_C1` (1 = NT1 slices, 2 = NT2 stages), `OKC_C1NW` (8 or 12), `OKC_C1D`, `OKC_C1KS` (256 or 512).

## 3. Launch config, registers, smem (grid 132, one CTA/SM, `__launch_bounds__(NTL,1)`)

| build (rows) | warps (threads) | NT, n-tiles | stage | ring | x region | Smem struct | dyn smem | regs (est.) |
|---|---|---|---|---|---|---|---|---|
| C1-TINY, d1 (M ≤ 4) | 8 (256); knob 12 (384) | 1, 1 | 512-K slice, 4.5 KB, 2 copies | D=3: 108 KB (12 warps: D=2, 108 KB) | A 32 KB; B 2 x 32 KB: **64 KB** | about 17 KB (routing tables; `red[]` dropped) | **about 189 KB** | 96-112 (cap 255 or 168) |
| C1-TINY2, v1 (5-8) | 8 (256) | 1, 1 | 4.5 KB | D=2: 72 KB | A 64 KB; B 2 x 64 KB: **128 KB** | about 17 KB | **about 217 KB** | 96-112 |
| C1-MID, d16 (9-16) | 8 (256) | 2, 1-2 | 256-K pair stage, 2 x 2.25 KB | D=2: 72 KB | A 128 KB; B a pool of 16 rows: **128 KB** | about 17 KB | **about 217 KB** | 130-150 |

**Master for comparison:** 288 threads, 149-152 registers (64 of them x), 215 KB.

**Bytes in flight per SM:**
- Master: 108 KB.
- C1: 108 KB for TINY, 72 KB for TINY2 and MID.
- Need: at 33 GB/s per SM and about 1.5-2 us loaded latency, a ring needs 50-66 KB in flight. D=2 is at the margin, so M0 and M1 test D.

The phase-A x and the phase-B job buffers share one region. The first phase-B x copy waits for a CTA barrier at the A→B boundary, where master already waits on doneA.

## 4. Instructions per weight word (SASS-level arithmetic; per lane per k16 step, 8 weights)

| op | master TINY / TINY2 / MID | C1(a) NT=1 (TINY, TINY2) | C1(a) NT=2, N ≤ 8 (MID) | C1(a) NT=2, N 9-16 |
|---|---|---|---|---|
| LDS.128 codes | 0.25 | 0.25 | 0.25 | 0.25 |
| B fragments (x) | 0 (64 regs) | 0.50 | 0.25 | 0.50 |
| scale LDS | 0.03 | 0.03 | 0.03 | 0.03 |
| scale decode PRMT + IMAD | 1.00 | 0.50 (paired) | 0.50 | 0.50 |
| DQ7 decode, 7 ALU + 2 IMAD.SHL | 9 | 9 | 9 | 9 |
| HMUL2.BF16 | 4 | 4 | 4 | 4 |
| HMMA.16816.F32.BF16 | 1 | 1 | 1 | 2 |
| **inner subtotal** | **15.28** | **15.28** | **15.03** | **16.28** |
| stage or unit structure | 1.10 | 0.60 | 0.55 | 0.55 |
| helper, epilogue, fixups | 0.45 | 0.25 | 0.20 | 0.20 |
| **total** | **about 16.8** | **about 16.1** | **about 15.8** | **about 17.0 per 16 tokens** (master runs 2 jobs: about 33.6) |
| ALU pipe | about 8.1 | about 7.7 | about 7.6 | about 7.6 |
| LSU | about 0.35 | about 0.85 | about 0.60 | about 0.85 |

What the structure rows contain:
- Master's 1.10: FADD merge, STS partial, BAR, 2 mbar waits, refill, bookkeeping.
- C1's 0.55-0.60: one full-wait, 2 self-issued copies and bookkeeping per 32 words.

**Rate model:** R = 4 · U · 256 / I weights/clk/SM.

| case | I | U | R (w/clk/SM) | vs master |
|---|---|---|---|---|
| master in situ | 16.8 | 0.55 | 33.5 | — |
| C1-TINY at the anchor U | 16.1 | 0.69 | 43.9 | **1.31x** |
| C1-TINY | 16.1 | 0.62 | — | 1.18x |
| C1-TINY | 16.1 | 0.58 | — | 1.10x |

**GO (1.25x) needs U ≥ 0.66 with NT=1, or ≥ 0.645 with NT=2.**

**Pipe check at 43.9 w/clk/SM (0.171 warp-words/clk/SM):**
- ALU: 1.31 of 2.0 warp-instructions/clk, 66%.
- IMAD + HMUL2: under 25%.
- HMMA: one per 23 clk per partition.
- smem port (TMA writes + code reads + x reads): 39% at N=1, 73% at N=8 with NT=1. This is why MID is NT=2, and why TINY2 relies on N being about 1-3 per job on judge draws.

**Value cap per row.** A 1.25x consumer realizes min(consumer gain, stream headroom).
- d1: about 10% headroom with today's 6.3 us chain. The finder's floor at a 2.5 us chain is 21.9 against 25.0 us.
- v1: about 7% (63.3 against a 59.0 floor).
- d16: phase B, plus the split expert's second consumption (7.7-11% of consumer bytes, FACTS:107).

**Expected judge rows if M1 and M2 pass at gate level:**
- d1 -5 to -8%, v1 -3 to -5%, d16 -6 to -10%.
- **Headline +1.5% to +2.6%** (the finder's sensitivities).
- Decode port: about +0.5% to +0.9% by GAPMAP §1.1's method.

**EV:** p(M0 GO) about 0.3 x p(transfer) about 0.6 gives about 0.18, so **about +0.4% moe**.

## 5. Compliance: exact dequant, FP32 accumulation, BF16 activations

**Weights stay NVFP4.**
- C1 decodes master's own packed bytes: `wAu` for NT=1, and a byte permutation of it (`wAc`) for NT=2.
- The code layout is the same DQ7 / KPERM-4 order.
- Nothing is requantized, cached or precomputed from inputs.

**Dequant is exact, by the same ops as master:**
- The code becomes bf16 e2m1·2^-126 by masks.
- The scale becomes bf16 e4m3·2^64 by PRMT and IMAD. This is exact for bytes 0x08-0x7E. `prepare()` keeps master's `_scales_prmt_exact` check: if it fails, master's TINY/TINY2/MID are compiled instead of C1.
- The product has at most 6 significant bits and an exponent well inside the normal range, so HMUL2 is exact. `bmul2l`/`bmul2h` give bit-identical A fragments to master's `bmul2`.

**Accumulation is FP32:**
- The HMMA is bf16 x bf16 with FP32 accumulate.
- Epilogues are FP32, with the same expressions as master.
- inter is stored BF16, as master and SGLang do. ybuf and ss are FP32.
- The final pass is unchanged.

**Activations stay BF16:** x and inter are byte-copied into smem as BF16.

**Determinism.** The only difference from master is the FP32 order of the K sum: one warp's 2 chains plus at most 3 fixed-order fixup partials, against master's 8 warps x 4 chains plus the helper's sum. C1(a) adds no global atomics. Its output is deterministic apart from master's existing ss atomics.

**Rules.** Graph-capturable: routing-dependent control flow stays on the device and the build is chosen by M only. `load_inline` only.

## 6. C1(b): the K-fused A→B step. Design, count and verdict

**Design**
- **Phase A'.** Items are 16-inter-column groups (32 w13 rows), each run as (a)'s NT=2 item. h stays BF16 in smem.
- **W2 repack.** W2 is repacked in prepare, losslessly, as `[expert][row chunk of 256][16-col group][256 rows x 16 K]`. A CTA's (row chunk, group range) is then one copy.
- **Phase B'.** Each warp takes 2 row tiles x G_c k16 steps. G_c is about 5 groups per CTA at d1 and about 17 at v1. The y partials are pre-scaled by gsB·2^62·(1+w) and added into ybuf by FP32 red.add.
- **ss.** It needs a pass after the cross-CTA sum. Either one CTA per token re-reads its routes' rows twice, or a last arriver per (expert, chunk) finishes it.

**Count on W2, per word:**

| term | d1 | v1 |
|---|---|---|
| DQ7 + HMUL2 + HMMA (9 + 4 + 1) | 14 | 14 |
| scale decode: one block per row per group, no KPERM | 2.0 | 2.0 |
| code and scale LDS | 0.5 | 0.5 |
| red.add epilogue per row tile (2 FMUL + 2 RED + about 4 address), amortized over G_c words | 1.6 | 0.5 |
| stage | 0.4 | 0.4 |
| **total** | **about 18.5** | **about 17.4** |

That is +2.4 to +2.9 per word over (a) on a third of the bytes.

**Red.add budget at d1**
- **Traffic:** about 21 KB per CTA (4096 x N=1 x 4 B x about 1.3 experts per CTA). That is 2.8 MB in total, landing in 42 KB of ybuf, with about 66 contributors per address.
- **Same-address cost:** red.v4 serializes at about 17 ns (FACTS:41). That gives about 1.1 us per address if the adds arrive in a burst, about 0.3 us if staggered.
- **Throughput:** 8 MB of merges takes 0.96-1.5 us (FACTS:80), so 2.8 MB takes about 0.35-0.5 us.
- **So the traffic alone fits under 1.6 us (0.3-1.1 us).**

**Balance sheet at d1**

| term | us |
|---|---|
| gain: hand-off removed | ≤ 1.6 (finder); about 1.0 by NW x1.040 on 25 us |
| W2 consume (+16% of about 5-6 us) | -0.8 to -1.0 |
| red.add | -0.3 to -1.1 |
| ss pass: one CTA per token re-reading rows, or a finisher (fence 0.13 us + atomic 1.1-1.4 + read-back 1.5-2.2, FACTS:49, :102) | -0.6 to -2 |
| **net** | **-0.9 to -3.2** |

- **v1** is negative too: the gain is at most 1.3 us against costs of about 2.3-3.5 us.
- **d16** gains nothing (NW x0.999).

**Verdict: do not build (b).**
- Reopen only if a W2 layout keeps at least 64 contiguous K per row per CTA (KPERM back), and ss can be formed without a cross-CTA finisher.
- Proposed CLAIMS line, if the judge agrees: `[refuted] C1(b) K-fused A->B (paper)`, citing this file.

## 7. M0: warm-L2 consumer microbenchmark (pre-registered; at most 3 h)

**Where and how.**
- gpunode06, GPU 6 only, as the operator, through the hopper-probe harness in a `~/git/.takemyjob-wt/*` worktree.
- Node-local `/tmp/USER-hopper` only. Honor the hopper queue STOP file.
- No root, no Slurm, no node state changes. Move dangling `sub_*.py` symlinks aside for the build, then restore them with `git checkout --`.

**0. Compile census (about 20 min, no timing).**
- `ptxas -v` and `cuobjdump -sass` for both arms. Record registers, spills (must be 0), and inner-loop instructions per word from a static count of the unrolled chunk body.
- Pass: C1 inner ≤ 15.4 (NT=1) or ≤ 15.2 (NT=2), and master's inner count 15.28 ± 0.3.
- A model error above 0.5 per word stops the run. Fix the model before timing.

**1. Bench kernel.**
- Both arms come from one CUDA text: master `CUDA_SRC` plus the `#if OKC_C1` block.
- A host-built job list fills `S.jE/jS/jC/ent` in place of `route()`: one local expert, all M tokens routed to it, so N = M.
- Each CTA runs its real split range R = 8 times back to back: phase-A units, then phase-B units on synthetic inter. Steady state dominates; there is no grid barrier and no final pass.
- Weights are `prepare()`'s own `wAu` (and `wAc`) for that expert, 28.3 MB. They are L2-resident after 3 warm-up launches, and nothing is flushed.
- R50 mode: every weight copy is halved in both arms, the same as DX2's 18432-B sites. It is timing-only; outputs are not read. R50 keeps L2 below about 5 TB/s so the consumer is what gets measured.

**2. Arms.**

| arm | N |
|---|---|
| INC-TINY | 1 |
| INC-TINY2 | 8 |
| INC-MID (two 8-token jobs, as master splits) | 16 |
| C1-TINY 8 warps, D=3 | 1 |
| C1-TINY 12 warps, D=2 | 1 |
| C1-TINY2 8 warps, D=2 | 8 |
| C1-MID 8 warps, one job | 16 |

Arms are interleaved: 20 reps x 3 processes.

**3. Metric.**
- Weights/us/SM = 65536 x units x R / per-CTA consumer span. The span is globaltimer from the first stage wait to the last epilogue store, with clock64 alongside. Take the median over 132 CTAs.
- Cross-check against the graph-replay event time.
- Report the ncu warp-state mix (barrier, long scoreboard, wait, math throttle) and shared wavefronts for one rep per arm.

**4. Validity.**
- One full-bytes pass per arm. Check inter and y against an FP32 torch reference built from the same dequant.
- INC against C1: rel-L2 ≤ 1e-5, since only the FP32 order differs.
- Both arms must report equal unit and byte counters.

**5. Decision.** r = C1 / INC, weights/us.

| outcome | condition | action |
|---|---|---|
| GO | geomean(r(N=1), r(N=8)) ≥ 1.25 and r(N=16) ≥ 1.25 | M1 |
| KILL | geomean(N=1, N=8) < 1.15 | close C1 |
| BETWEEN | otherwise | one tuning round (12 warps, D, `OKC_C1KS` 256), then a re-measure |

Any GO first gets an adversarial verify: a fresh agent, swapped arm order, an independent R50 audit, and an equal-work proof.

## 8. M1 (TINY at d1), then M2

**Build.**
- C1-TINY replaces TINY for M ≤ 4 in `kernel_fn`.
- `prepare()` compiles C1-TINY instead of TINY when the PRMT check passes, and master's TINY when it fails. The compile count is unchanged.
- NT=1 needs no new weight copy, so prepare_s is expected about equal to master: 163.6-168.2 s on the judge, under 280 s.
- TINY2, MID and MAIN are untouched.

**H100 gates** (nodejudge-style; judge draws at bs 1, bs 2-4 as guards; interleaved pairs, 40 reps x 3 processes; 256 MiB flush; graph replay):

| | condition |
|---|---|
| GO | R50 d1 ≥ +8% and R67 d1 ≥ +5% |
| KILL | R67 d1 < +3% |

**Correctness.**
- 600 checked graph replays with fresh inputs and routings, 0 failures. Cases:
  - bs 1-4;
  - every local route on one expert;
  - routes only on experts 0 and 31;
  - no local routes (empty work);
  - range edges with 3 contributors per unit.
- The 5 TASK tests pass.
- Output against master stays within FP32 reordering.

**SASS.** The `cuobjdump -sass` hashes of TINY2, MID and MAIN are identical to master. 0 spills.

**M2.** C1-TINY2 at v1 and C1-MID at d16, under the same gates. d16 must include a draw with an expert of 9 or more tokens.

**After M1 and M2.**
1. Independent audit.
2. 2 moe checks. The auditor pre-registers the gate: headline mean ≥ 2.3867 x 1.012, and no row slower than its board row by more than 3%.
3. B4 can't ride along: it adds its own compile.

## 9. Size

**(a), about 680-830 lines.**

CUDA, about 600-750 lines:

| part | lines |
|---|---|
| consume loop, templated on NT and n-tiles | about 180 |
| per-warp ring, self-issue, FAST1 pre-issue | about 130 |
| warp-range split and fixups | about 110 |
| x staging, phases A and B | about 90 |
| epilogues | about 70 |
| guards and integration | about 60 |

Python, about 80 lines: defines and selection, the stage-major packer (NT=2), the prepare switch.

**M0 harness:** about 250 lines (bench kernel plus driver).

**(b), if built anyway:** about +500 lines.

## 10. The top three ways (a) fails

1. **Issue efficiency does not rise; the wait moves.**
   - Evidence: the tuned public builds all keep the per-unit CTA barrier (`-DOKC_WSB=1`, :1655), and the barrier-free compute path was pruned in v2. xlres (CLAIMS:120) showed a removed barrier's wait moving to the next wait.
   - With only a 4-6% instruction cut, U ≤ 0.62 reads 1.10-1.18x: KILL or BETWEEN.
   - Detect: M0's warp-state mix.
   - Lever: the 12-warp arm, 3 warps per scheduler. Master can't take it: NW16 is a dead end because of its reduction cost.
2. **The x feed.**
   - Row ownership puts B on LDS: 0.25-0.5 issue slots per word, and 73% of the smem port at N=8 with NT=1.
   - Phase-B x staging adds a TMA round trip after each doneA poll.
   - D=2 rings (72 KB/SM) may starve at the H200's loaded latency.
   - Detect: the M0 ratio at N=8 against N=1, shared wavefronts and bank conflicts, the slot-wait share of the span.
   - Levers: NT=2, D=3 where smem allows.
3. **Transfer is capped by bytes and codegen.**
   - The skinny rows sit near the stream ceiling: d16 phase A runs at 4.10-4.16 TB/s (FACTS:107), and d1 has about 10% headroom. A 1.25x consumer can realize only 3-8% per row.
   - R67 is a looser proxy on skinny rows: d1 +12% and v1 +7% against the judge (finder §2).
   - A new loop's ptxas schedule can cost what its count saves: FACTS:106, equal count, +2.2-3.0 us at d32.
   - Detect: the M1 R67 / R50 gap. This is why M1's KILL is R67 < +3%.

## 11. Decode port (§4 Q8, answered)

**The shared code.** Decode's `MOE_CUDA_SRC` (:3829-6587) carries moe's skinny path unchanged. A diff of the two CUDA texts gives 125 changed lines, none of them in the skinny consumer:
- B4FIN: defines, the `sspart` field, epilogue partials, the final.
- The `[WARFIX]` proxy fence in WGS.
- Decode keeps the fence that M1a removed.
- FINPF guards and the `unit`/`unit_next` fields.

`skinny_ring_ws`, `ws_helper`, `dq4` and SCD/SCDK are byte-identical.

**Defines.** `MOE_TINY2_DEFINES` and `MOE_MID_DEFINES` match moe's. Decode's `MOE_TINY_DEFINES` lacks `-DOKC_FINPF=1`.

**Port path.** C1 ports as the same `#if OKC_C1` block pasted into `MOE_CUDA_SRC`, plus `MOE_C1_*` defines picked in `_moe()` (:6620 onward). NT=1 needs no new weight copy. Check decode's prepare headroom before any compile is added.

The port becomes a decode phase-A item the day after a moe GO (GAPMAP §2.2).
