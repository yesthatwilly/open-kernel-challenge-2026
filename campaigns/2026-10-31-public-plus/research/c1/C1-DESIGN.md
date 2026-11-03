# C1-DESIGN: moe Build 2 (A1), judge's decision (2026-11-01, ~02:10Z)

Judge for the C1 panel (effort max). Read-only: nothing in musespark or takemyjob was edited, no GPU was used, no check
was dispatched. Inputs read in full: `c1-design-engine.md` (options a, b) and `c1-design-diet.md` (option c) in this
scratchpad; GAPMAP-1031 §2.1 (moe), §2.2, §4 (Q8), App. C; research/gapmap-1031/finder-moe.md (§1, §2, §4 C1);
LEVERS-1031b.md and the moe `dead_ends` in research/levers-1031b.json; muse_moe_w4a16/hypotheses/CLAIMS.md (all 143
lines); master `muse_moe_w4a16/submission.py` (sha256 c737e460, blob b95f9537; the skinny path, run_phase, moe_fused,
the final, okc_smem_bytes, okc_run, prepare, kernel_fn); decode's MoE sources in master `muse_decode_layer/submission.py`
(read-only); the public H200_MEASURED_FACTS.md and NEGATIVE_RESULTS.md (okc-muse-spark-kernels @6cae49a); the
public-ladder mo1-1030.md (branch campaign/public-ladder). Master line numbers below are for blob b95f9537.

## 0. Decision

**Chosen for M0: design (a), "C1-engine": row-owning warps that stream a whole 16-row unit's K through private
per-warp TMA rings, with no per-unit CTA barrier, no cross-warp reduction and no helper warp.** It carries the judge
amendments J1-J8 (§3). The diet (c) is not a C1 candidate. Its bit-identical core, D1 (paired scale decode, which is
(a)'s own scale decode) plus D6 (helper-warp rotation for TINY/TINY2), is the first stage of the same M0 run, as the
control arm `INC-D`. That is the staged pair: stage 1 shakes down the harness and prices the cheap rider; stage 2 is
the engine. (b) is rejected on paper.

| criterion | (a) engine (chosen) | (b) K-fused A->B | (c) diet |
|---|---|---|---|
| Consumer gain at equal issue utilization, from SASS-calibrated counts (§1): N=1 / 8 / 16 | 1.25 / ~1.15-1.25 (x LDS fully active, 8 warps only) / ~2.1 per expert weight (one pass, 2 n-tiles, vs MID's 2 passes) | negative: W2 loses KPERM sharing and gains a red.add epilogue (+2.4-2.9 instr/word on W2) | 1.05-1.09 / 1.05-1.09 / 1.6-1.9 per expert (D9) |
| Predicted M0 read, best pre-registered arm | N=1 1.15-1.40 (the 12-warp arm carries the upside), N=8 1.10-1.30 | not built | 1.04-1.08 (the diet designer's own 1.03-1.07) |
| p(M0 GO) | 0.30 for the full GO (N=1, 8, 16); 0.40 for the TINY-only GO | ~0 | 0.03 |
| Judge rows if M1/M2 pass their gates (H200 stream-capped, §0.1) | d1 -2..-8%, v1 -1..-5%, d16 -3..-8% | net -0.9..-3.2 us at d1 | d1/v1 -1..-3% |
| Headline if everything lands | planning +1.8% (range +0.8..+2.6) | < 0 | +0.4% as a rider |
| EV, moe headline | ~+0.25% (~0.35 x 0.6 x 0.7 x 1.8%) | < 0 | ~+0.15% as a rider, no check of its own |
| Compliance risk | low: same DQ7, PRMT/IMAD scale, HMUL2, HMMA f32 as master; only the FP32 summation order changes; no new global atomics; graph-capturable. Audit items: J4 proxy order, J5 fixup order | medium: FP32 red.add order is nondeterministic, plus a cross-CTA ss finisher | none: bit-identical |
| Decode port | direct: the same `#if OKC_C1` block pasted into decode's `moe_helper0` string; a PDL caveat (§6) | weak | trivial: D1 only (decode already has D6) |
| Size, build time | ~500-700 CUDA lines + ~60 Python; probe package ~300; no new compile (C1-TINY replaces TINY) | +~500 lines | ~60 lines (+~200 for D9) |

Why (a) and not the diet:
- Today's skinny engine is limited by its **critical SMSP**, not its average count. SMSP 0 carries two compute warps,
  the refill warp's TMA issue and the whole WS helper warp: ~1,310 issue slots per unit, against ~1,104 on the other
  three (SASS census, §1).
- (a) removes the helper, the per-unit barrier, the 8-way partial hand-off and the x reload. It balances the four
  SMSPs at ~1,048 slots per unit-equivalent: **1.25x at equal issue utilization** in phase A, 1.27x in phase B.
- So M0's bar is "(a) must not lose issue efficiency against today's critical SMSP". It is not the engine designer's
  "U from 0.55 to 0.66", nor the diet designer's "83% critical-SMSP issue" (that bar is for a design at today's
  count).
- The 12-warp arm (3 warps per scheduler), which (a)'s 64 freed x registers allow, is the margin. The diet cannot
  reach it: the count floor (§1) caps a diet at ~1.08x.

Why not (b):
- The hand-off it removes is worth at most d1 x1.040 (1.6 us), v1 x1.020 and ~0 at d16 (finder NW ablation).
- Against that, per CTA K-slices of 80 columns (d1) or 270 (v1) lose KPERM-4 scale sharing.
- y is complete only after the cross-CTA sum, so the sum of squares needs a finisher. FACTS:102 prices it at 3.6-4.4 us.
- The red.add traffic alone does fit under 1.6 us (0.3-1.1 us), so GAPMAP's literal condition holds, but the step as
  a whole loses.
- It is the same family as EPF (CLAIMS:110), LLA and LAST1, all refuted.

### 0.1 Value cap, for the main thread
- On the H200, d1's stream phases already move ~76 MB in ~17-19 us: 4.0-4.5 TB/s, against DX3's ceilings of 3.93 TB/s
  (73 MB) to 4.68 TB/s (2 GiB). DX23's r(d1) = 1.026 says the consumer is the binding constraint, but only just: the
  rows are **co-bound**.
- A 1.25x consumer therefore realizes only the HBM headroom. That is 0-14% of d1's stream (central ~5%, -0.9 us,
  +0.6% headline), ~1-7% at v1, and ~5-10% at d16 (phase B is consumer-bound, FACTS:107, plus MID's second dequant).
- **M0 GO is necessary, not sufficient.** M1's R67 gate (the H200's HBM/SM balance on the H100) is the real filter.
  Quote planning value only after M1.
- TINY alone (d1 -2..-8%) is +0.3..+1.3% headline. That is below the GAPMAP judge gate of 2.3867 x 1.012, so do not
  spend moe checks before M2, unless M1 reads R67 d1 >= +8%.

## 1. Verification

**Verified against SASS.** Source: takemyjob run `20261031-114139-gpunode06`, `out/sk2fix-sass/`. These are cuobjdump
dumps of decode's three skinny MoE modules, which compile `moe_helper0`. Their `skinny_ring_ws`, `ws_helper`, `dq4`,
`ring_issue` and `load_x_regs` are byte-identical to moe's CUDA_SRC apart from the `TIDX` line (diffed today). I parsed
`glue_moe_3f7296ae9c4b52bd_sm90a`, function `moe_fused<false>`, with the scratchpad script `c1j/sassloop.py`. The
other two modules agree within 6 instructions (unit loops 700/752 and 704/755).

| item | SASS | the designers had |
|---|---|---|
| phase-A unit loop 0x5640-0x8250 | 706 instructions, of which 68 are the job-switch block (16 LDG.128, 32 CS2R, 20 more) and 86 the tid-0 refill block (two TMA variants) | — |
| **hot path, compute warp, per unit (32 words)** | **552 = 17.25/word**: HMMA 32, HMUL2.BF16 128, LOP3 166, SHF 77, IMAD.SHL 58, IMAD 22, PRMT 16, FADD 12, LDS.128 9, STS.128 1, BAR 1, TRYWAIT 2, control ~28 | ~531 (diet), 16.8/word (engine) |
| inner (DQ7 + scale + HMUL2 + HMMA + LDS) | 489 = 15.28/word, matching both source counts | 15.28 |
| refill, extra on the warp that owns tid 0 | ~51 per unit (phase-A path) | ≈25 (diet) |
| WS helper per unit (loop 0x106d0-0x11d20, 358 total) | ~96 common (8 LDS.128, 32 FADD, waits) + ~57 phase-A epilogue or ~80 phase-B = **~155 / ~175** | ≈100-120 (diet) |
| phase-B unit loop 0x8ed0-0xbe50 | 761; its job switch includes the doneA acquire (LDG.STRONG.GPU + CCTL.IVALL, NANOSLEEP loop) | — |
| registers | max index R149 (3f7296ae) and R142 (the other two), so 143-152 with reserved registers | finder nsys 149-152 |

Per-SMSP issue need per unit, assuming warp-to-SMSP = warp id mod 4 (unverified; M0.0's ncu per-SMSP counters check
it):
- moe TINY/TINY2 (helper = physical warp 8, on SMSP 0 with compute warps 0 and 4, and warp 0 refills): SMSP 0 ≈
  552 + 603 + 155 = **1,310** (phase A) / ~1,336 (B). SMSPs 1-3: 1,104-1,110.
- With D6 (helper0 rotation; moe MID and every decode skinny build already have it): SMSP 0 ≈ 1,259, SMSP 1 ≈ 1,155.
- With D1 too: SMSP 0 ≈ 1,227. **That is (c)'s ceiling, 1.07x**, plus D8's latency item.
- (a) at 8 warps: 2 x ~524 = **~1,048** per unit-equivalent (§3 budget), so 1.25x at equal utilization.

**Verified against source or documents:**
- **18.44** is the public WGS RS-wgmma loop after KPERM3 (FACTS:114, "20.12 to 18.44"), not TINY/TINY2/MID.
  GAPMAP §2.2's "decode minimum ~10-11" drops the 4 HMUL2 per word (0.25 + 9 + 0.5 + 1 = 10.75).
- The exact BF16 mma.sync floor is ~14.75/word in the body and ~15.3 with minimal control. DQ7 is 9
  (dq4 :468-478: 5 LOP3, 2 SHF, 2 IMAD.SHL; CLAIMS:117). Scale ≥ 0.5, HMUL2 4, HMMA 1, LDS 0.25.
- KPERM3 cut 8.35% of the count for +1.25% on the board (FACTS §7).
- The paired scale decode (`__byte_perm(..., 0x4040|sb|(sb+1)<<8) * k16 + 0x5C005C00` with `bmul2l`/`bmul2h`) is
  already master's WGS form (:2226-2231). It gives the same bf16 halves as two SCDKs (:669), so D1 is bit-identical.
- Smem: `okc_smem_bytes` (:2794-2802) = Smem (~36.6 KB) + max(3 x 36,864, 178,176) = 214,784 B. 67,584 B of the ring
  term is the dead fat path in the skinny builds.
- R-mode sites: `"r"(UNIT_BYTES)` occurs 8 times in CUDA_SRC and 8 times in ALT_SRCS["helper0"] (mklev2.py R67 asserts
  the same).
- The final: `FSEGP`/`FSEG = HD / (NT * 4)` in both sources. At NT = 384 that is 2, which covers only 3,072 of the
  4,096 columns (J7).
- `#define OKC_SKT 8` is unconditional (:329), so the diet's D9 `-DOKC_SKT=16` needs a source edit.
- Decode: TINY/TINY2/MID compile `@moe_helper0` (:12948 with FINPF, :12947, :12253). The engine's "decode TINY lacks
  FINPF" is outdated. Decode's MoE module name hashes the source (:6646-6650). Decode launches the MoE with PDL per T
  (:11261, :12170).
- Sensitivities 0.666 / 0.263 / 0.182 %/us (d1 / v1 / d16) equal 1/(6·t_row). The NW ablation (x1.040 / x1.020 /
  x0.999) and the dead ends come from finder §2 and levers-1031b.json.
- The decode port value: a 4% cut of decode's skinny MoE (moe_fused 23.0/22.1/78.7/50.7/59.8/72.1 us at
  d1s/d1f/v1s/v1f/d16s/d16f) gives +0.94% decode (GAPMAP §1.1); I re-derived it.

**Not verified (marked as estimates):**
- moe's own TINY build SASS: non-rotated helper, `-DOKC_L2H=0`. It is the same body; M0.0 re-censuses it.
- The helper's exact per-phase split (I counted paths by hand from the 358-instruction loop).
- The in-situ per-unit cycles. The H100 R67 stamps give 0.97-1.25 us per unit depending on the unknown expert count
  of the stamped draws; M0 measures it.
- The H200 SM clock in the skinny kernel (moe DX1 is unrun).
- The warp-to-SMSP mapping.
- DX23's r(d1) and ceilings: taken from the task text and LOG 00:45Z. The harvest file holds only the control table;
  I did not re-read the job log.
- Every (a) count in §3, which is a source estimate until M0.0.

## 2. Corrections to the panel and to GAPMAP §2.2

1. **Baseline.** Today's skinny consumer is 17.25 instr/word per compute warp. Its critical SMSP sits at ~20.5
   per word-equivalent. The 18.44 figure (WGS) and the "10-11 minimum" both leave the GAPMAP; the floor is ~15.
2. **Engine (a)'s structure cost was 0.55-0.6/word. That is optimistic.**
   - A self-issued bulk copy costs 12-20 SASS instructions. Master's refill block is ~51 for one copy, with its
     address lookups.
   - Two copies per 32-word stage (codes + scales from `wAu`) would add ~0.9-1.25/word.
   - **Use one copy per stage from a K-slice-major weight copy (J1).** Then (a) ≈ 16.3-16.4/word.
3. **Engine: the 12-warp knob breaks the final** (FINCOL/FINPF at NT = 384). J7 fixes it.
4. **Engine: smem.**
   - C1-TINY2 (8 warps, D=2, 2 x 8-row phase-B x buffers, fixups) is ≈ 233.9 KB, over the 232,448 B limit. J6 trims it.
   - 12 warps at M ≤ 8 does not fit with 512-K stages.
   - The C1-MID phase-B x pool at N 9-16 is unresolved: a 16-row job at a range edge needs 16 + N_next rows. This is
     an M2 design item.
5. **Engine: phase-B x by TMA after the doneA poll.** That is an async-proxy read of other CTAs' generic stores. It
   needs `fence.proxy.async.global` after the acquire, or generic loads (J4).
6. **Diet:** the refill (≈25 → ~51) and the helper (≈100-120 → ~155/175) were undercounted. Its 1.03-1.07x
   prediction stands, and if anything rises to 1.05-1.09x.
7. **Both:** the realized value is capped by the H200 stream headroom (§0.1). M0 GO does not by itself imply the
   GAPMAP's +2.0% planning number.

## 3. The chosen design: C1-engine with judge amendments

**Kept from master unchanged:**
- route() and the FAST1 mask arithmetic (:2423-2509);
- the per-CTA unit split (`muldiv_g` over `njS x RB_A/RB_B`, as the WS ranges at :2561-2569);
- doneA (credit with `__threadfence` + atomicAdd per job, exactly as ws_helper :1747-1756);
- the grid barrier, the FINCOL/FINPF final and the ss double buffer;
- the epilogue expressions (ws_helper :1736-1757 phase A, :1758-1776 phase B, verbatim);
- the dequant: the same packed bytes, `dq4`, the PRMT scale decode and HMMA m16n8k16 f32.

**The engine:**
- **Item and warp ranges.** CTA c owns units [v0, v1) per phase. Its stage list is (unit, K-slice k = 0..7), unit
  major. Warp w takes stages [⌊T·w/NW⌋, ⌊T·(w+1)/NW⌋), T = 8(v1 - v0).
  - A warp's accumulators carry over the slices of one unit.
  - At a unit end the warp runs the epilogue if it owned all 8 slices. Otherwise it is a contributor (J5).
- **Stage.** 512 K of one unit: 32 words per lane, 4,608 B (4,096 B codes `[c][lane][16 B]` + 512 B scales
  `[lane][16 B]`).
- **Weight feed.** Each warp has a private ring of D slots, one mbarrier each. Lane 0 refills slot s with stage s + D
  right after the warp's last read of slot s:
  1. `fence.proxy.async.shared::cta`;
  2. `mbarrier.arrive.expect_tx(OKC_C1COPY)`;
  3. one `cp.async.bulk` of OKC_C1COPY bytes (J1).
  There is no producer warp and no empty barrier.
- **Consume loop per stage** (the J3 budget is the census gate). Per chunk c (8 per stage):
  - 1 LDS.128 of codes;
  - 2 LDS.128 of x (B fragments, predicated by g < N);
  - 1 PRMT + 1 IMAD for the scale pair, every other chunk from the stage's scale uint4 (paired decode, :2228-2231);
  - 4 x (dq4 → `bmul2l`/`bmul2h` x4 → HMMA), alternating between ACC = 2 accumulator chains.
- **x (B operand) from smem, row pitch 8,208 B.** Lanes (g, t) hit banks 4g + 8t, so LDS.128 is conflict-free in 4
  wavefronts (checked).
  - Phase A: all M token rows by TMA right after `griddepcontrol.wait`. Job row r maps to token `S.ent[st + r] >> 3`
    through a per-lane row pointer.
  - Phase B: J4.
- **Epilogues are warp-local.**
  - Phase A: SwiGLU, then a BF16 store to inter.
  - Phase B: x gsB·2^62·(1+w) → ybuf (FP32); Σy² by shuffle-xor 4/8/16, then atomicAdd(ss). Both as master.
- **A→B.** After the CTA's last phase-A epilogue (a CTA barrier), one thread runs master's doneA credit. Phase B then
  starts.
- **No helper warp:** NTL = NT = NW x 32, `__launch_bounds__(NTL, 1)`, 1 CTA/SM, grid = SM count.

**Judge amendments (mandatory):**
- **J1. One copy per stage.** prepare() builds a K-slice-major copy `wAk`/`wBk`.
  - It is a lossless byte permutation of the unit-order copies w[10]/w[11]: per 36,864-B unit,
    `cat(codes.view(8, 4096), scales.view(8, 512), dim=1)` → `[blk][k][4608]`.
  - It costs +0.9 GB and ~1-2 s of prepare, as w[14]/w[15] after the WGS copies.
  - C1 reads stage (blk, k) at `wAk + (blk*8 + k) * 4608`, with blk computed as `ring_issue` does (:685-690).
  - The copy size is `OKC_C1COPY` (default 4608). The bench R-modes override it.
- **J2. Warps and depth are arms, not one choice.** 8 warps D=3 and 12 warps D=2 for TINY (M ≤ 4); 8 warps D=2 for
  TINY2/MID (smem). ACC = 2 by default, 4 in the tuning round.
- **J3. Census budget** per 32-word stage, hot path of one warp: inner 489 + full-wait ~5 + lane-0 refill ~14 + loop
  ~4 + amortized fold/epilogue/fixup ~8-12 = **~520-525 (16.3-16.4/word). Gate: ≤ 535.** Above that, fix before
  timing.
- **J4. Phase-B x staging.**
  - Use generic `ld.global.cg` after master's `ld_acquire` doneA poll. The first warp that needs a job stages its N
    inter rows into one of 2 job buffers, then `__syncwarp` and an mbarrier arrive; other warps wait on it.
  - If TMA is used instead, `fence.proxy.async.global` must follow the acquire. The audit checks this either way.
  - The phase-A region (all M rows) is reused for the 2 phase-B buffers after the A→B barrier.
- **J5. Range-edge fixups are deterministic.**
  - A contributor stores its C fragment (4 floats per lane per n-tile) to its warp's head or tail slot.
    `__syncwarp`, then lane 0 runs `__threadfence_block` and `atomicAdd` on the unit's smem counter.
  - The last arriver runs `__threadfence_block`, sums the slots in **K-slice order** (its own registers in its
    position), then runs the epilogue.
  - So no cross-CTA fixups and no global atomics are added.
- **J6. Smem fit.**
  - Under `#if OKC_C1`, size the Smem routing tables `eid`/`ent`/`pos` for 16 x 8 entries instead of EMAX = 3,072.
    That saves ~14.7 KB. Check route() and the final index them only up to M x 8.
  - Budgets: C1-TINY 8w D3 ≈ 205 KB, 12w D2 ≈ 209 KB; C1-TINY2 8w D2 ≈ 219 KB after the trim. The limit is 232,448 B.
- **J7. NW = 12 builds** run the FINPF prefetch and the FINCOL final on tid < 256, with FSEG/FSEGP computed for 256
  threads. route(), FAST1 (`tid >= NT - 32`) and the flag zeroing are NW-generic.
  - In C1 builds run_phase calls `c1_phase` for both BIG values, so the fat/skinny templates (NCH = 64/NW) are never
    instantiated.
- **J8. Guards.**
  - Every C1 line sits under `#if OKC_C1`, including the Params fields, the Smem fields, the okc_smem_bytes branch
    and okc_run's w[14]/w[15]. TINY2, MID and MAIN therefore preprocess to master's text (the B4 lesson).
  - Default `#ifndef OKC_C1 / #define OKC_C1 0`.

**Compliance:**
- NVFP4 stays NVFP4: the same bytes, only permuted.
- The dequant is exact: the PRMT path requires every scale byte in 0x08-0x7E, which prepare's `_scales_prmt_exact`
  check (:3333) guarantees. Otherwise master's TINY is compiled.
- HMUL2 products have ≤ 6 significant bits, from 2^-69 to 2^-51, so they are exact. Master relies on the same
  subnormal-input behaviour.
- FP32 accumulation and FP32 epilogues; BF16 x and inter.
- No input-derived data persists across calls. The build is chosen by M only. No host sync; graph-capturable.
  `load_inline` only.
- The only numeric difference from master is the FP32 order of the K sum.

## 4. M0: warm-L2 consumer microbenchmark (H100, GPU 6, ≤ 3 h)

**Where.**
- takemyjob worktree `~/git/.takemyjob-wt/c1m0`, branch `hopper/c1m0` off takemyjob master.
- Package `prototypes/hopper-probe/probes/c1m0/`, modeled on `~/git/.takemyjob-wt/gapmap-moe/prototypes/hopper-probe/probes/gapmoe/` (load_sub with the hashing load_inline
  wrapper, capture, `reference.generate_weights/generate_input/check_implementation`, the `run-gapmoe.sh` runner with
  the busy-GPU check and clock monitor) and on `~/git/.takemyjob-wt/lev-moe/prototypes/hopper-probe/probes/levmoe/` (`mklev2.py` exact-match patcher with asserted counts in
  CUDA_SRC **and** ALT_SRCS["helper0"]; `sub_R67TLAB.py` for the INC phase stamps; `levsass.py` for the SASS loop).
- The C1 code is written on musespark branch `campaign/c1-moe` (worktree `/home/user/git/musespark/.worktrees/c1-moe`,
  from master; never master or campaign/public-plus). Copy it into the probe as `sub_c1.py`, next to `sub_board.py`
  (= master moe bytes c737e460).
- Commands (match the allow rules):
  - `HP_BUNDLE=c1m0 bin/hopper-probe build`
  - `HP_BUNDLE=c1m0 bin/hopper-probe run gpunode06 --gpus 6 --tmax 85 c1m0-<job> ...`
  - end every session with `c1m0-clean`.
- Run as the operator. Node-local `/tmp/USER-hopper/c1m0-cache` only, removed by `c1m0-clean`. Env
  `/tmp/USER-hopper/venv-cu130`; ncu/nsys from `/opt/apps/nvhpc/26.9-cuda-13.3/...` (read-only).
- queue/STOP is present: it only stops `bin/hopper-queue` (LOG 02:05Z). Never start the queue runner or drop .job
  files. No root, no Slurm, no node state changes.
- Move dangling `sub_*.py` symlinks aside before `build`, then restore them with `git checkout --`.

**Arms** (one generated file per arm; every arm carries the bench patches below):

| arm | builds | what |
|---|---|---|
| INC | master TINY (N=1), TINY2 (N=8), MID (N=16): the exact prepare() define tuples + `-DOKC_PRMTSC=1 -DOKC_DQ7=1` | the incumbent |
| INC-AA | INC + `-DOKC_AA=1` (a no-op define, so a separate module) | A/A; must read within ±0.5%, else the run is void |
| INC-D | INC + D1 (paired scale in skinny_ring_ws, both sources) + D6 (TINY/TINY2 compile `@helper0` with `-DOKC_WSH0=1`; MID already does) | stage 1: the diet's bit-identical core |
| C1-8 | `-DOKC_C1=1 -DOKC_NW=8 -DOKC_C1D=3` (TINY) / `-DOKC_C1D=2` (TINY2) | stage 2 |
| C1-12 | `-DOKC_C1=1 -DOKC_NW=12 -DOKC_C1D=2` (TINY only) | stage 2 |
| C1-M | C1-8 with 2 n-tiles (B LDS 1.0/word, 2 HMMA/word), M ≤ 16, phase A only (J6 trim required) | the N=16 point |

**Bench patches** (applied by the patcher to INC and C1 alike; timing-only, never shipped):
- `#define OKC_DEBUG 0` → 1.
- Stamps, per CTA, by one thread:
  - `routed` (existing DBG(1));
  - `A_all`: every phase-A epilogue of the CTA done. INC: the helper at its `q == nA-1` credit point, as in TLAB.
    C1: after its A→B barrier.
  - `Bgo`: the first phase-B unit's x is ready in warp 0.
  - `B_all`: INC after the post-phase-B `WS_JOIN`; C1 after its phase-B barrier.
  Assert that the unit-loop SASS opcode sequence is identical with and without stamps; if not, use the kernel-time
  metric only.
- **Job replication** `-DOKC_M0R=R`, inserted right after `route(p, S);` in moe_fused, both sources: tid 0 copies
  `jE/jS/jC[f + j]` to `[f + r·n + j]` for r < R, sets `njS = n·R`, then CTA_SYNC.
  - JMAX = 88 ≥ 2·24.
  - FAST1 keeps its own (un-replicated) count, so its pre-issued slots hold wrong bytes but the count stays
    consistent: timing-only, no hang.
  - All replicas share entries, so inter rows coincide (benign).
- **R-modes.**
  - R50: INC 8+8 sites `"r"(UNIT_BYTES)` → `"r"(18432u)`; C1 `-DOKC_C1COPY=2304`.
  - R16B: 16 B everywhere, consumer-only diagnostic.
  - Verify statically: the UBLKCP size immediate is size/16 (master's 0x900 = 36,864 B); the expect_tx immediates too.
- C1 arms pre-issue nothing before routing. INC keeps FAST1. The difference metric below cancels that head start.

**Data and warm-L2 setup.**
- Weights: `reference.generate_weights(WEIGHTS_SEED)`, then one `prepare()` per process from the C1 module. Its
  state["w"] is a superset; INC kernels read w[0..13].
- Pinned routing: every token routes to local expert 0 plus non-local ids 32..38 (`topk_w` = 1/8); x is bf16 randn
  with a fixed seed. bs = N ∈ {1, 8, 16}, so one job per replica (MID: two 8-token jobs per replica, as master
  splits).
- Warm L2: no flush ever. 1 expert at R50 touches 14.2 MB per arm, so two interleaved arms stay ≤ 28.3 MB of the 50 MB
  L2. Run 3 untimed replays after every arm switch.

**Timing.**
- One CUDA graph per (arm, N, R) with fixed inputs. R ∈ {8, 24}, which gives per CTA ~31+16 and ~93+47 units at
  N ≤ 8.
- Blocks of 20 timed replays (CUDA events) after 3 warm-ups. Read the stamp buffer (scratch[3], G x 8 u64) after the
  block's last replay.
- Arms in a balanced Latin-square order per round; 6 rounds x 3 processes. Log `nvidia-smi` clocks every 5 s.

**Metric: steady-state consumer weights/us per SM, by difference.**
- Per CTA c: U_c = expert-units processed in phases A+B (INC-MID's pass-units count 1/2). Consumer span
  S_c = (A_all − routed) + (B_all − Bgo).
- **rate_c = 65,536 · (U_c(24) − U_c(8)) / (S_c(24) − S_c(8))**. The difference cancels route/FAST1 head start, the
  first-copy latency, the last epilogue and the A→B wait.
- Primary per (arm, N, round): the median over CTAs.
- r_N = median over the 18 (process, round) pairs of rate(C1 arm) / rate(INC), paired by round.
- Also report:
  - the kernel-time version (graph replay Δt / (ΔU_total / 132));
  - clock64 cycles per unit;
  - the R=8 absolute spans (the in-situ-like number);
  - phase A and phase B separately.

**Census (M0.0, no GPU timing, ~10 min).**
- `cuobjdump -sass` + `-res-usage` of every arm. Find unit/stage loops as backward branches enclosing 32 HMMA
  (`c1j/sassloop.py` logic).
- Hot path per unit/stage with cold blocks (job switch, lane-0 refill) separated. Registers; spills must be 0.
- Pre-registered values:
  - INC compute warp 552 ± 20 per unit;
  - refill +~51; helper ~155/175;
  - INC-D 536 ± 20;
  - C1 stage ≤ 535 (J3).
- A model error above 0.5/word stops the run until it is explained.

**ncu (one replay per arm and N, R=8, ~15 min).**
- `smsp__inst_executed` max/min/avg over SMSPs and `smsp__issue_active` (this tests the SMSP-0 premise).
- Warp stalls per issue: barrier, long/short scoreboard, wait, math-pipe throttle, mio throttle, dispatch.
- Shared wavefronts and bank conflicts; `sm__pipe_alu_cycles_active`.

**Correctness against the incumbent consumer** (R=1, full bytes, FAST1 as built):
- Inputs: the pinned routing at bs 1/8/16 plus 3 `generate_input` draws each at bs 1, 8 and 16.
- C1 vs INC:
  - inter: ≥ 99.9% bit-equal, the rest within 1 BF16 ulp (FP32-order rounding flips only);
  - ybuf: rel-L2 ≤ 1e-4;
  - out: rel-L2 ≤ 2e-3;
  - every arm passes `reference.check_implementation`.
- INC-D vs INC: inter and ybuf bit-identical.
- Equal work: a bench counter of units per CTA per phase. C1 = INC at N ≤ 8; C1 = INC/2 pass-units at N=16. Every
  job's doneA reaches RB_A.

**Sequence and pre-registered decision.**

| stage | content | budget | rule |
|---|---|---|---|
| M0.0 | harness, census, INC / INC-AA / INC-D timing at N=1/8/16 | ~60 min | A/A ±0.5%. **Rider sub-rule:** INC-D ≥ 1.03 at N=1 and N=8 and bit-identical → queue D1+D6 as a rider for the next moe carrier (D1 for decode). Not a check of its own. |
| M0.1 | C1 phase A (`-DOKC_C1M0A=1`: phase B skipped, timing-only) at N=1/8/16 | ~90 min | **Early KILL** if every C1 arm has r_A(N=1) < 1.10 and r_A(N=8) < 1.10 |
| M0.2 | C1 phase B, correctness, ncu | ~60 min | the final rule below |

- **GO:** r1 ≥ 1.25 and r8 ≥ 1.25 and r16 (per expert, phase A) ≥ 1.25 → M1.
- **GO-TINY:** r1 ≥ 1.25 and (r8 or r16) < 1.25 → M1. M2 only for the builds with r ≥ 1.25.
- **KILL:** r1 < 1.15 and r8 < 1.15 for every C1 arm → close C1 for moe and for decode's T ≤ 16 (CLAIMS refuted).
- **BETWEEN:** otherwise. One tuning round (≤ 45 min: warps 8/12, D 2/3, ACC 2/4), then a re-measure; the adversarial
  verify decides.
- The best arm per N is selected on processes 1-2 and confirmed on process 3.
- Context, not part of the rule: report C1 / INC-D at N=1. Below 1.15, flag that the cheap rider holds most of the
  gain.
- Any GO or BETWEEN gets the workflow's adversarial verify:
  - a fresh agent, swapped arm order;
  - an independent R50 audit from the SASS immediates;
  - an equal-work proof from the counters;
  - the metric recomputed from the raw stamps;
  - one process on a second weights seed.
- My prediction: INC per-unit ≈ 1,800-2,250 cycles. C1-8 r1 1.12-1.30, C1-12 r1 1.20-1.40, r8 1.10-1.30, r16 1.8-2.2.
  INC-D 1.04-1.08.

## 5. M1: C1-TINY replaces TINY for M ≤ 4 (d1), only after a verified GO

**CUDA, under `#if OKC_C1`.** Line numbers are for master b95f9537.

| where | change |
|---|---|
| knobs :209-340 | `OKC_C1` (default 0), `OKC_C1D`, `OKC_C1ACC`, `OKC_C1COPY` (4608). C1 builds have OKC_WS = 0, so NTL = NT (:319). Keep `NCH = 64/NW` (:328) out of C1 code. |
| Params :348-372 | `const unsigned char *wAk, *wBk;` |
| Smem :374-418 | drop `red[]` (`#if !OKC_WGS && !OKC_C1`); add the per-warp ring mbarriers `[NW][C1D]`, the x-buffer mbarriers and tags `[2]`, fixup counters `[2][32]`, fixup slots `[NW][2][32][4·ntiles]`; J6 table trim |
| new block after ws_helper (:1784, before skinny_split :1787) | `c1_issue`, `c1_consume<PB>` (stage loop), `c1_fixup`, `c1_epi<PB>` (ws_helper expressions), `c1_phase<PB>(p, S, v0, v1, ...)` |
| run_phase :2356-2391 | `#if OKC_C1`: v0/v1 by the WS split, then `c1_phase<PB>`, for both BIG values (J7) |
| moe_fused FAST1 :2423-2509 | C1 variant: mbarrier init for the per-warp rings. Each warp's first D stages are pre-issued from the touched-expert mask (same nS/v0/v1 math as :2484-2486, then the stage split). The phase-A x TMA goes here, after `griddepcontrol.wait`. |
| moe_fused :2413-2415, :2561-2570, :2607-2609 | WS dispatch, ranges and WS_JOIN are off (OKC_WS = 0) |
| final :2615-2629 (FINPF), :2711-2747 (FINCOL) | J7 for NW = 12 |
| okc_smem_bytes :2794-2802 | `#if OKC_C1`: base + NW·C1D·OKC_C1COPY + x region (2·M_max·8,208) |
| okc_run :2804-2817 | `#if OKC_C1`: TORCH_CHECK(w.size() > 15), then p.wAk = w[14], p.wBk = w[15] |

**Python.**
- `_stage_order()` next to `_unit_order` (:3200).
- prepare (:3288-3345): when `prmt`, append the stage copies after the WGS copies, and compile `C1_TINY_DEFINES` in
  place of `TINY_DEFINES` in the module list (:3335-3340). The compile count stays 4.
- kernel_fn (:3378-3385): `M <= TINY_MAX_M` picks C1-TINY when `sfx`, else TINY.
- `C1_TINY_DEFINES = ("-DOKC_NW=8|12", "-DOKC_TINY=1", "-DOKC_C1=1", "-DOKC_C1D=3|2", "-DOKC_C1ACC=2", "-DOKC_EPI4=0",
  "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_UWARP=0", "-DOKC_FINPF=1")`. The warps and D come from M0's winner.

**Launch.** grid = SM count, block = NW·32 (256 or 384), `__launch_bounds__(NTL, 1)`, 1 CTA/SM, dyn smem ≈ 205-209 KB,
0 spills.

**Tests.**
- The 5 TASK tests.
- 600 checked graph replays, fresh inputs each, against the reference: bs 1 ×300 (judge distribution), bs 2/3/4 ×100.
- Pinned edge cases:
  - all routes on one expert (bs 4);
  - experts 0 and 31 only;
  - no local routes (nU = 0 on every CTA);
  - 1 expert at bs 1 (3.9 phase-A units per CTA, so units with 3-5 contributors).
- C1 vs master TINY on the same inputs: inter ≥ 99.9% bit-equal, rest ≤ 1 ulp.
- Determinism: same inputs twice give bit-identical inter and ybuf.
- A 1,000-replay graph soak at bs 1 (hang check).
- compute-sanitizer racecheck and synccheck on 1 replay at bs 1 and bs 4.
- `cuobjdump -sass` of TINY2, MID and MAIN is identical to master's.

**Timing (GAPMAP §2.2, nodejudge style).**
- Judge-distribution draws at bs 1, with bs 2-4 as guards.
- Interleaved pairs, 40 replays x 3 processes x 4 pads (0-32 MiB before prepare).
- A 256 MiB **read** flush before each replay, then graph replay.
- R67 and R50 modes (INC's 8+8 sites and C1's OKC_C1COPY), plus full bytes for correctness.

**Gates.**
- GO: R50 d1 ≥ +8% and R67 d1 ≥ +5%; 0 failures; other builds' SASS unchanged; no bs 2-4 guard more than 2% slower
  in R67.
- KILL: R67 d1 < +3%.
- Prepare ≤ 280 s cold on the H100. Expect +1-2 s over master's ~151 s.

**M2.**
- C1-TINY2 (v1, 8 warps D=2, J6 trim) and C1-MID (d16, 2 n-tiles). MID's phase-B x pool needs a design first: a
  per-entry row pool ≤ 24 rows, or LDG for the second n-tile at a range edge.
- Same gates on their rows. MID is gated on 40 judge-distribution draws at bs 16, at least 10 of them with an expert
  of 9-16 tokens.

**Judge stage (GAPMAP).**
- Independent audit first:
  - output vs INC within FP32 order;
  - adversarial bs 1-17;
  - capture and replay;
  - J4/J5 memory-order review;
  - prepare.
- Then 2 moe checks against the auditor's pre-registered gate: headline mean ≥ 2.4153 (2.3867 x 1.012), no row more
  than 3% slower than its board row.
- §0.1: run them after M2, not after M1 alone.

## 6. Decode port notes (GAPMAP §4 Q8)

**Shared code (verified).**
- Decode's MOE_CUDA_SRC (:3829-6587) has the skinny path byte-identical to moe's.
- Decode's `_MA_ALT_SRCS["moe_helper0"]` (:12252) differs from moe's helper0 only around the always-rotated TIDX.
- Every decode skinny build compiles `@moe_helper0`: TINY at :12948 (+FINPF), TINY2 at :12947, MID at :12253.
- Selection is in `_moe()` (:6676-6720) and the T dispatch (:13105-13110).
- Module names hash the source (:6646-6650), so there is no stale-module hazard.

**Port.**
1. Paste the same `#if OKC_C1` block into the `moe_helper0` string. With no helper warp, TIDX = threadIdx.x, so the
   rotation is moot.
2. Swap the TINY defines for C1 defines (drop the WS knobs).
3. Add the K-slice copy to decode's MoE prepare: +0.9 GB, ~1-2 s. Decode's H200 prepare is 134.8 s.
4. okc_run's w[14]/w[15]: check decode's MoE weight-list layout first.

**PDL caveat.**
- Decode launches the MoE with PDL at some T (:11261, :12170), so x is the previous kernel's output.
- C1's phase-A x TMA must stay after `griddepcontrol.wait`, as master's FAST1 does.
- Pre-issuing weight stages before the wait is a separate lever (FACTS §2: two pre-wait stages help GEMVs). It is not
  part of the port.

**Diet rider for decode.** D1 only: decode already rotates the helper (D6).

**Value.** +0.94% decode per 4% cut of the skinny MoE (GAPMAP §1.1). At the realized 2-5%: +0.5-1.2% decode, EV ~+0.1%.
It becomes a decode phase-A item only the day after a moe judge GO (GAPMAP §2.2). It needs its own T=8/16 draws,
because decode's router differs from the moe generator.

## 7. Proposed CLAIMS lines (moe; the main thread appends)

- `- [refuted] C1(b): K-fused A->B step (W2 K-sliced per CTA, FP32 red.add into ybuf, ss finisher) | rows: load=decode_1, load=verify_1, load=decode_16 | evidence: paper (C1 judge, scratchpad C1-DESIGN.md + c1-design-engine.md §6): NW ceiling d1 x1.040 / v1 x1.020 / d16 x0.999 vs W2 +2.4-2.9 instr/word (no KPERM-4 at 80-270 K per CTA), ss finisher 3.6-4.4 us (FACTS:102); net -0.9..-3.2 us at d1 | board: 2.3867 | blob: b95f9537 | reopen: a W2 layout keeping >= 64 contiguous K per row per CTA AND ss without a cross-CTA finisher | 2026-11-01 campaign`
- `- [node-only] skinny consumer SASS census (TINY/TINY2/MID unit loop): 552 instr per 32-word unit per compute warp (17.25/word; inner 15.28), +~51 refill on the tid-0 warp, WS helper ~155/175 per unit; critical SMSP ~1,310 vs ~1,104 issue slots per unit; 18.44 is the public WGS loop, not this one | rows: load=decode_1, load=verify_1, load=decode_16 | evidence: takemyjob run 20261031-114139-gpunode06 out/sk2fix-sass (decode moe_helper0 modules, skinny body byte-identical to moe), C1-DESIGN.md §1; transfer: n/a (count) | board: 2.3867 | blob: b95f9537 | reopen: - | 2026-11-01 campaign`
