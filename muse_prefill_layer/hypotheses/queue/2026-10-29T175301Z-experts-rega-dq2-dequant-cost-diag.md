---
problem: muse_prefill_layer
slug: experts-rega-dq2-dequant-cost-diag
created: 2026-10-29T175301Z
base: master
mint_blob: "c63801e7a1dcfdb7cbdd0d38bc9b295407be7489"
best_at_mint: 1.5008
area: "w13 + w2 expert GEMMs (rega dq_stage instruction cost: prepare-only diag of cheaper exact dequants, gated route)"
change: >
  Built on master (submission.py blob c63801e7, board 1.5008). Changes the INSTRUCTION COST of the
  rega dequant, not the stage schedule. A prepare-only diagnostic with an optional gated route;
  when nothing routes, kernel_fn, _experts, _experts_rega and every function the timed call
  reaches stay byte-for-byte master's. MASTER'S COST (source count; no CUDA toolchain on the
  minting node): dq_stage + dequant8 spend ~24 issue slots per k16 step per consumer thread, i.e.
  ~6 per bf16x2 (32 bf16x2 / ~190-210 slots per thread per stage vs 8 wgmma): per bf16x2
  ~3.75-4.75 integer ops (q <<= 4, shf >> 6, two lop3 for the 0x80008000 / 0x01C001C0 masks) + 1
  HFMA2.BF16 (fma with -0), plus per scale ~3 ops (__half22float2, fmul 2^119,
  __float2bfloat162_rn broadcast) on top of 2 cvt.rn.f16x2.e4m3x2 + 2 LDS per record (~7 of the 24
  per k16 step are scale work; 6be2a8d3's spec counted the same). The moe Engine A consumer does
  the same per-weight work in ~17 dequant instr per k16 step (moe CLAIMS:73 ncu census: ext_prop
  11, scale decode 4, HMUL2 4), via its dq2 form, which is already bit-identical to this dequant
  and landed in moe/decode (moe CLAIMS:74 L3). ARMS (all exact by construction: e2m1 x e4m3 x 2^-7
  has <= 6 significant bits and fits bf16, so every arm must produce the SAME bf16 A fragments;
  the gate proves it): (A1) scale-L3: master's code layout, scale path replaced by moe's
  scale_pair_rt / scale_pair_imad_rt (muse_moe_w4a16/submission.py ~:5050-5080: one 2-source prmt
  of row g's and row g+8's e4m3 words -> [b_g, 0, b_g8, 0], then t * 16 + 0x77807780 = bf16 bits
  of sf * 2^119 for every positive normal e4m3 byte in [0x08, 0x7E]) and the per-row broadcast
  folded into the bf16x2 multiply's half selectors (__byte_perm 0x1010 / 0x3232 on the paired
  word, or the bf16 mul with .H0_H0/.H1_H1 that ptxas folds); prepare checks once that every scale
  byte of the rega bank is in [0x08, 0x7E] (print the counts below 0x08 / zero / above) else A1
  and A3 are OFF. No bank change, so A1 is the cheap-to-route arm. (A2) ext_prop: moe's
  ext_prop<I> nibble extraction (:5030-5045: pair 0 = one lop3, pairs 1-2 = shift + lop3, pair 3 =
  the split-bit form) on a prepare-time lossless permutation of each 32-bit code word of a COPY of
  the rega bank (the same permutation moe's Engine A applies; codes stay NVFP4 bits, scales
  untouched), master's scale path. (A3) dq2 = A1 + A2 (moe Engine A's full dq2 + L3 form).
  Rejected and say why in one comment: a register prmt LUT (a bf16 e2m1 code needs bits 6-8, which
  straddle the byte boundary, and prmt's msb mode replicates a sign instead of OR-ing it, so >= 2
  prmt + selector building per pair: no cut) and an smem LUT (random-index bank conflicts on the
  MIO pipe). Hopper has no e2m1 cvt; use no sm_100 instruction. BUILD: one module
  muse_prefill_rega_dq derived from _rega_fsilu_src() (the fsilu_w2a source master dispatches) by
  asserted (old, new) edits each occurring exactly once (else '[rega-dq] OFF: edit'), with a
  template int DQ (0 = master's dq_stage text verbatim, 1/2/3 = A1/A2/A3) and the anatomy r2
  runtime MODE argument (reuse experts-rega-stage-anatomy-diag-r2.patch's mode plumbing): MODE 0 =
  full, MODE 2 = no MMA (dq_stage every stage folded into acc by one dependent add so it is kept),
  MODE 5 = pure dequant issue (no ring waits: dequant the stage-0 buffer KS x items times into a
  live sink; measures instruction cost without the producer floor t4). Print regs / local bytes /
  smem / CTAs per SM per instantiation; any spill turns that arm OFF. One nvcc build, master's
  REGA_CUDA_CFLAGS. PHASE _rega_dq_phase(state, weights, device, t_prepare0, fsilu_info), called
  after _rega_w2st_phase and before 'del fsilu_info', summary <= 3 KB printed at once, the align2
  / fa3c / fsilu / w2st echo block stays last. Per class low / high / ft (REGA_FSILU_TOKENS 753 /
  3012 / 16384, classes with rega_ok only), per GEMM (w13, w2), on master's routed inputs (as anat
  r2: xs / hs from master's own _experts_rega pipeline) under pop routing (_rega_pop_case at
  fsilu_info['alpha'][cls]) and random: (a) EXACT: hs and y of every arm vs master (torch.equal,
  max |diff| printed) on master's self-check routings of the class, error word 0, launch markers;
  (b) TIME through _rega_fsilu_bench, interleaved master / DQ0 / A1 / A2 / A3, REGA_FSILU_ROUNDS 2
  x REGA_FSILU_ALTERNATIONS 11, warm and cold L2, empty graph subtracted: master's launch, DQ0
  MODE 0 (fidelity, must be 0.97-1.03x master else INVALID), each arm MODE 0 / 2 / 5, and the w13
  + w2 pair MODE 0 per arm. ROUTE (only if it fits: no nvcc build after 205 s, no class step after
  255 s, the replay check started before REGA_COPY_DEADLINE_S): per class, the arm with the lowest
  pop pair ratio routes only when it is bit-exact on every self-check routing AND its pair MODE 0
  ratio vs master is <= 0.97 in every round, both L2 modes, both routings, and it passes master's
  _rega_replay_check copy plan; then state['rega_dq_ok'][cls] = DQ and _experts_rega launches the
  dq module's MODE 0 DQ instantiation for that class (dispatch on prepare flags only; A2 / A3 also
  keep the permuted bank copy in state under a REGA_STATE_KEYS-style key, cloned with the state,
  ~0.9 GB). Otherwise master runs unchanged. If routing does not fit the deadlines or the size
  budget (master 377,981 B; cap 1,000,000 B), ship print-only and print '[rega-dq] ROUTE:
  print-only'. RULES: kernel_fn stays graph-capturable, no host sync, no prints, no allocation
  sized by input values, no caching of anything derived from inputs (the permuted bank is a
  lossless retiling of the weights made in prepare, as master's bank); NVFP4 codes and E4M3 scales
  stay NVFP4/E4M3 bits in global memory; no child processes; eager canary + synchronize first,
  flushed prints, never raises (try / except / finally: synchronize, drop the module's tensors and
  graphs, empty_cache).
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "only if an arm routes at low: rega w13 + w2 are 48-53% of sliding rows and 28-43% of full rows; A3 ~-25-30% dequant instr (24 -> ~17-19 per k16 step), and if t0 tracks issue cost ~5-10% of rega -> ~2-4% on turn_1 rows; print-only or unrouted: flat (master re-measured, row noise +-1.5-2.5%)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "only if routed at high: g 0.21 there, so <= ~3% headline ceiling (CLAIMS:71); expect <= 3% on turn_4 rows"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "only if routed at ft: g 0.18-0.21; expect 0-3% on first_turn rows; the value of the run is the [rega-dq] per-arm MODE 0/2/5 ratios, not the headline"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-producer-scale-expand-all-t: that moved the unchanged scale conversion onto producer warps sharing the same SMSPs plus an extra mbarrier hop and smem ring (gated 1.03-1.10x slower); this run cuts the dequant's own instruction count on the consumers (moe Engine A's landed dq2 + L3 paired scale forms, A1 needs no new barrier or smem) and measures it per mode (pure-dequant MODE 5, full MODE 0). Also differs from experts-rega-stage-antiphase-ordered-issue and experts-rega-low-kpair-four-consumer-wgs (schedule/overlap changes, refuted): this keeps master's schedule and meets CLAIMS:73's reopen ('a cheaper dequant'); builds on experts-rega-stage-anatomy-diag-r2's modes."
target: diag
value_gain: 3.0
value_p: 0.25
value_ev: 0.75
value_cost: 300k tokens, 1 judge runs
evidence: >
  check cb22e6104be142ee [rega-anat] low w13 warm pop: full t0 293.6 us, MMA-only t1 194.4,
  dequant-only t2 191.1, producer-only t4 145.1; serial in every class, g 0.34 low / 0.21 high /
  0.18-0.21 ft; ceilings ~5.2% headline low / ~3% high (CLAIMS:70-71). Overlap closed: K-pair
  bbca0105 x1.09-1.14 slower; ordered anti-phase 17d3fdf1a6594d33 formed (retire skew 0.44-0.47)
  yet x1.07-1.16 slower, reopen 'a cheaper dequant' (CLAIMS:73); scale expansion moved to producer
  warps 6be2a8d3 gated 1.03-1.10x slower (LESSONS:182). Master rega dq ~24 instr per k16 step (~7
  scale; source count, 6be2a8d3 spec) vs moe Engine A dq2 ~17 dequant instr per step (moe
  CLAIMS:73, ncu P3c) and its L3 paired scale decode landed -2.0 ALU instr/step, eac bench
  x1.10-1.12, check f2100f49 every row +2.3-7.1% (moe CLAIMS:74).
decision: >
  Read '[rega-dq] VERDICT <cls>' (warm L2, pop routing, best exact arm). (1) best arm cuts
  pure-dequant MODE 5 by >= 25% (A3 predicted ~25-30%) AND the w13 + w2 pair MODE 0 by >= 8% at
  low: if it routed, the headline is the result -> append [lead]/[landed] by the board, and port
  the same arm to the moe/decode wgmma hot half (wg_kernel calls the identical dq_stage, moe
  submission.py:1128 / decode :15330; decode T=256 hot half 134.5 us vs ~39 us math floor) as the
  10-30 phase-A candidate, as its own hypothesis; if print-only, mint the 10-30 routed rega build
  first, port second. (2) MODE 5 cut >= 25% but pair MODE 0 cut < 3% in every class: the stage
  contention is in the MMA pipe / register file / smem, not dequant issue count -> close
  cheaper-dequant on the rega tile ([refuted]), name producer-side dequant (SS wgmma from smem
  bf16, the ssdq lineage) as the last prefill angle, and do NOT port to the wg hot half on this
  evidence (moe CLAIMS:62 already says that hot half's gap is fixed costs). (3) MODE 5 cut < 12%
  for every arm: master's rega dequant is already near its instruction floor (and the Engine A
  census does not transfer to the rega tile) -> close the rega dequant lever and the hot-half
  port. (4) anything between (MODE 5 cut 12-25%, or pair cut 3-8%): [shelf] with the per-class
  ratios, routed only where <= 0.97, offered as a stack row. INVALID (DQ0 fidelity outside
  0.97-1.03, spill, or any arm not torch.equal) -> no conclusion; fix and rerun.
instrument: >
  prepare-time CUDA-graph timings in the judge check's own prepare(): '[rega-dq] attrs' (regs /
  local / smem / CTAs per SM per DQ instantiation), '[rega-dq] SCALES' (bank scale-byte range
  counts), '[rega-dq] EXACT <cls>' (torch.equal per arm), '[rega-dq] TIME <cls> <gemm> <routing>
  <L2>' (master us, DQ0 fidelity, per-arm MODE 0 / 2 / 5 ratios per round), '[rega-dq] PAIR <cls>'
  (w13 + w2 MODE 0 ratios), '[rega-dq] ROUTE <cls>: A1|A2|A3|master|print-only', and one
  '[rega-dq] VERDICT <cls>' line per class, all within the last 16 KB of the log
user_requested: true
review: proceed
status: queued
review_at: 2026-10-29T175903Z
review_companions: null
review_best: 1.5007957663757285
review_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
---

# experts-rega-dq2-dequant-cost-diag

## Change

Built on master (submission.py blob c63801e7, board 1.5008). Changes the INSTRUCTION COST of the rega dequant, not the stage schedule. A prepare-only diagnostic with an optional gated route; when nothing routes, kernel_fn, _experts, _experts_rega and every function the timed call reaches stay byte-for-byte master's. MASTER'S COST (source count; no CUDA toolchain on the minting node): dq_stage + dequant8 spend ~24 issue slots per k16 step per consumer thread, i.e. ~6 per bf16x2 (32 bf16x2 / ~190-210 slots per thread per stage vs 8 wgmma): per bf16x2 ~3.75-4.75 integer ops (q <<= 4, shf >> 6, two lop3 for the 0x80008000 / 0x01C001C0 masks) + 1 HFMA2.BF16 (fma with -0), plus per scale ~3 ops (__half22float2, fmul 2^119, __float2bfloat162_rn broadcast) on top of 2 cvt.rn.f16x2.e4m3x2 + 2 LDS per record (~7 of the 24 per k16 step are scale work; 6be2a8d3's spec counted the same). The moe Engine A consumer does the same per-weight work in ~17 dequant instr per k16 step (moe CLAIMS:73 ncu census: ext_prop 11, scale decode 4, HMUL2 4), via its dq2 form, which is already bit-identical to this dequant and landed in moe/decode (moe CLAIMS:74 L3). ARMS (all exact by construction: e2m1 x e4m3 x 2^-7 has <= 6 significant bits and fits bf16, so every arm must produce the SAME bf16 A fragments; the gate proves it): (A1) scale-L3: master's code layout, scale path replaced by moe's scale_pair_rt / scale_pair_imad_rt (muse_moe_w4a16/submission.py ~:5050-5080: one 2-source prmt of row g's and row g+8's e4m3 words -> [b_g, 0, b_g8, 0], then t * 16 + 0x77807780 = bf16 bits of sf * 2^119 for every positive normal e4m3 byte in [0x08, 0x7E]) and the per-row broadcast folded into the bf16x2 multiply's half selectors (__byte_perm 0x1010 / 0x3232 on the paired word, or the bf16 mul with .H0_H0/.H1_H1 that ptxas folds); prepare checks once that every scale byte of the rega bank is in [0x08, 0x7E] (print the counts below 0x08 / zero / above) else A1 and A3 are OFF. No bank change, so A1 is the cheap-to-route arm. (A2) ext_prop: moe's ext_prop<I> nibble extraction (:5030-5045: pair 0 = one lop3, pairs 1-2 = shift + lop3, pair 3 = the split-bit form) on a prepare-time lossless permutation of each 32-bit code word of a COPY of the rega bank (the same permutation moe's Engine A applies; codes stay NVFP4 bits, scales untouched), master's scale path. (A3) dq2 = A1 + A2 (moe Engine A's full dq2 + L3 form). Rejected and say why in one comment: a register prmt LUT (a bf16 e2m1 code needs bits 6-8, which straddle the byte boundary, and prmt's msb mode replicates a sign instead of OR-ing it, so >= 2 prmt + selector building per pair: no cut) and an smem LUT (random-index bank conflicts on the MIO pipe). Hopper has no e2m1 cvt; use no sm_100 instruction. BUILD: one module muse_prefill_rega_dq derived from _rega_fsilu_src() (the fsilu_w2a source master dispatches) by asserted (old, new) edits each occurring exactly once (else '[rega-dq] OFF: edit'), with a template int DQ (0 = master's dq_stage text verbatim, 1/2/3 = A1/A2/A3) and the anatomy r2 runtime MODE argument (reuse experts-rega-stage-anatomy-diag-r2.patch's mode plumbing): MODE 0 = full, MODE 2 = no MMA (dq_stage every stage folded into acc by one dependent add so it is kept), MODE 5 = pure dequant issue (no ring waits: dequant the stage-0 buffer KS x items times into a live sink; measures instruction cost without the producer floor t4). Print regs / local bytes / smem / CTAs per SM per instantiation; any spill turns that arm OFF. One nvcc build, master's REGA_CUDA_CFLAGS. PHASE _rega_dq_phase(state, weights, device, t_prepare0, fsilu_info), called after _rega_w2st_phase and before 'del fsilu_info', summary <= 3 KB printed at once, the align2 / fa3c / fsilu / w2st echo block stays last. Per class low / high / ft (REGA_FSILU_TOKENS 753 / 3012 / 16384, classes with rega_ok only), per GEMM (w13, w2), on master's routed inputs (as anat r2: xs / hs from master's own _experts_rega pipeline) under pop routing (_rega_pop_case at fsilu_info['alpha'][cls]) and random: (a) EXACT: hs and y of every arm vs master (torch.equal, max |diff| printed) on master's self-check routings of the class, error word 0, launch markers; (b) TIME through _rega_fsilu_bench, interleaved master / DQ0 / A1 / A2 / A3, REGA_FSILU_ROUNDS 2 x REGA_FSILU_ALTERNATIONS 11, warm and cold L2, empty graph subtracted: master's launch, DQ0 MODE 0 (fidelity, must be 0.97-1.03x master else INVALID), each arm MODE 0 / 2 / 5, and the w13 + w2 pair MODE 0 per arm. ROUTE (only if it fits: no nvcc build after 205 s, no class step after 255 s, the replay check started before REGA_COPY_DEADLINE_S): per class, the arm with the lowest pop pair ratio routes only when it is bit-exact on every self-check routing AND its pair MODE 0 ratio vs master is <= 0.97 in every round, both L2 modes, both routings, and it passes master's _rega_replay_check copy plan; then state['rega_dq_ok'][cls] = DQ and _experts_rega launches the dq module's MODE 0 DQ instantiation for that class (dispatch on prepare flags only; A2 / A3 also keep the permuted bank copy in state under a REGA_STATE_KEYS-style key, cloned with the state, ~0.9 GB). Otherwise master runs unchanged. If routing does not fit the deadlines or the size budget (master 377,981 B; cap 1,000,000 B), ship print-only and print '[rega-dq] ROUTE: print-only'. RULES: kernel_fn stays graph-capturable, no host sync, no prints, no allocation sized by input values, no caching of anything derived from inputs (the permuted bank is a lossless retiling of the weights made in prepare, as master's bank); NVFP4 codes and E4M3 scales stay NVFP4/E4M3 bits in global memory; no child processes; eager canary + synchronize first, flushed prints, never raises (try / except / finally: synchronize, drop the module's tensors and graphs, empty_cache).

## Expected

- `*load=turn_1*` faster: only if an arm routes at low: rega w13 + w2 are 48-53% of sliding rows and 28-43% of full rows; A3 ~-25-30% dequant instr (24 -> ~17-19 per k16 step), and if t0 tracks issue cost ~5-10% of rega -> ~2-4% on turn_1 rows; print-only or unrouted: flat (master re-measured, row noise +-1.5-2.5%)
- `*load=turn_4*` faster: only if routed at high: g 0.21 there, so <= ~3% headline ceiling (CLAIMS:71); expect <= 3% on turn_4 rows
- `*load=first_turn*` faster: only if routed at ft: g 0.18-0.21; expect 0-3% on first_turn rows; the value of the run is the [rega-dq] per-arm MODE 0/2/5 ratios, not the headline

## Value case

- **Target:** diag: a prepare-only judge diagnostic that sizes a lever.
- **Evidence:** check cb22e6104be142ee [rega-anat] low w13 warm pop: full t0 293.6 us, MMA-only t1 194.4, dequant-only t2 191.1, producer-only t4 145.1; serial in every class, g 0.34 low / 0.21 high / 0.18-0.21 ft; ceilings ~5.2% headline low / ~3% high (CLAIMS:70-71). Overlap closed: K-pair bbca0105 x1.09-1.14 slower; ordered anti-phase 17d3fdf1a6594d33 formed (retire skew 0.44-0.47) yet x1.07-1.16 slower, reopen 'a cheaper dequant' (CLAIMS:73); scale expansion moved to producer warps 6be2a8d3 gated 1.03-1.10x slower (LESSONS:182). Master rega dq ~24 instr per k16 step (~7 scale; source count, 6be2a8d3 spec) vs moe Engine A dq2 ~17 dequant instr per step (moe CLAIMS:73, ncu P3c) and its L3 paired scale decode landed -2.0 ALU instr/step, eac bench x1.10-1.12, check f2100f49 every row +2.3-7.1% (moe CLAIMS:74).
- **Gain if it works:** 3.0% headline; **p** 0.25; **EV** 0.75% (bar: gain >= 3.0%, EV >= 1.5%); **below the bar** (EV 0.75% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge runs.
- **Decision rule:** Read '[rega-dq] VERDICT <cls>' (warm L2, pop routing, best exact arm). (1) best arm cuts pure-dequant MODE 5 by >= 25% (A3 predicted ~25-30%) AND the w13 + w2 pair MODE 0 by >= 8% at low: if it routed, the headline is the result -> append [lead]/[landed] by the board, and port the same arm to the moe/decode wgmma hot half (wg_kernel calls the identical dq_stage, moe submission.py:1128 / decode :15330; decode T=256 hot half 134.5 us vs ~39 us math floor) as the 10-30 phase-A candidate, as its own hypothesis; if print-only, mint the 10-30 routed rega build first, port second. (2) MODE 5 cut >= 25% but pair MODE 0 cut < 3% in every class: the stage contention is in the MMA pipe / register file / smem, not dequant issue count -> close cheaper-dequant on the rega tile ([refuted]), name producer-side dequant (SS wgmma from smem bf16, the ssdq lineage) as the last prefill angle, and do NOT port to the wg hot half on this evidence (moe CLAIMS:62 already says that hot half's gap is fixed costs). (3) MODE 5 cut < 12% for every arm: master's rega dequant is already near its instruction floor (and the Engine A census does not transfer to the rega tile) -> close the rega dequant lever and the hot-half port. (4) anything between (MODE 5 cut 12-25%, or pair cut 3-8%): [shelf] with the per-class ratios, routed only where <= 0.97, offered as a stack row. INVALID (DQ0 fidelity outside 0.97-1.03, spill, or any arm not torch.equal) -> no conclusion; fix and rerun.
- **Instrument:** prepare-time CUDA-graph timings in the judge check's own prepare(): '[rega-dq] attrs' (regs / local / smem / CTAs per SM per DQ instantiation), '[rega-dq] SCALES' (bank scale-byte range counts), '[rega-dq] EXACT <cls>' (torch.equal per arm), '[rega-dq] TIME <cls> <gemm> <routing> <L2>' (master us, DQ0 fidelity, per-arm MODE 0 / 2 / 5 ratios per round), '[rega-dq] PAIR <cls>' (w13 + w2 MODE 0 ratios), '[rega-dq] ROUTE <cls>: A1|A2|A3|master|print-only', and one '[rega-dq] VERDICT <cls>' line per class, all within the last 16 KB of the log

## Risk

User-requested (10-29 ~15:04Z steer: run below-bar diagnostics). Builds on CLAIMS:70-71 [lead] (rega super-serial, cb22e610) and CLAIMS:73 [refuted] anti-phase whose reopen it meets; avoids CLAIMS:44 [refuted] (scale expansion on producer warps) by staying consumer-local. Cross-problem: moe/decode's wgmma hot half (wg_kernel, rewrite/lib/wg_core.py; moe submission.py:1128, decode :15330) calls the identical master-form dq_stage (dequant8 + per-k scale convert), while the Engine S hot half (es_hot) already uses dq2; decode's hot half at T=256 is 134.5 us vs ~39 us math floor, but moe CLAIMS:62 says it already runs at the K-step rate with a fixed-cost gap, so a port pays only if this run shows full-mode (not just MODE 5) gains. Keep this run prefill-only (one area); the port is a separate hypothesis per --decision. Risk: A2/A3 need a permuted bank copy (~0.9 GB per state) if routed; scale-range precondition [0x08, 0x7E] must hold on the actual bank or A1/A3 stay OFF; prepare-time budget (one extra nvcc build) - print-only if deadlines would be at risk.

## Adversarial review

Reviewed 2026-10-29T175903Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: this is the exact reopen named by CLAIMS.md:73 ('a cheaper dequant'), on a fresh base (1.5008, c63801e7 unchanged). The mechanism is sound and one run is readable, because the arms are interleaved against an in-module DQ0 control. The build must fix the prepare-time budget first: 16 instantiations started at ~181 s can overrun the 300 s prepare limit. It must also take the timing gates against DQ0 and key the decision on pair MODE 0, not on MODE 5.

### Justification
A, mechanism. Master's dq_stage/dequant8 (submission.py:2673-2741) costs about 15-16 INT ops per k16 step: per bf16x2 SHF + 2 LOP3 + SHL, then HFMA2 against -0. The scale path adds about 7 per step: HADD2.F32 x2, FMUL x2, F2FP pack x2, plus the amortized e4m3 cvt and LDS. A2 cuts the code path to about 10 per word with moe's ext_prop (muse_moe_w4a16/submission.py:5033-5045: 1 + 2 + 2 + ~5). A1 cuts the scale path to about 2 per step (prmt + IMAD), with the broadcast folded into HMUL2 .H0_H0/.H1_H1. So A3 removes roughly 35-40% of the dequant instructions, more than the 24 -> 17-19 the hypothesis claims. Its own census also does not add up: 11 + 4 + 4 = 19, not 17. Exactness holds: e2m1 x e4m3 x 2^-7 has at most 6 significant bits. The judge's scales are clamped to [8, 448] (nvfp4.py:123), i.e. bytes 0x50-0x7E, and even wide_scales stays at >= 0x08, so the [0x08, 0x7E] precondition will hold and A1/A3 will not switch OFF. Evidence for the lever (cb22e610, CLAIMS.md:70-71): t0 - t1 (what dequant adds over MMA-only) is 99 of 294 us at low w13, 157 of 747 at high w13 and 643 of 3570 at ft w13. The overlap levers K-pair, anti-phase and ping-pong all lost (CLAIMS.md:44, 72, 73), which points at per-SMSP contention (ALU pipe / RF ports during RS wgmma) rather than latency. Cutting the dequant's own ALU work attacks that directly. WEAKEST LINK: whether t0 tracks dequant instruction count at all. At high and ft, MODE 2 already sat on the producer floor (t2 = t4 at ft; 356 vs 340 us at high), so a cheaper dequant can only pay there through reduced contention, not a shorter serial dequant. Expect the gain to concentrate at low; high and ft could be flat. Moe's dq2 ran in mma.sync Engine A and at the RS wgmma ladder (LOG:1295, (e) dq2 + RS ~29.8 el/clk/SM vs Engine A ~27); it has never been measured on the rega tile. That is exactly what this run answers. Prior check: experts-rega-producer-scale-expand-all-t (CLAIMS.md:44, 6be2a8d3) moved the same scale work onto producer warps through an extra mbarrier hop and smem ring. This run cuts work on the consumer and adds no barrier or smem, so the difference is real. experts-triton-nvfp4-packed-asm-dequant is a different engine, superseded per CLAIMS.md:39. Not a duplicate. Expected shapes and directions match TASK.md weights and the b2941ea profile: rega is 48-53% of sliding rows; turn_1 is the class with the largest g.

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue and the w2st arm a landing (CLAIMS.md:51, 54, 56): the dq module derives from _rega_fsilu_src(), so master, DQ0 and A1-A3 all carry the same epilogue. They do not confound the arm/DQ0 ratio. They do confound arm/master by the MODE plumbing (anat r2 m0/m was 0.974-1.023, worst at ft cold), which can push a do-nothing arm toward the 0.97 route threshold.
- norm-gather + two-phase align (CLAIMS.md:42, 46): they fix the routed inputs and tables (xs/hs from master's own _experts_rega). All arms share them. No confound.
- rega engine (CLAIMS.md:41): the dq_stage under test is this engine's. Changing it is the point, not an interaction.

### Interference and A/B needs
No companion. The diagnostic controls itself: master, DQ0 (master's dequant text in the dq module) and A1/A2/A3 are timed interleaved in the same prepare, on the same inputs, L2 modes and routings. A1 vs A3 isolates the scale path from the code path, and MODE 5 vs MODE 2 vs MODE 0 separates instruction cost from stage time. A gain and a loss are both readable from the per-arm/DQ0 ratios without another check. An ab-revert of fast-silu or w2st would not change what the arms measure. One condition makes this hold: the verdict and the route gate must use arm/DQ0 as well as arm/master. That way a route cannot be credited to the MODE plumbing, which anat r2 showed moves m0/master by up to 2.6%.

### Open questions
- BUILD MUST (prepare budget). Pick one of two. (a) Cut instantiations to fit: DQ in {0, 1, 3} x {w13, w2} at BIG=false only (6 instantiations, low + high classes), with ft skipped and printed SKIPPED. A2 alone is redundant because A3 - A1 gives it. (b) Keep all of them but start the dq build in a background thread at the top of prepare(). The source is static text derived from _rega_fsilu_src(), so nvcc overlaps master's first ~180 s; join it in _rega_dq_phase. Either way, gate the build start predictively: start only if t_now + 1.3 x (fsilu build seconds x n_inst / 4) + 25 s <= 270 s, else print '[rega-dq] OFF: budget'. Print the build seconds and the prepare total.
- Use moe's mul_bf16x2_h<H> spelling (mov.b32 {l,l} into fma.rn.bf16x2), not __byte_perm 0x1010/0x3232. Moe's own comment (muse_moe_w4a16/submission.py:5083) says the byte_perm form does NOT fold into HMUL2 .H0_H0/.H1_H1, so it would add an instruction per bf16x2.
- Scale layout differs from moe. In rega's record the scale bytes are interleaved per k16 step (byte 2k = row g, byte 2k+1 = row g+8 of sc.x/sc.y; dq_stage's hs[k] -> fs.x/fs.y). So [b_g, 0, b_g8, 0] is a ONE-source prmt of one word (selector from bytes 2k, 2k+1 of sc.x for k<2, sc.y for k>=2), not moe's two-source scale_pair_rt. Keep moe's IMAD form (rt16 m16) so the add stays on the FMA pipe. In rega, dequant8 pairs i=0,2 use sg (H0) and i=1,3 use sg8 (H1) from the same code word, so the A2 permutation is _ea_luts index 2 per pair, applied per 32-bit word of every uint4.
- Decision rule. Key lead/landed/refuted on the w13+w2 pair MODE 0 ratio against BOTH DQ0 and master (warm, pop). Use MODE 5 only to attribute the result and to decide the moe/decode hot-half port. As written, branch (1) needs MODE 5 >= 25%, so an arm that routes with a pair cut >= 8% but a MODE 5 cut of 20% would file as [shelf]. A MODE 5 cut < 12% next to a pair cut >= 3% should read as 'contention relief', not 'close'.
- Route gate: require arm/master <= 0.97 AND arm/DQ0 <= 0.97 in every round, L2 mode and routing, so the MODE plumbing (anat m0/m 0.974 at ft cold) cannot route a do-nothing arm.
- INVALID scope: an arm that is not torch.equal should be OFF for that arm only, with its max|diff| printed. It should not invalidate the whole run. DQ0 fidelity outside 0.97-1.03 should invalidate only that class.
- Log tail: anat r2's low-class TIME lines fell outside the 16 KB tail (cb22e610.log starts mid-ft). The <= 3 KB summary printed at once must itself carry the decision numbers per class: per-arm MODE 0/2/5 ratios vs DQ0 and vs master (warm pop, plus cold pop), the PAIR ratio, ROUTE, VERDICT, attrs and SCALES counts. Do not rely on the per-routing TIME lines surviving.
- Cheap exhaustiveness check, optional: a one-warp canary that runs dq_stage DQ0 vs each arm over every byte value in each nibble-pair position and every scale byte 0x08-0x7E, compared bitwise, like moe's dq_exact (0 / 121,856, LOG:677). This makes 'exact by construction' a proof rather than coverage over the routings that happen to be sampled.

