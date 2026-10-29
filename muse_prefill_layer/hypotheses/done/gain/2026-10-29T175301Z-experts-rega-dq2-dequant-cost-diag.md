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
status: done
review_at: 2026-10-29T175903Z
review_companions: null
review_best: 1.5007957663757285
review_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
stage: harvested
claimed_at: 2026-10-29T175904Z
base_sha: 2f7dcbf03b7345f8e21d32c433510942d4a23926
base_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
branch: hyp/muse_prefill_layer/experts-rega-dq2-dequant-cost-diag
worktree: .worktrees/muse_prefill_layer/experts-rega-dq2-dequant-cost-diag
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T182804Z
impl_commit: 21127beb9e921d0e9aad874e268d9c6a711da7cb
submission_sha256: 56570c0030c551c01cef091abdab18e4d17b06699ef19ca8f2c98253688568cf
norm_sha256: 368cef334baba077650e0fcd582c4bc5f4cce300e1ceec415c25389e2ab0d562
submission_blob: d8ce4cbf6e2e54cb8294363e025022d9284a0762
lint_warnings: 121
attempt_file: 2026-10-29T182805Z-check-experts-rega-dq2-dequant-cost-diag.md
prepared_at: 2026-10-29T182805Z
job: 835e1a443a604eb2
dispatched_at: 2026-10-29T182813Z
outcome: gain
verdict_detail: gain
speedup: 1.5555848458088881
speedup_ci: [1.5322510731217547, 1.5789186184960213]
harvested_at: 2026-10-29T183514Z
best_at_harvest: 1.5007957663757285
promotion: pending
mechanism: flat
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


## Implementation notes

**Mechanism:** a prepare-time diagnostic of the rega dequant's instruction cost: a separate muse_prefill_rega_dq module (built in a background thread from prepare()'s start) times moe's exact dq2 / L3 dequant arms (A1 scale-L3, A2 ext_prop on a permuted bank copy, A3 both) against an in-module DQ0 control in MODE 0 / 2 / 5 on the w13 + w2 rega GEMMs of the low / high / ft classes, with a gated per-class route of the best exact arm.

Implemented 2026-10-29T1826Z in the worktree's `muse_prefill_layer/submission.py` (+1264 lines, 0 deletions; 447,531 B, cap 1,000,000). Base master c63801e7.

What changed, where:
- `prepare()` is now a thin wrapper: `_rega_dq_start(t0)` first, then master's body (renamed `_prepare_master`, unchanged except two inserts), with `_rega_dq_join` in a `finally`, so no build thread outlives prepare() even if master's setup raises. After `_rega_w2st_phase` and before `del fsilu_info`: `dq_lines = _rega_dq_phase(...)`; its summary lines are echoed after the w2st block (last lines of prepare).
- `_experts_rega`: one prepare-time int per class, `state['rega_dq_ok'][cls]` (1/2/3), selects `rega_dq_ext.rega_gemm(..., mode=0, dq)` for BOTH GEMMs (A2 / A3 read `state['rega_dq_bank']`, the permuted code copy; global scales are master's). Precedence: rega_lo variant, then dq, then fsilu / w2st, then master. When nothing routes the three dq keys are absent and the launches are master's.
- New block before `kernel_fn` (`REGA_DQ_*`, `_REGA_DQ_*`, `_rega_dq_*`): 18 asserted (old, new) edits on `_rega_fsilu_src()` (each old exactly once, else `[rega-dq] OFF: edit`) + one cpp-declaration edit: `rega_kernel<G, BIG, DQ>` / `consume_item<NB, G, DQ>` with a runtime `mode`, 16 instantiations (`rega_init` returns 16 rows, flag = G + 2 BIG + 4 DQ), `rega_gemm(..., mode, dq)`, and an exactness-canary kernel `rega_dq_exact`. DQ 0 calls master's `dq_stage` verbatim. A1 = master's code extraction, `dq_scale_pair<KK>` (one-source `prmt` of rega's interleaved scale word -> [b_g, 0, b_g8, 0], then `mad.lo.u32 t, m16, 0x77807780` with moe's `rt16` so the add stays an IMAD) and `dq_mul_h<H>` (moe's `mul_bf16x2_h` spelling: `mov.b32 {l,l}` / `{h,h}` into `fma.rn.bf16x2`). A2 = moe's `ext_prop<I>` on the permuted word (pair i = `ext_prop<i>`, i even row g / sg, i odd row g+8 / sg8, per the review), master's f16 scale path. A3 = both. The comment rejecting the register-prmt and smem LUTs is in `_REGA_DQ_HELPERS`.
- MODE 0 = master's k loop with the arm's dequant. MODE 2 = no MMA: ring waits / releases as master's, the arm's dequant every stage, the registers kept live by anat r2's predicated-store sink (`dq_sink`: a setp and 8 `@p st.shared.v4` on the dead flag; no adds). MODE 5 = pure dequant issue: the producer is skipped (`tid == 0 && mode != 5`), no ring waits, no MMA, no epilogue; KS dequants per item of the stage buffers (kb % STAGES) into the sink.
- Phase `_rega_dq_phase`: join; `[rega-dq] attrs` (16 rows; local bytes > 0 = spill -> that arm OFF for that class; DQ0 bad -> class skipped); `[rega-dq] SCALES` (bank scale bytes < 0x08 / == 0 / > 0x7E for q13, q2; any outside -> A1, A3 OFF); the eager canary (announced, synchronized, printed); the permuted bank (`_rega_dq_perm_bank`, every 8192-record chunk checked lossless by the inverse, scale bytes asserted unchanged); then per class (low / high / ft with rega_ok, low only at rega_low_variant 0; deadlines 255 s, ft 240 s): EXACT on random / skewed / one / pop (fsilu's calibrated alpha) - hs (MODE 0 w13 on xs) and y (MODE 0 w2 on master's hs, sentinel-prefilled) torch.equal vs DQ0 (arm gate) and vs master's dispatched launches (route gate), max|diff| when not equal; the first step announces every launch (MODE 0, 2, 5 of every DQ) with a synchronize. TIME on pop and random through `_rega_fsilu_bench` (2 rounds x 11 alternations, cold + warm, empty graph subtracted), interleaved in three sets: w13 alone and w2 alone (master's dispatched 'm' + DQ x MODE 0/2/5) and the w13 + w2 pair ('m' + DQ MODE 0).
- VERDICT (per class, warm pop): INVALID if DQ0 MODE 0 / master is outside 0.97-1.03 on either GEMM (that class only). Otherwise the best arm by pair / DQ0; pair cut = 1 - max(arm/DQ0, arm/master). >= 8% LEAD, 3-8% SHELF; < 3%: REFUTED if MODE 5 cut >= 25%, CLOSE if < 12%, else FLAT-SHELF. The MODE 5 cut ((w13 + w2) MODE 5 vs DQ0's) is the attribution ('issue-count' >= 12%, else 'contention relief') and the wg hot-half port call (YES >= 25%).
- ROUTE: an arm passes when it is equal to master's dispatched launches on all four self-check routings and its pair MODE 0 is <= 0.97 x master AND <= 0.97 x DQ0 in every round, L2 mode and routing (8 values each). The lowest warm-pop pair/DQ0 among the passing arms then runs master's `_rega_replay_check` (`REGA_FSILU_REPLAY_PLAN[cls]`, keys `REGA_DQ_STATE_KEYS` = w2st keys + `rega_dq_ext` / `rega_dq_ok` / `rega_dq_bank`, so the permuted bank is copied with the state). It must start before REGA_COPY_DEADLINE_S, else ROUTE prints `print-only`.
- Summary (<= 3000 chars, capped): SUMMARY (build s, bg start, est, join wait, per-step s, prepare total, rega_dq_ok), OFF, attrs, SCALES + canary counts, a key line, then per class RATIOS (warm pop: master us, DQ0 MODE 0/2/5 us, fidelity, each arm's MODE 0/2/5 vs DQ0 [MODE 0 vs master]; cold pop: fidelity + the same ratios), PAIR (pop warm, pop cold, random warm: vs DQ0 / vs master), ROUTE, VERDICT. It is printed at the phase end and echoed as the very last prepare() lines. On synthetic timings it measures 2,958 chars (scratch `test_report.py` / `.out`).

Review MUST-fix items applied:
1. Budget: option (b). `_rega_dq_start` runs first in prepare(). It derives and audits the source, gates predictively (t + 1.3 x est + 25 s <= 270 s, est = 40 s x 16/4 = 160 s; at t ~ 0 the gate value is ~233 s; otherwise `[rega-dq] OFF: budget (...)`), then starts a non-daemon `threading.Thread` running `load_inline` with master's REGA_CUDA_CFLAGS. `_rega_dq_phase` joins it, and so does prepare()'s finally. Build seconds, bg start, join wait and the prepare total are printed. Arch-list safety: the MAIN thread sets TORCH_CUDA_ARCH_LIST=9.0a before starting the thread (after checking that `_get_cuda_arch_flags()` accepts it; if not, no build) and restores it only after the join. Master's builds in between see prev = 9.0a and restore 9.0a, so no thread ever pops the variable under another thread's build.
2. `dq_mul_h<H>` is moe's `mul_bf16x2_h` spelling, not `__byte_perm`.
3. Scale bytes are interleaved per k16 step: a one-source prmt of one word (selector 0x4040 | (2KK+1) << 8 | 2KK; sc.x for k < 2, sc.y for k >= 2), with moe's IMAD (rt16) form.
4. The decision is keyed on the pair MODE 0 against BOTH DQ0 and master. MODE 5 only attributes it and decides the port; 'MODE 5 cut < 12% with pair cut >= 3%' prints as 'contention relief'.
5. Route gate: arm/master <= 0.97 AND arm/DQ0 <= 0.97 in every round, L2 mode and routing. Only state flags are read on the timed path, and the permuted bank is in the replay-check key set.
6. An arm that is not equal is OFF for that arm and class only, with its max|diff| printed. DQ0 fidelity outside 0.97-1.03 makes that class INVALID only.
7. The <= 3 KB summary is printed at the END (echoed last) and carries per class the per-arm MODE 0/2/5 ratios vs DQ0 plus MODE 0 vs master (warm pop with absolute us, cold pop), PAIR, ROUTE and VERDICT, with attrs (regs / local / occupancy) and SCALES counts.
8. Done (optional): the exhaustive canary covers all 256 values of each of the 4 code-word bytes (so every code at every nibble position) x scale bytes 0x08-0x7E, 121,856 cases x 2 k16 positions x 4 fragments, every arm compared bitwise with master's per-k body. It also checks the Python permutation, because pwords comes from the same `_rega_dq_perm_words` that builds the bank. If DQ0 returns all-zero fragments, every arm is OFF.

Deviations from the spec, and why:
- The filled prompt `hypotheses/prompts/filled/implement-muse_prefill_layer-experts-rega-dq2-dequant-cost-diag.md` contained the pre-flight AUDIT template, not the implement prompt. I followed `.claude/skills/churnsubs/IMPLEMENT_PROMPT.md`, `BUILD_BRIEF_CUDA.md` and the caller's message instead.
- MODE 2 keeps its registers live with anat r2's zero-ALU sink, not "one dependent add". 32 adds per stage would themselves add to the instruction count being measured, and the spec asks to reuse anat r2's mode plumbing.
- MODE 5 turns the producer off. If the consumers skip the ring while the producer runs, its empty-barrier wait times out (error word 1, ~34 ms stall). MODE 5 also rotates through the 4 stage buffers (kb % STAGES) instead of stage 0 alone (the same instruction count; nothing for the compiler to CSE).
- Exactness compares arms with DQ0, which gates the arm. Routing additionally requires both GEMMs to be equal to master's DISPATCHED launches. The module derives from the fsilu_w2a source, so a route is possible only in classes where fsilu and w2st are ON. This holds for every class on the current board, and routing can never change the silu form.
- In the summary, "vs master" is given for MODE 0 only, because MODE 2 / 5 have no master counterpart. DQ0's absolute MODE 0/2/5 us are printed, so any arm-vs-master ratio can be derived.
- The build thread raises its own nice value by 10 (`os.setpriority(PRIO_PROCESS, 0, ...)` inside the thread: a per-thread attribute on Linux, inherited by the ninja / nvcc children). Under CPU contention, master's own builds (ssdq 5-37 s, rega v2 ~40-80 s before its 150 s ft-start deadline, fsilu ~128 s) keep priority. This is not in the spec; I added it because master's ft deadline is the one master behaviour the overlap could break.
- The est uses a constant 40 s per 4 instantiations (fsilu 39.3 s, anat 39.8 s in cb22e610), because fsilu's own build seconds are not known when prepare() starts.

Evidence (scratch `/tmp/claude-1000/-home-user-git-musespark/c24a6e21-451f-44bd-9a25-cf12944ffdc0/scratchpad/build-pf-dq2/`):
- `emulate_dq.py` / `emulate_dq.out`: a numpy proof that the A1 scale pair equals master's bf16x2_bcast(e4m3 x 2^119) bits for every (byte_g, byte_g8) in [0x08, 0x7E]^2 at both KK (28,322 cases), and that 270 out-of-range non-NaN bytes differ (so the SCALES gate is needed). It also proves `ext_prop<i>(perm(q))` == master's extraction i for every 16-bit half value (exhaustive) + 2^20 random words, and that unperm(perm(q)) == q.
- `mock/`: master's fsilu source and the derived dq source both pass `g++ -std=c++20 -fsyntax-only` against stub CUDA/torch headers (asm discarded, launches rewritten, all 16 instantiations reached through rega_init and the dispatch). Two mutation tests (`mut1.cpp`, `mut2.cpp`) confirm the mock catches template-body errors.
- `test_report.py` / `test_report.out` exercise the verdict, the route gate and the summary on synthetic timings: LEAD / SHELF / INVALID, a NOT-eq-master arm blocked from the route, and the summary at 2,958 chars.
- Master's `_rega_audit_cpp` and the templated-accessor scan on the derived source return no findings.

- Cross-check against the H100 lane (takemyjob hopper/rgx 1a17a00, `prototypes/hopper-probe/probes/rgx/rgx_dev.cuh` `rgx_scale_pair` / `dq_stage_x`, `rgx.cu` `permute_bank`; write-up `campaigns/2026-10-26-expert-bandwidth/research/h100-rega-1029.md`: arms bit-exact on gpunode06, SASS dequant per k16 master 26 / A1 22 / A2 22 / A3 18, A3 pair x0.84-0.89 low, x0.91-0.92 high). My selectors are the same (KK 0 -> 0x4140, KK 1 -> 0x4342; rgx K2 = 0 / 2 gives the same values), and so are the per-word permutation formula (nibble 4((3-p)+4h) -> lut index 2, code words 0..1023 of each 1152-word record, scale words copied), the k -> (sc.x / sc.y, KK) mapping, and the `o[i] = mul_h<i&1>(ext_prop<i>(q), sp)` assignment. A3 is in the build: option (b) keeps all four DQ values, so no budget trimming drops it.

Risks:
- The 16-instantiation build time is the main unknown (est 160 s; anat's 4 inst x 6 modes built as fast as fsilu's 4). A build still running at the phase is waited out, because load_inline cannot be interrupted.
- CPU contention with master's builds is mitigated by nice +10, not eliminated.
- An illegal-instruction fault in a new kernel would poison the context for the whole run. The canary and the announced eager launches name the kernel, but the run would be lost.
- A2 / A3 hold a ~1.2 GB permuted bank copy (x3 during the replay check).

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-dq2-dequant-cost-diag diff` (7 hunks: import, prepare wrapper, the dq phase call + echo, three in `_experts_rega`, the new block before `kernel_fn`).
- [x] CUDA-graph capturable: the timed path changes only in `_experts_rega`, which reads plain prepare-time ints / module / bank refs from `state` (`rega_dq_ok`, `rega_dq_ext`, `rega_dq_bank`) keyed by `_rega_class(num_tokens)` (a shape). No `.item()` / `.cpu()` / synchronize / print, and no allocation or branch on input values. The dq `rega_gemm` host code is master's (tensor maps encoded per call on the host). Every sync / print in the new code is in prepare-only functions.
- [x] Nothing derived from `data` is cached across calls: the state holds only the prepare-built module, the per-class ints and the permuted copy of the WEIGHT bank. The phase's synthetic routings and buffers are local and dropped.
- [x] Weights: the permuted bank is a lossless retiling of the NVFP4 code bytes made in prepare() (each 32-bit word's nibbles move position; every chunk is checked by the exact inverse, scale bytes are asserted unchanged). Codes stay NVFP4 bits and scales E4M3 bits in global memory, dequantized in-kernel. A1 / A3 produce the same bf16 values (canary + torch.equal gate; the SCALES precondition is checked once).
- [x] Activations BF16 as master (xs / hs / y bf16, fp32 accumulate, same epilogue); the KV cache and queries are untouched: n/a for attention.
- [x] n/a (attention untouched): every key is still attended (fa3 / fa3c paths unchanged).
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocess or network. The background build is a `threading.Thread` (not a process) that calls `load_inline` (nvcc via torch, as master's builds do), and it is joined before prepare() returns. `os.setpriority` only lowers that thread's own CPU priority.
- [x] All work happens inside the timed call; no output caching or replay. Routing only selects which exact kernel the timed call launches.
- [x] First line `# okc: problem=muse_prefill_layer` kept; size 447,531 B (cap 1,000,000).
- [x] No triton.autotune, no recompiling per call (one build in prepare).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`). Exit 0, `OK with warnings`: 121 warnings (master: 112). 116 are `host-sync ... (outside kernel_fn's call graph; fine in prepare())`; the 9 new ones are the dq phase's prepare-time synchronizes. The 5 `dtype` warnings are all master's lines (L3490, L3491, L3796, L4346, L8077: uint8 byte buffers). Full output: scratch `lint.out`. `python3 -I -m py_compile` passes, and `__pycache__` was removed from the worktree.

## Result

- job `835e1a443a604eb2` (check), status done, passed True, tests 5/5
- speedup 1.5556x, 95% CI [1.5323, 1.5789], score 249040.67498713257 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.5007957663757285, "ci": [1.4782838298800927, 1.5233077028713644], "job": "b2941ea23dc743a3"}; base result: muse_prefill_layer/hypotheses/BEST.result.json
- **outcome: gain** (gain); mechanism: flat (7/8 predicted rows past ±2%)
- files: `muse_prefill_layer/hypotheses/done/gain/experts-rega-dq2-dequant-cost-diag.result.json`, `muse_prefill_layer/hypotheses/done/gain/experts-rega-dq2-dequant-cost-diag.patch`; branch `hyp/muse_prefill_layer/experts-rega-dq2-dequant-cost-diag`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 700.96 | 709.56 | 1.0471 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1083.73 | 1067.11 | 1.0327 | gained |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2088.45 | 2077.81 | 1.0327 | gained |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3614.44 | 3533.58 | 1.0201 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 11011.98 | 11052.78 | 1.0425 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 10865.92 | 10545.85 | 1.041 | gained |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14763.78 | 14570.54 | 1.0343 | gained |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18546.94 | 18087.08 | 1.0138 | flat |

gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_1. Expected hit: 3/3. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands. Gain not attributable to the mechanism (race/draw?): its predicted rows were flat.

