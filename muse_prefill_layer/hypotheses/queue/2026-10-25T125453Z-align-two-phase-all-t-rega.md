---
problem: muse_prefill_layer
slug: align-two-phase-all-t-rega
created: 2026-10-25T125453Z
base: master
area: "MoE align/dispatch on the rega path (200 < T <= 16384): _align_parallel replaced by the two-phase chunk-count + prefix-place align, including the first_turn class"
change: >
  Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft
  master; patch
  muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.patch) onto master
  a1a5497d, extended to the first_turn class. WHY NOW: the new master's profile (job
  07a069bc69ba42a8) shows _align_parallel_kernel at 17.1/16.7 us (turn_1), 71.2/69.6 us (turn_4)
  and 392.5-418.9 us at T=16384 (3.2% of every first_turn shape, including the weight-138 sliding
  ctx32768 row). The ft class (experts-turn-rega-s4-first-turn-n128) moved first_turn from
  SGLang's moe_align_block_size (about 50 us in the d792 profile) onto _align_parallel, whose 33
  programs each stream all 131072 route ids twice with a CHUNK x 32 compare (prepare tracer:
  441-454 us eager at T=16384). In d792 the two-phase align measured _align2_place_kernel at
  7.2/7.4 us at turn_1 and dropped out of the top-12 rows (< 17.6 us) at turn_4, against 71 us for
  master; sliding turn_4 moved +4%. CHANGE: (1) Port from the patch with its logic unchanged: the
  ALIGN2_* constants, _align2_count_kernel, _align2_place_kernel, _align_two_phase, _align2_cases,
  _align2_first_mismatch, _align2_gate_time and _align2_setup, and the call _align2_setup(state,
  device) right after _align_par_setup in prepare(). It must come before _rega_setup, so rega's
  gates and copy/replay checks run on the align that will be timed. (2) Give _align_dispatch a
  three-way class on T: 'low' T <= 1024, 'high' 1024 < T <= MARLIN_MAX_TOKENS, 'ft'
  MARLIN_MAX_TOKENS < T <= REGA_FT_MAX, with the key f'{bsm}:{cls}'. (3) Change ONE call site: in
  _experts_rega (master line ~3158), the call _align_parallel(local_ids, 0, LOCAL_EXPERTS,
  REGA_BSM) becomes _align_dispatch(local_ids, REGA_BSM, state). _experts_marlin and _experts_ssdq
  keep master's _align_parallel. They are fallbacks now, and keeping them byte-identical keeps
  attribution on the rega rows. Leave the diagnostic _align_parallel call in the rega gate code
  (~line 3295) alone. (4) Set ALIGN2_BSMS = (16,) only, and ALIGN2_GATE = ((16, 753, 'low'), (16,
  3012, 'high'), (16, 16384, 'ft')). Add 4500, 8192 and 16384 to ALIGN2_TESTS_T. At T >= 8192,
  _align2_cases also yields a 'local-saturated' case (every route local: ids % 32), so the tail
  program's block_experts and sentinel loops cover the ft worst case (8222 blocks, cap 131552).
  Keep ALIGN2_CHUNKS = (512, 1024): NCH_P2 is then 256 or 128 at T=16384 (count table 32 or 16
  KB). The tail program alone fills the sentinel range [total, cap), about 100k int32 at T=16384.
  If that tail shows more than 10 us in a gate diagnostic, spread the fill over the place programs
  by slot range instead (the stores stay disjoint), and print which form ran. (5) Add 'align2_ok'
  to REGA_STATE_KEYS. Its value is a plain dict of ints. The judge prepares once and copies the
  state by cloning tensors, so no data_ptr or pointer-equality condition may gate anything.
  Dispatch reads only T and state['align2_ok']. (6) Gate per class. First check equivalence:
  torch.equal on local_ids, sorted_ids, block_experts and n_padded against _align_parallel for
  every case. Then time cold-L2 graph medians over ALIGN2_ALTERNATIONS with the empty-graph
  baseline subtracted. A class turns ON at <= ALIGN2_RATIO 0.98 x master. Print the per-class
  align2 PATH CHOSEN line again at the very end of prepare(), after the rega PATH CHOSEN lines, so
  the archived log tail keeps it. kernel_fn rules: no host sync, no prints, no try/except, and
  per-call torch.empty sized by shape only (the count table is (nchunks, 32) int32 from T).
  Nothing is cached from inputs.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_align_parallel_kernel 393-419 us -> ~30-60 us (count + place over 128-256 programs each): about -350 us per shape, ~-3% on the weight-138 sliding ctx32768 row"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "align 71/70 us -> ~10-15 us as measured in d792 (sliding turn_4 +4% there): ~-2.5% sliding, ~-1.5% full"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "align 17 us -> ~9 us (place 7.2 + count ~2 in d792): ~-1%, at the edge of drift; read the align rows, not the headline"
risk: low
depends_on: experts-turn-rega-s4-first-turn-n128
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# align-two-phase-all-t-rega

## Change

Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft master; patch muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.patch) onto master a1a5497d, extended to the first_turn class. WHY NOW: the new master's profile (job 07a069bc69ba42a8) shows _align_parallel_kernel at 17.1/16.7 us (turn_1), 71.2/69.6 us (turn_4) and 392.5-418.9 us at T=16384 (3.2% of every first_turn shape, including the weight-138 sliding ctx32768 row). The ft class (experts-turn-rega-s4-first-turn-n128) moved first_turn from SGLang's moe_align_block_size (about 50 us in the d792 profile) onto _align_parallel, whose 33 programs each stream all 131072 route ids twice with a CHUNK x 32 compare (prepare tracer: 441-454 us eager at T=16384). In d792 the two-phase align measured _align2_place_kernel at 7.2/7.4 us at turn_1 and dropped out of the top-12 rows (< 17.6 us) at turn_4, against 71 us for master; sliding turn_4 moved +4%. CHANGE: (1) Port from the patch with its logic unchanged: the ALIGN2_* constants, _align2_count_kernel, _align2_place_kernel, _align_two_phase, _align2_cases, _align2_first_mismatch, _align2_gate_time and _align2_setup, and the call _align2_setup(state, device) right after _align_par_setup in prepare(). It must come before _rega_setup, so rega's gates and copy/replay checks run on the align that will be timed. (2) Give _align_dispatch a three-way class on T: 'low' T <= 1024, 'high' 1024 < T <= MARLIN_MAX_TOKENS, 'ft' MARLIN_MAX_TOKENS < T <= REGA_FT_MAX, with the key f'{bsm}:{cls}'. (3) Change ONE call site: in _experts_rega (master line ~3158), the call _align_parallel(local_ids, 0, LOCAL_EXPERTS, REGA_BSM) becomes _align_dispatch(local_ids, REGA_BSM, state). _experts_marlin and _experts_ssdq keep master's _align_parallel. They are fallbacks now, and keeping them byte-identical keeps attribution on the rega rows. Leave the diagnostic _align_parallel call in the rega gate code (~line 3295) alone. (4) Set ALIGN2_BSMS = (16,) only, and ALIGN2_GATE = ((16, 753, 'low'), (16, 3012, 'high'), (16, 16384, 'ft')). Add 4500, 8192 and 16384 to ALIGN2_TESTS_T. At T >= 8192, _align2_cases also yields a 'local-saturated' case (every route local: ids % 32), so the tail program's block_experts and sentinel loops cover the ft worst case (8222 blocks, cap 131552). Keep ALIGN2_CHUNKS = (512, 1024): NCH_P2 is then 256 or 128 at T=16384 (count table 32 or 16 KB). The tail program alone fills the sentinel range [total, cap), about 100k int32 at T=16384. If that tail shows more than 10 us in a gate diagnostic, spread the fill over the place programs by slot range instead (the stores stay disjoint), and print which form ran. (5) Add 'align2_ok' to REGA_STATE_KEYS. Its value is a plain dict of ints. The judge prepares once and copies the state by cloning tensors, so no data_ptr or pointer-equality condition may gate anything. Dispatch reads only T and state['align2_ok']. (6) Gate per class. First check equivalence: torch.equal on local_ids, sorted_ids, block_experts and n_padded against _align_parallel for every case. Then time cold-L2 graph medians over ALIGN2_ALTERNATIONS with the empty-graph baseline subtracted. A class turns ON at <= ALIGN2_RATIO 0.98 x master. Print the per-class align2 PATH CHOSEN line again at the very end of prepare(), after the rega PATH CHOSEN lines, so the archived log tail keeps it. kernel_fn rules: no host sync, no prints, no try/except, and per-call torch.empty sized by shape only (the count table is (nchunks, 32) int32 from T). Nothing is cached from inputs.

## Expected

- `*load=first_turn*` faster: _align_parallel_kernel 393-419 us -> ~30-60 us (count + place over 128-256 programs each): about -350 us per shape, ~-3% on the weight-138 sliding ctx32768 row
- `*load=turn_4*` faster: align 71/70 us -> ~10-15 us as measured in d792 (sliding turn_4 +4% there): ~-2.5% sliding, ~-1.5% full
- `*load=turn_1*` faster: align 17 us -> ~9 us (place 7.2 + count ~2 in d792): ~-1%, at the edge of drift; read the align rows, not the headline

## Risk

Low risk. Before the timing gate, every case is checked bit for bit against master's align, and any failure keeps master's align for that class. The ft class is new territory for the align2 kernels (n=131072, 8222 blocks), and the local-saturated case is what proves capacity. Confounder: rega's own ft gate (rega vs Triton at T=16384) now includes the faster align, so its ratio can only improve; this cannot turn rega OFF. Attribution: compare _align_parallel_kernel against _align2_count_kernel + _align2_place_kernel per shape. GEMM rows should stay flat within ~2%. SUPERSEDES the damaged file 2026-10-25T125421Z-align-two-phase-all-t-rega-restack.md: shell backtick substitution blanked its call-site text, so that file must be deleted and must not run.
