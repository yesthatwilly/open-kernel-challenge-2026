# muse_prefill_layer · 2026-10-25T145748Z · submit · align-two-phase-all-t-rega

Hypothesis: `muse_prefill_layer/hypotheses/done/neutral/2026-10-25T125453Z-align-two-phase-all-t-rega.md` (branch `hyp/muse_prefill_layer/align-two-phase-all-t-rega`).

- **Job:** `bff8e36788a04917` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `f92f45e1`

## Goal
Promote check-proven gain align-two-phase-all-t-rega (1.4458x [1.4241,1.4674], check job 4e6f815a8dd945ba) to the board.

Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft master; patch muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.patch) onto master a1a5497d, extended to the first_turn class. WHY NOW: the new master's profile (job 07a069bc69ba42a8) shows _align_parallel_kernel at 17.1/16.7 us (turn_1), 71.2/69.6 us (turn_4) and 392.5-418.9 us at T=16384 (3.2% of every first_turn shape, including the weight-138 sliding ctx32768 row). The ft class (experts-turn-rega-s4-first-turn-n128) moved first_turn from SGLang's moe_align_block_size (about 50 us in the d792 profile) onto _align_parallel, whose 33 programs each stream all 131072 route ids twice with a CHUNK x 32 compare (prepare tracer: 441-454 us eager at T=16384). In d792 the two-phase align measured _align2_place_kernel at 7.2/7.4 us at turn_1 and dropped out of the top-12 rows (< 17.6 us) at turn_4, against 71 us for master; sliding turn_4 moved +4%. CHANGE: (1) Port from the patch with its logic unchanged: the ALIGN2_* constants, _align2_count_kernel, _align2_place_kernel, _align_two_phase, _align2_cases, _align2_first_mismatch, _align2_gate_time and _align2_setup, and the call _align2_setup(state, device) right after _align_par_setup in prepare(). It must come before _rega_setup, so rega's gates and copy/replay checks run on the align that will be timed. (2) Give _align_dispatch a three-way class on T: 'low' T <= 1024, 'high' 1024 < T <= MARLIN_MAX_TOKENS, 'ft' MARLIN_MAX_TOKENS < T <= REGA_FT_MAX, with the key f'{bsm}:{cls}'. (3) Change ONE call site: in _experts_rega (master line ~3158), the call _align_parallel(local_ids, 0, LOCAL_EXPERTS, REGA_BSM) becomes _align_dispatch(local_ids, REGA_BSM, state). _experts_marlin and _experts_ssdq keep master's _align_parallel. They are fallbacks now, and keeping them byte-identical keeps attribution on the rega rows. Leave the diagnostic _align_parallel call in the rega gate code (~line 3295) alone. (4) Set ALIGN2_BSMS = (16,) only, and ALIGN2_GATE = ((16, 753, 'low'), (16, 3012, 'high'), (16, 16384, 'ft')). Add 4500, 8192 and 16384 to ALIGN2_TESTS_T. At T >= 8192, _align2_cases also yields a 'local-saturated' case (every route local: ids % 32), so the tail program's block_experts and sentinel loops cover the ft worst case (8222 blocks, cap 131552). Keep ALIGN2_CHUNKS = (512, 1024): NCH_P2 is then 256 or 128 at T=16384 (count table 32 or 16 KB). The tail program alone fills the sentinel range [total, cap), about 100k int32 at T=16384. If that tail shows more than 10 us in a gate diagnostic, spread the fill over the place programs by slot range instead (the stores stay disjoint), and print which form ran. (5) Add 'align2_ok' to REGA_STATE_KEYS. Its value is a plain dict of ints. The judge prepares once and copies the state by cloning tensors, so no data_ptr or pointer-equality condition may gate anything. Dispatch reads only T and state['align2_ok']. (6) Gate per class. First check equivalence: torch.equal on local_ids, sorted_ids, block_experts and n_padded against _align_parallel for every case. Then time cold-L2 graph medians over ALIGN2_ALTERNATIONS with the empty-graph baseline subtracted. A class turns ON at <= ALIGN2_RATIO 0.98 x master. Print the per-class align2 PATH CHOSEN line again at the very end of prepare(), after the rega PATH CHOSEN lines, so the archived log tail keeps it. kernel_fn rules: no host sync, no prints, no try/except, and per-call torch.empty sized by shape only (the count table is (nchunks, 32) int32 from T). Nothing is cached from inputs.

## What changed, and where
Edited only `.worktrees/muse_prefill_layer/align-two-phase-all-t-rega/muse_prefill_layer/submission.py` (+347/-2). The base is 8517012b, which differs from master 77b2e850 only in this md, so no merge was needed.

- Ported the patch's align2 block (constants, `_align2_count_kernel`, `_align2_place_kernel`, `_align_two_phase`, `_align2_cases`, `_align2_first_mismatch`, `_align2_gate_time`, `_align2_setup`). It sits after `_align_par_ft_setup`. `_align2_setup` runs right after `_align_par_setup`, before `_rega_setup`.
- NOT ported from the patch: its edits to `_experts_marlin` and `_experts_ssdq`. Both keep master's `_align_parallel`, and the rega gate diagnostic call (`_rega_tiles`) is untouched.
- `_align_dispatch(local_ids, bsm, state)` has three classes: 'low' T<=1024, 'high' T<=MARLIN_MAX_TOKENS, 'ft' T<=REGA_FT_MAX. The key is `f'{bsm}:{cls}'`. T<=200 or T>REGA_FT_MAX goes to master. It reads only `local_ids.shape[0]` and `state['align2_ok']`. It has no try, print or .item, and no pointer-equality condition. The count table is `torch.empty((nchunks, 32))`, sized by shape.
- `_experts_rega`'s align call is the only call-site change. `ALIGN2_BSMS=(16,)`. `ALIGN2_GATE=((16,753,'low'),(16,3012,'high'),(16,16384,'ft'))`. `ALIGN2_TESTS_T` gained 4500, 8192 and 16384. `ALIGN2_CHUNKS=(512,1024)`.

Review items:
1. Equivalence suite: at T=16384 it includes 'one-expert' (all routes on expert 7) and 'local-saturated' (`raw % 32`, added from T>=8192). Every case is checked with `torch.equal` on local_ids, sorted_ids, block_experts and n_padded against `_align_parallel`. The suite prints which (kind, chunk, fill form) passed at T=16384 for those two kinds.
2. `'align2_ok'` is in `REGA_STATE_KEYS`. Its value is a plain dict of ints: the "bsm:class" chunk sizes plus key "dist" (0 or 1).
3. Prints: per-class gate lines with the master median, each chunk's median, the chunk chosen, the ratio and the fill form. Prepare also prints the register count and spills of the place kernel at CHUNK 1024 (from the Triton launch handle's `n_regs` and `n_spills`; it prints "n/a" if the Triton version does not expose them). `_align2_setup` returns its gate lines and PATH CHOSEN line, and `prepare()` prints them again after `_rega_setup`.
4. Tail program: a diagnostic (count+tail minus count-only graph medians at T=16384, using the place kernel's `TAIL_ONLY` constexpr) measures it. Above 10 us the code switches to the distributed fill, which is a `DIST` constexpr. Each place program fills the sentinel slots of its own range `[pid*span, (pid+1)*span)`, and the tail program skips the fill. The `dist` form has its own equivalence pass on all cases with T>=8192, which includes one-expert and local-saturated at 16384. If `dist` fails, the tail form is kept and the line says so. The chosen form is printed.
5. kernel_fn: as in the dispatch bullet above. The state is cloned by the judge, and nothing is gated on pointers.
6. Open question about align_par_ok: `_align2_setup` returns OFF unless `align_par_ok`. The log shows that flag in the `[align-local]` line.

Not run: no GPU here, so the Triton kernels are uncompiled. The `dist` loop uses a runtime-start `range(lo, hi, SENT)`.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path adds only torch.empty (shape-sized) and two Triton launches. The diff has no .item/.cpu/synchronize outside prepare.
- [x] Nothing derived from `data` is cached across calls: the count table is allocated per call, and state holds only prepare-time ints.
- [x] Weights untouched (the diff touches only the align).
- [x] Activations/KV/queries: n/a, the align works on int32 ids only.
- [x] Attention sees every key: n/a, no attention change.
- [x] No timing patches, harness frames, I/O, module edits or reference import. No subprocess or process spawn.
- [x] All work inside the timed call: the align runs in `_experts_rega` as before. Outputs are bit-identical to master's (torch.equal in prepare).
- [x] Judge state copy: no module/static globals hold tensors; state['align2_ok'] is a dict of ints inside REGA_STATE_KEYS.

okc_lint: `OK with warnings: muse_prefill_layer/submission.py` (only host-sync warnings in prepare code and the known dtype warnings). undef.py: `[]`. py_compile: OK. __pycache__ deleted.

## Risks / what could fail
low

## Result (after the run)
Promotion passed: 1.4480x [1.4263,1.4698], board: {"rank": 142, "team": "yesthatwilly", "speedup": 1.448, "first": false, "ahead": 141, "to_first": 1.3299}
