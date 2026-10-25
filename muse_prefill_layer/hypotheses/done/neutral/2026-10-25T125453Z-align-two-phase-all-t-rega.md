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
review: proceed
status: done
review_at: 2026-10-25T125918Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T125919Z
base_sha: 8517012b49a52a4bad4cd73abb2b38533a9adde2
base_blob: c2e3e595def2e6b5949e0fd350ce98ed59729147
branch: hyp/muse_prefill_layer/align-two-phase-all-t-rega
worktree: .worktrees/muse_prefill_layer/align-two-phase-all-t-rega
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T130218Z
impl_commit: 52b7bc8a5aa03656df66add7466ef473284843ef
submission_sha256: d14e76b6b72de0f2a6866e964caa63eabba5c2b068bb500527fa0cb16742d563
norm_sha256: 314fa6c25eec5de828251a2da8374d11abd960d6e9aca6fa820ea8e07657cce7
submission_blob: a2fe2a75893c60d4f68cf94fcce8ce8f9b708a71
lint_warnings: 65
attempt_file: 2026-10-25T130218Z-check-align-two-phase-all-t-rega.md
prepared_at: 2026-10-25T130218Z
job: 4e6f815a8dd945ba
dispatched_at: 2026-10-25T130234Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4457544543391616
speedup_ci: [1.4240681375240742, 1.467440771154249]
harvested_at: 2026-10-25T135010Z
best_at_harvest: 1.4060131381907617
promotion: merged
hand_landed: true
promotion_commit: 1932a9c4fb3d8dc735bb1a513853637c0ada892c
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

## Adversarial review

Reviewed 2026-10-25T125918Z by an adversarial subagent (model opus).

**Verdict: proceed.** Isolated, bit-exact replacement of a serial ~393-419 us align at first_turn (verified in 07a069 profile) plus 17/71 us at turn_1/turn_4 on the rega path; first_turn carries 186 of 310 shape weight, so a ~2-3% headline gain is plausible and attribution is readable straight from the align kernel rows. No companion is worth one of the 2 remaining checks.

### Justification
Profile 07a069bc69ba42a8 confirms _align_parallel_kernel at 17.1/16.7 us (turn_1), 71.2/69.6 us (turn_4), 393.9/392.5/396.1/418.9 us (first_turn, 2.1-3.3%), and the prepare log shows 441-454 us eager at T=16384. The mechanism is structural: 33 programs each stream all n=131072 ids twice with a [512x32] compare, so time scales with n on 33 SMs; the two-phase form reads each id ~twice in total over 128-256 programs per launch. The align sits serially between the router join and the rega gather/GEMMs (sorted_ids/block_experts feed them), so saved us become wall-time. Outputs are torch.equal to master's align, so rega GEMM rows should not move. Directions match TASK weights: first_turn ~-350 us/shape on 11.7-19.2 ms is ~2-3% on 186/310 of the weight, against a noise of ~1.5%. That is likely, but not certain, to clear the CI. Weakest links: (1) the ft-class cost of align2 is unmeasured. At T=16384 the place kernel holds a [NCH_P2=128-256, 32] count table plus a [CHUNK, 32] one-hot cumsum per program (128 regs/thread of one-hot at CHUNK 1024, so possible spills), and one tail CTA writes ~115k sentinels plus 33 [256x32] block_experts iterations. My estimate is 15-40 us, not the 30-60 us stated, so the risk is low but real. (2) The turn_1/turn_4 expectations rest on d792, where only sliding turn_4 moved (+4%, which is more than a 55 us saving can explain, so partly noise), turn_1 was flat, and the headline 1.3369 was within CI of its 1.3275 base. Read the turn rows as align-kernel rows, not headline deltas.

### Confounders on the current kernel
- experts-turn-rega-s4-first-turn-n128 (parent): created the ft call site that put _align_parallel on T=16384. Its ft gate (rega 3520 vs Triton 5584 us, ratio 0.630) now includes the faster align, so it can only move further ON. The margin rules out a flip, so no rega-vs-Triton confound. If rega ft were OFF, the ft class is dead code (the Triton path keeps moe_align_block_size), and the ft PATH CHOSEN line would show it.
- experts-turn-rega-s4-port-t4096: owns the low/high call site (_experts_rega). Gates at 0.649/0.651 and bit-identical align outputs mean rega's GEMM rows cannot shift through the change. Only the launch count changes (1 to 2 kernels), worth ~2-3 us of launch/tail overlap.
- stack-tonight-five-prefill (carries moe-align-local-routes-parallel-port): defines _align_parallel, which is both the equivalence reference and the incumbent timed in the gate, and _align2_setup returns OFF unless align_par_ok. That is a precondition, not a confound. Check that align_par_ok is ON in the log.
- experts-turn4-ssdq-s1b-dataptr-symbol-fix, experts-hybrid-marlin-turn-triton-first-turn, sglang-pipeline-parity, experts-triton-w4a16-first-turn-blockt256: the ssdq and Marlin fallbacks keep _align_parallel byte-identical and are not the timed path at turn_1/turn_4 (rega wins those gates), so they do not interact.

### Interference and A/B needs
No companion. The change is one call site on one path, with outputs proven bit-identical in prepare, so every downstream kernel row should stay flat. The per-shape profile gives _align_parallel_kernel (master, 17/71/393-419 us) against _align2_count_kernel + _align2_place_kernel directly, and the re-printed per-class PATH CHOSEN line says which classes ran. A gain or a loss is therefore attributable from this one run. An ab-revert of the parent would remove the ft call site and with it the main expected gain. An ab-tune of chunk size is already covered by the in-prepare gate choosing between 512 and 1024. A control isolating the ft class from low/high would only split ~2-3% into finer pieces already visible per load row. With 2 prefill checks left today, spending one on a companion buys nothing the profile does not already show.

### Open questions
- Which chunk size wins the ft gate, and do the place kernel's regs or spills at CHUNK 1024 with NCH_P2 128 matter? Print the per-class medians next to master's ~393 us.
- Does the single tail program (≈115k sentinel stores + 8222-entry block_experts at T=16384) stay under 10 us? If it falls back to the distributed fill, that form needs its own equivalence pass on the local-saturated and one-expert cases at T=16384.
- d792's log tail lost the align2 lines. Confirm in the new log that the 16:low, 16:high and 16:ft classes are each ON, so that flat turn rows mean 'ran and did not help' and not 'never ran'.
- Does the ft equivalence suite at T=16384 include 'one-expert' (all 131072 routes on expert 7: 8192 blocks for one expert) as well as 'local-saturated' (cap fully used, empty sentinel range)? Both extremes should be required.


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn path adds only torch.empty (shape-sized) and two Triton launches. The diff has no .item/.cpu/synchronize outside prepare.
- [x] Nothing derived from `data` is cached across calls: the count table is allocated per call, and state holds only prepare-time ints.
- [x] Weights untouched (the diff touches only the align).
- [x] Activations/KV/queries: n/a, the align works on int32 ids only.
- [x] Attention sees every key: n/a, no attention change.
- [x] No timing patches, harness frames, I/O, module edits or reference import. No subprocess or process spawn.
- [x] All work inside the timed call: the align runs in `_experts_rega` as before. Outputs are bit-identical to master's (torch.equal in prepare).
- [x] Judge state copy: no module/static globals hold tensors; state['align2_ok'] is a dict of ints inside REGA_STATE_KEYS.

okc_lint: `OK with warnings: muse_prefill_layer/submission.py` (only host-sync warnings in prepare code and the known dtype warnings). undef.py: `[]`. py_compile: OK. __pycache__ deleted.

## Result

- job `4e6f815a8dd945ba` (check), status done, passed True, tests 5/5
- speedup 1.4458x, 95% CI [1.4241, 1.4674], score 267959.6793475404 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4060131381907617, "ci": [1.3849229411179003, 1.4271033352636229], "job": "07a069bc69ba42a8"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-first-turn-n128.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.result.json`, `muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.patch`; branch `hyp/muse_prefill_layer/align-two-phase-all-t-rega`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 777.02 | 704.38 | 1.0629 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1136.06 | 1097.89 | 1.015 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2251.37 | 2098.36 | 1.0374 | gained |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3785.25 | 3644.6 | 1.0144 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 12148.27 | 12078.49 | 1.0345 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 11658.83 | 11285.6 | 1.0115 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15589.38 | 15310.56 | 1.0135 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 19167.43 | 18819.51 | 0.9914 | flat |

gained on sliding turn_1, turn_4; full first_turn. Expected hit: 3/3. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-two-phase-all-t-rega`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

