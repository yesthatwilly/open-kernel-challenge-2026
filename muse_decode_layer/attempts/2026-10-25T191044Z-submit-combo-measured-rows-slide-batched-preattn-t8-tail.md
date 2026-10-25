# muse_decode_layer · 2026-10-25T191044Z · submit · combo-measured-rows-slide-batched-preattn-t8-tail

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T161651Z-combo-measured-rows-slide-batched-preattn-t8-tail.md` (branch `hyp/muse_decode_layer/combo-measured-rows-slide-batched-preattn-t8-tail`).

- **Job:** `a903b9d004d2452d` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `b257ef2b`

## Goal
COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached master, each routed only to the shape class where it won, on master 6773ec5cf5 (dense-chain-s3 + dl3 piece A, 1.3080 job 933d3f825a044d36). No new load_inline module (all three sources are Triton or already-built dense-chain kernels). SOURCE S1 sliding-fused-batched-decode-port (branch hyp/muse_decode_layer/sliding-fused-batched-decode-port, impl 3ddcc4f2a8, job b01fd2df41a040b3) in the rebased form of sliding-batched-triton-b16-band (branch hyp/muse_decode_layer/sliding-batched-triton-b16-band, impl 1cabea713e, job 02bf417257e14a3d): port _sliding_b16_gate and _sliding_forward_b16 (the existing _sliding_fused_kernel launched with grid (batch, hkv, 16), S=16 hard-coded, BLOCK_N 64, num_warps 4, num_stages 2) and widen the band to two classes: 'sliding q_len 1 page 1 2 <= B <= 16' and '17 <= B <= 32'. Measured: sliding decode_16 +3.8% / +5.5% / +5.2% and sliding decode_32 +3.1% / +4.1% / +4.8% against their own bases (b01fd2df, 2193fc0d, b94e7973), still +1.7..+4.1% / +2.6..+4.3% today by paired SGLang ratio vs master 933d; the incumbent on both rows is unchanged since (FA3 13.6/18.2 + combine 4.3/2.9 + prepare_varlen 2.3/2.2 + FusedQKNorm, casts, store_cache, FusedRMSNormGate). Route: in _parent_attention after the batch-1 branch, 'elif state flag slide_b16 and class A' and 'elif slide_b32 and class B' -> _sliding_forward_b16 with state['slide_cnt']. Check: o rel-L2 <= 1e-2 vs master's chain and cache bytes vs master (k within one E4M3 code on < 6% of bytes, v exact) at B in {2, 9, 16, 17, 32}, lens {1, 63, 2047, 2048, 2049, 80000}, kv_scale {1, 0.015625}. Selector per class at B=16 and B=32: the whole attention side (qkv -> o-GEMM input). SOURCE S2 stack-route-topk-pdl-unpool-nf1 (branch hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1, impl 2fe4d35e13, job 87211e1362b64c4c): its two T>1 pieces, which master's dl3 B/C gate to tokens == 1: (S2a) routing R variant 0 (per-warp top-8, PDL after P) at 2 <= T <= 32 instead of stage-1 R (dl_route_norm_align_v1_kernel, 7.3-7.9 us in every T=8/16/32 row of 933d); its own selector chose it on the sum of P->R pairs at T=1/8/32 (87.3 vs 93.0 us); (S2b) dl_unpool_kernel NF=1 at 2 <= T <= 8 (= the queued dl3-unpool-nf1-t2-t8 widening) instead of _gemm(y, unpooler[:1024]) + splitKreduce + F.pad. Measured: sliding verify_1 +4.3% vs its base and +2.9% / +3.1% (sliding/full verify_1) today vs 933d. Both kernels already live in master's muse_decode_dense_chain_s3 module; the change is gates + per-state flags dl3['b_t'] = {8, 16, 32} and dl3['u8'], extending _dl3_check_r / _dl3_check_u to T in {2, 5, 8, 16, 32} (bit-identical ids and align, near-tie rule as master's; unpool rel-L2 <= 2e-3 and exact zero columns) and adding selectors _dl3_select_r at T=8/16/32 (P->R pair) and _dl3_select_u at T=8, plus the T=8 eager layer check and the T=8 two-state alternation through _dl3_layer_checks. SOURCE S3 fused-preattention-paired-pad-restack (branch hyp/muse_decode_layer/fused-preattention-paired-pad-restack, impl 963bcf0267, job 5e9b5464d1444708): port _pre_attention_kernel (PAIRED and unpaired) and _paired_norm_gate_kernel. It shipped with NO gate (every T > 1); here route it per class only where the attention side still runs FusedQKNorm + k/v/q FP8 casts + store_cache after S1: (c1) full decode 1 < B <= 64 (feeds master's sk single-level ticket kernel its FP8 q; master's split-KV A/B fallback too), (c2) sliding verify T=8/128/256 (unpaired FA3 feed), (c3) full verify B > 1 (PAIRED feed into paired FA3 + _paired_norm_gate_kernel), (c4) full verify_1 (unpaired). Measured: full decode_16 +4.7%, full decode_32 +3.3%, sliding verify_1 +3.7% against its own base. Check: q (bf16 before the cast) rel-L2 <= 1e-3 vs master, cache k bytes by rel-L2 vs an accurate-rope fp32 reference (LESSONS: Triton sin/cos), v exact, paired halves exactly zero, kv_scale {1, 0.015625}, T in {2, 8, 16, 32, 40, 128, 256}. Selector per class at T in {16, 32} (c1), {8, 128, 256} (c2), {128, 256} (c3), 8 (c4) on the attention side. Every other shape, and every cell whose gate fails, is master byte for byte (sliding decode_1, full decode_1 piece A, T>32 MoE, all GEMMs). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, counter/ticket array, workspace and TMA map lives in the state, never in a module global (this includes the Triton sliding kernel's split counters: master's module-global _COUNTERS dict must NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per state and slice it). Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed load_inline module built at import like master's, with an in-process canary (synchronize + 'launching'/'ok' prints, flush=True) after each first launch; on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line keeps master for that cell. GATE PER SOURCE PER CLASS: each source path has its own prepare-time gate = (a) correctness check (bound stated per source below) on two alternating states with refilled inputs under captured-graph replay, then (b) a cold-L2 CUDA-graph selector (256 MB flush before each replay, >= 41 interleaved reps, empty-graph baseline subtracted) that enables the cell only when new/master <= 0.98 (i.e. master/new >= 1.02) and asserts the enabled flag equals the printed winner. Print exactly one 'PATH CHOSEN: <source> <class>: <winner> [new X us vs master Y us]' line per source per class. Every added prepare() phase runs AFTER master's own setup (after _dl3_finish), each with its own start deadline measured from prepare()'s t_prepare0 (skip the phase with a fallback line if the deadline has passed) and a print of its own wall seconds plus the cumulative prepare wall time; master's own internal deadlines and its _sk_prepare budget do not move. Print compile seconds per load_inline module. First measurement is an okc check.

## What changed, and where
Built 2026-10-25 on base 0c4719336e (master's muse_decode_layer/submission.py is byte-identical at c3f81624, so no merge was needed). Only `muse_decode_layer/submission.py` changed: +1390 / -166 lines. The diff adds no new load_inline module.

**Layout for the sibling combos C2 and C3.** Each source has one named prepare phase and one per-state flag:

| Source | Phase | Flag |
|---|---|---|
| S1 | `_cb_prepare_s1` | `state["s1"][A or B]` |
| S2a | `_cb_prepare_s2a` | `state["dl3"]["b_t"][8, 16 or 32]` |
| S2b | `_cb_prepare_s2b` | `state["dl3"]["u8"]` |
| S3 | `_cb_prepare_s3` | `state["s3"][cell]`, with cells c1-b16, c1-b32, c2-t8, c2-t128, c2-t256, c3-t128, c3-t256, c4-t8 |

After those phases, `_cb_prepare_t8` runs the T=8 layer check. `_combo_prepare` calls all of them after `_dl3_finish`, and each phase goes through `_cb_phase`. `_cb_phase` checks the start deadline measured from t_prepare0 (S2a and S2b 120 s, S1 150 s, S3 180 s, T8 210 s; no new cell starts after 255 s). It catches exceptions, prints the phase's wall seconds and the cumulative prepare time, and applies the flags. `_cb_report` then prints one `PATH CHOSEN: <source> <class>: NEW|MASTER ... [new X us vs master Y us]` line for each of the 14 source x class cells. It checks that each flag equals the printed winner and forces master on a mismatch. Master's T>1 attention side is now `_attn_core`: `_master_front` + `_attn_route` + FA3, sk or split-KV. With every flag off it issues master's launches in master's order. `_parent_attention` is `_gemm(_attn_core(...), o)`.

**kernel_fn** reads only config and this state's flags. `_attn_core` routes S1 (`_sliding_forward_band`, S=16, BLOCK_N 64, 4 warps, 2 stages) and S3 (`_s3_pre_attention`, plus `_s3_paired_norm_gate` in c3). kernel_fn itself passes `r_t` to `_dl_forward` for S2a and takes `dl3["u8"]` into the unpool branch for S2b. kernel_fn has no prints, no try/except and no per-call mutation.

**Selectors (all cells).** `_cb_time` captures each path as a graph, flushes 256 MB before every replay, runs 41 interleaved rounds (order reversed every round, after 3 warm rounds) and subtracts the median of a one-kernel graph. A cell is enabled only when new/master <= 0.98. The plain numbers from the first prepare() are reused by the second one (`_CB_SELECT`). Each state still runs its own correctness gates.

**Correctness gates.** Every source x class cell gets two-state alternating graph replay with refilled inputs and poisoned outputs. `_cb_second_state` gives the second state its own slide_cnt, P/R, Q/O/unpool and sk buffers.
- S1 and S3 (`_cb_attn_alt`) compare the attention side (qkv to o-GEMM input) against master run eagerly on a clone, and against an fp64 reference (`_cb_attn_ref`).
- S2a (`_dl3_check_r` at T in {2, 5, 8, 16, 32} plus `_cb_s2_alt` "R") requires all seven R outputs to be bit-identical to stage 1's R. That includes sorted_ids, block_experts and n_padded at T=32 (bsm 16), which answers the review's open question.
- S2b (`_dl3_check_u` at T in {2, 5, 8} plus `_cb_s2_alt` "U") requires rel-L2 <= 2e-3 and exact zero columns.
- S3 also runs a kernel-level `_s3_front_check` on q, k and v, described under review item (4).

### Review fixes applied
1. **S1 is ported only in the sliding-batched-triton-b16-band form (1cabea713e).** `_sliding_b16_gate` became `_s1_class` and `_sliding_forward_b16` became `_sliding_forward_band`, with the band widened to classes A (2..16) and B (17..32). Nothing from sliding-fused-batched-decode-port was taken: no try/except and no `_SLIDING_FUSED_ON`.
2. **Counters and `_INV_FREQ` moved into the state.** `state["slide_cnt"]` is `torch.zeros(128, int32)` allocated in prepare() through `_slide_state`. The band uses [0:B*Hkv] and master's batch-1 `_sliding_forward` now uses its own slice [64:66]. `state["slide_inv_freq"]` replaces `_INV_FREQ`, and `_COUNTERS`, `_counters` and `_inv_freq` are deleted. `_dl3_ministate` also gives the second state its own slide_cnt.
3. **Counter assertion.** `_cb_clean` runs after every check, alternation, selector and phase. If any counter (slide_cnt, cnt_p/r, cnt_q/o/u, both sk ticket levels) is not 0, it zeroes them and sets `recs["_dirty"]`, which keeps master for every combo cell of that state.
4. **S3 RoPE check by rel-L2.** `_s3_front_check` covers position bases 80000, 299995 and 1048500 on the sliding cells, and 1048500 on c4. It compares q and the new k rows by rel-L2 against an fp64 accurate-RoPE reference (`_cb_rope64`) and requires new <= master + 1e-3 and at most one E4M3 code gap from master. It also requires v exact, the paired zero half exactly 0 and every other row untouched.
5. **FULL-layer T=8 check.** `_dl3_layer_checks(..., t8=True)` calls `_cb_t8_layer_check`, which runs kernel_fn at B=1, q_len 8 on three cases: full ctx 80000, full ctx 1048570, and sliding ctx 80000 (the sliding companion the review asked for). Each case runs two states with one graph each and 4 alternating replays with refilled inputs and poisoned outputs.
   - The judge's routing rules are applied against master's fp64 router scores (`_cb_route_rules`): pick within 0.05 of the 8th, clear-margin (> 0.03) local top-8 present, routes >= 92% - 4, distinct >= 85% - 2.
   - It prints the max error per output: hidden, stream, update, k_cache, k_gap and token ids.
   - On a miss, S2a class 8, S2b, S3 c4-t8 and S3 c2-t8 go off for that state. If the check cannot run (deadline or exception), those cells also go off.
6. **Every source x class has its own gate, selector and line.** That is 14 cells, each with its own correctness gate, its own selector at <= 0.98 and one PATH CHOSEN line.

### Answers to the open questions
- **Is class A kept only because the selector is cheap?** Yes. It is gated, and the expected verdict is MASTER, which keeps Sd16 flat.
- **Deepcopy vs a second prepare() call:** this works either way, because every buffer is per state and no module global holds a tensor.
- **Deadlines** are per prepare() call, measured from that call's own t_prepare0.

### Deviations
- **S1/S3 attention-side bound.** The bound is "o rel-L2 <= 1e-2 vs master OR new vs fp64 <= 1.25 x master vs fp64 + 1e-3". The spec says only 1e-2 vs master. Master's FP8-q error alone can exceed 1e-2 on peaked synthetic softmax (LESSONS: calibrate bounds against fp32 and master's own error). Cache bounds follow the spec: k within 1 code on < 6% of bytes, v exact, other rows identical.
- **S3 q check.** The spec asks for q (bf16 before the cast) rel-L2 <= 1e-3 vs master, but the S3 kernel never produces a bf16 q. The check instead compares the FP8 q against the fp64 reference (new <= master + 1e-3) and requires at most one code gap from master.
- **S3 classes as cells.** S3 has 8 cells, one per selector point the spec lists: c1 {16, 32}, c2 {8, 128, 256}, c3 {128, 256}, c4 8. Batch splits at 23/24, as the sk batch classes do. Verify cells are limited to 2 <= q_len <= 8, B <= 64, and include a (B=5, L=3) check. c1 runs only when this state feeds sk or split-KV (unpaired q); otherwise that cell stays on master.
- **Synthetic keys** have std 0.3 (soft) and are clamped to stay inside E4M3.
- **T=8 layer tolerances.** With identical ids: hidden <= 2e-2, stream <= 1e-2, update <= 5e-2. If ids differ but the routing rules pass: 5e-2 / 2.5e-2 / 1e-1. Cache: k <= 1 code and rel <= 2e-2, v exact.

### Biggest risk
Prepare wall time, roughly 60-90 s per call and doubled across two calls:
- Graph captures at ctx 80000, B=32, plus data builds of about 650 MB.
- Full-layer verify at ctx 1048570.
- About 50 S3 front checks and fp64 references.

The per-phase deadlines bound it. First measurement must be an okc check.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable. kernel_fn, `_attn_core`, `_sliding_forward_band`, `_s3_pre_attention` and `_s3_paired_norm_gate` have no .item(), .cpu(), synchronize, prints or try/except (grep of the diff). Allocations are sized by config only. Branches use config plus this state's prepare() flags only.
- [x] Nothing derived from `data` is cached. The state holds only prepare()-built buffers and flags. `_CB_SELECT` holds plain floats and bools from prepare() timing on synthetic data.
- [x] Weights are untouched (no new repack). NVFP4 and BF16 are as in master.
- [x] Activations: BF16 as master. KV FP8 (S1 and S3 store E4M3). q is E4M3 (S3) or bf16 (S1), never coarser than E4M3.
- [x] Every key is attended. S1 covers the full 2048-token window with 16 splits; S3 changes only the front half before FA3, sk or split-KV.
- [x] No timing patches, harness access, module edits, reference import, processes or network. There is no subprocess, multiprocessing or os.fork anywhere.
- [x] All work is inside the timed call, with no output caching. prepare() checks and selectors use synthetic data only.
- [x] Two-state rotation is safe: there is no module global with a device pointer. `_COUNTERS` and `_INV_FREQ` were removed and slide_cnt is per state.
- [x] n/a: there are no new load_inline, sm_90a or wgmma builds (no new CUDA module); Triton launches are announced and synced in prepare() (`_cb_launch`).

```
python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer
-> OK with warnings (71 host-sync warnings, every one "outside kernel_fn's call graph; fine in prepare()"; 0 errors)
python3 -I -m py_compile -> ok; symtable scan for undefined globals -> none; __pycache__ removed
```

## Risks / what could fail
medium
Winner map behind this stack (paired SGLang ratio vs master 933d, which re-bases each source to today): Sd16 S1 +1.7..+4.1; Sd32 S1 +2.6..+4.3; Sv1 S2 +2.9, S3 +2.3; Fv1 S2 +3.1, S3 +3.1; Fd16/Fd32 S3 +4.7/+3.3 on its own base (today's paired ratio is from a pre-sk base, so only the own-base delta transfers); every other row master. Expected headline about +1.5..+2% (decode_16 ~3.5%, decode_32 ~2.7%, verify_1 ~3.5% of their loads). Prepare phases (all after _dl3_finish; master's own prepare is 4.5 s in 933d): S2 checks + selectors + T=8 layer checks ~4 s (start deadline 120 s); S1 Triton re-specialisation, checks, 2 selectors ~10 s (deadline 150 s); S3 Triton JIT for paired/unpaired, checks, 8 selector cells ~25 s (deadline 180 s); total ~45 s, no import-time compile added (still 3 modules, ~120 s at import).

## Result (after the run)
- job `a903b9d004d2452d` (submit), status done, passed True, tests 13/13
- speedup 1.3142x, 95% CI [1.2945, 1.3339], score 17211.74728083805 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.3079569800910293, "ci": [1.288337625389664, 1.3275763347923946], "job": "933d3f825a044d36"}; base result: muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.result.json`, `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.patch`; branch `hyp/muse_decode_layer/combo-measured-rows-slide-batched-preattn-t8-tail`
