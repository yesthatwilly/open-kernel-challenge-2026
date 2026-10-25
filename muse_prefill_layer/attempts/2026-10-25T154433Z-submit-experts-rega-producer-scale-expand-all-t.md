# muse_prefill_layer · 2026-10-25T154433Z · submit · experts-rega-producer-scale-expand-all-t

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T152138Z-experts-rega-producer-scale-expand-all-t.md` (branch `hyp/muse_prefill_layer/experts-rega-producer-scale-expand-all-t`).

- **Job:** `6be2a8d3cf7d489f` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `a7f4c3df`

## Goal
Built on master (1932a9c4). Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~42 per expert, buckets 32/64) streams w13's 604 MB in 283 us = 2.1 TB/s and w2's 302 MB in 144 us = 2.1 TB/s (44% of HBM), so at small N the per-stage dequant issue dominates (one stage = 2 x 8 m64n32k16 = 256 clk of wgmma vs ~180 dequant instructions per consumer warp, 2 warps per SMSP); the two-CTAs-per-SM run (LESSONS) already showed latency/issue hiding pays 9% there. In dq_stage every consumer lane spends ~7 of ~24 instructions per k16 step on scales: a uint2 load, cvt.rn.f16x2.e4m3x2 x2, then per k16 step __half22float2, two fmul by 2^119 and two __float2bfloat162_rn. Producer warps 1-3 (96 threads, setmaxnreg.dec 40) are idle. CHANGE (port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-producer-scale-expand to prefill's rega): (1) a scale buffer per ring stage in dynamic smem, 2 records x 1024 B = 2048 B per stage, 8192 B for 4 stages, placed after the epilogue staging tile (Smem<false>::BYTES 203776 -> 211968; update the static_assert to <= 227 KB with the 16384 static bound: 228352 <= 232448). (2) STAGES more mbarriers sfull[s] (static), count 3. Producer warps 1-3 walk the same item / k-stage sequence as tid 0 (same decode_item, same stage counter n and parity): for each fill they mbar_wait(full[s], parity), then convert the stage's 2 x 512 scale bytes (st + r*REC + REC_CODES) into bf16 in the SAME byte order: sx[s][r*512 + i] = bf16(float(e4m3 byte i) * 2^119), computed exactly as master's consumers do (cvt.rn.f16x2.e4m3x2 -> __half2float -> * 2^119 -> __float2bfloat16_rn; exact, since e4m3 has 3 mantissa bits and 448 * 2^119 < FLT_MAX), 256 32-bit scale words spread over the 96 threads, then __syncwarp and lane 0 arrives on sfull[s]. (3) In the new consumer, wait full[s] then sfull[s] (same parity), and in dq_stage replace the uint2 scale load + e4m3x4_to_f16x2x2 + per-k convert with ONE uint4 load per record at sx[s] + r*1024 + cw*128 + g*16 (the 8 bf16 = rows g / g+8 for k16 steps 0-3, the same positions master's uint2 at cw*64 + g*8 holds); per k16 step sg = prmt(w_k, w_k, 0x1010) (row g) and sg8 = prmt(w_k, w_k, 0x3232) (row g+8): 2 instructions instead of ~7. dequant8, the wgmma sequence, wgmma_wait<0>, the empty arrive, epilogues, producer copies, ring depth, buckets, chunking, gather and the weighted sum are unchanged. Reuse is safe: helpers overwrite sx[s] for fill f only after full[s] phase f, which the producer issued only after the consumers' empty[s] arrives for fill f-1, which follow their dq_stage reads of sx[s]. The dead/timeout path must still arrive on sfull. Expose it as rega_kernel_sx<G, BIG> (4 instantiations) behind a host entry rega_gemm_sx with rega_gemm's signature; _experts_rega uses it for a class when state['rega_sx'][class] is True (prepare flag per class 'low' / 'high' / 'ft', default False). Outputs are bit-identical to master's rega. BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY strings (the v1 build-failure fallback derives its source from them by text substitution and checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a guards; if that build or its import fails, build master's v2 source under its master name exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants. PREPARE GATE (only for classes whose rega_ok is ON): (1) exactness on master's self-check routings at every class size (753 / 1024 / 3012 / 4096 random, skewed, one; 4500 local; 8192 random, local; 16384 random, skewed, one, local): max |SX - master| == 0 on hs and y, error word 0, launch markers; (2) master's _rega_replay_check copy plan with SX on; (3) cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, SX block vs master's rega block at each class gate T; SX ON for a class only if <= 0.98x. Print per class the w13 and w2 kernels alone (graph of one launch) for both variants with TB/s (weight bytes) and TFLOP/s, then one '[rega-sx] PATH CHOSEN <class>: sx|master' line per class as the last lines of prepare(). Any failure prints 'OFF: <reason>' and master's rega runs unchanged. RULES: kernel_fn stays CUDA-graph capturable with no host sync, no allocation sized by input values, no data-dependent host control flow, no prints, no try/except, no timing and no per-call mutable Python state; it dispatches only on the token count and on flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert weights stay NVFP4/E4M3 bytes in global memory and are dequantized in registers (any bf16 scale values live only in per-stage shared memory inside one launch); activations BF16, fp32 accumulation; every route is computed. No child processes (in-process load_inline only). The judge prepares TWO state copies: no module, static or C++ global may hold device pointers, TMA maps or buffers; keep master's per-call tensor-map encoding (__grid_constant__ Maps) and per-state bank / error word.

## What changed, and where
Built 2026-10-25 in the worktree, base 1b7b960d (its submission.py blob equals master 47eaa27e's, so no merge was needed). Edits: worktree `muse_prefill_layer/submission.py` only, +924/-1. Not compiled (no GPU or nvcc on this node). The new CUDA passed a g++ `-fsyntax-only` check against CUDA/torch stubs, covering all 4 `rega_kernel_sx` instantiations and `consume_item_sx` at N 16/32/64/128. A static undefined-name check of every new or edited Python function was clean.

**Profile anchor (review fix 5).** All numbers are from the current best (bff8e367 / FINDINGS):
- turn_1 sliding: w13 259.3 us, w2 122.1 us
- turn_1 full: w13 226.4 us
- turn_4 sliding: 710.0 / 343.6 us
- first_turn sliding (weight 138): 3734.0 / 1847.8 us

The spec's 283/144 were older numbers. The review expects about 5-10% on w13/w2 at turn_1 if the kernel is latency-bound, a few % if it is issue-bound, and 1-3% at turn_4 and ft. Read each class's KERNELS line against these numbers.

**What changed.**
1. **New CUDA module `muse_prefill_rega_se`** (review fix 1; the caller named it, replacing the spec's single-module muse_prefill_rega_v3).
   - Source: `_REGA_HEAD` + `_rega_wgmma_src()` + the namespace part of `_REGA_BODY` (split at its unique `}  // namespace`; checked to occur exactly once) + `_RSE_BODY`. Master's templates (`rega_kernel`, `consume_item`, `info_one`) are never instantiated there, so the module compiles only the 4 new instantiations, plus master's tiny gather kernel.
   - Host entries `rega_se_init` / `rega_se_gemm` have `rega_gemm`'s signature and checks. Tensor maps are encoded per call into a stack-local `__grid_constant__ Maps`. There are no globals; `_rega_audit_cpp` on the new source finds none.
   - Built with master's exact `REGA_CUDA_CFLAGS` (-O3, -std=c++20, -lineinfo, sm_90a gencode), with `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored after. The arch flags and nvcc seconds are printed.
2. **`rega_kernel_sx<G, BIG>`.** This is `rega_kernel` plus the following:
   - **Smem.** `sx` = STAGES x 2048 B after the epilogue tile, so `SmemSx<false>` = 203776 + 8192 = 211968 B. The static_assert `SmemSx<false>::BYTES + 16384 <= 227 KB` gives 228352 <= 232448.
   - **Barriers.** STAGES more mbarriers, `sfull[s]`, count 3.
   - **Producer warps 1-3** (96 threads, still under `setmaxnreg.dec 40`) walk the same item / k-stage sequence as tid 0. Every item has KS stages, so no `decode_item` is needed. Per fill they `mbar_wait(full[s])` and convert the stage's 256 scale words (thread ht takes words ht, ht+96, ht+192 < 256). The conversion is the exact op sequence of `dq_stage`: `cvt.rn.f16x2.e4m3x2`, `__half22float2`, x 2^119, rn to bf16, written as `sx[s][r*512 + i]` in byte order. Then each warp does `__syncwarp` and lane 0 arrives on `sfull[s]`, unconditionally.
   - **Consumers** wait `full[s]`, then `sfull[s]` (same parity). `dq_stage_sx` loads one uint4 at `sx + r*1024 + cw*128 + g*16` and uses `__byte_perm(w, w, 0x1010)` for row g and `0x3232` for row g+8.
   - Everything else is master's text verbatim: `dequant8`, the wgmma sequence, `wait<0>`, the empty arrive, the epilogue, the producer, tables and buckets.
   - **Launch marker** is `0x53584d4d` (master's is `REGA_MAGIC`), so every check proves which kernel ran.
3. **Python.**
   - `_experts_rega_sx`: `_experts_rega` with both GEMMs on `rega_se_gemm`.
   - `_experts` dispatches a class to it only when `state['rega_ok'][cls]` and `state['rega_sx'][cls]` are both set (shape plus prepare flags).
   - `prepare()` calls `_rega_sx_setup` AFTER `_rega_setup` returns and prints its summary (KERNELS / GATE lines, `OFF <cls>: reason`, `[rega-sx] PATH CHOSEN <cls>: sx|master`) as the very last lines.

**Review fixes and asks.**
1. **Build order (mandatory).** Master's v2 module is built unchanged under its master name and source (v2 sha256 5892475a... is identical in master and worktree, and the v1 fallback sha check still passes). Master's whole setup runs first, in its original order: low / high gates, then ft with `REGA_FT_START_DEADLINE_S` = 150 s and `REGA_COPY_DEADLINE_S` = 255 s untouched. Only then is the separate module built. It has its own deadlines: `REGA_SX_START_DEADLINE_S` = 185 s (no sx build after that; master currently finishes at about 124 s) and `REGA_SX_CLASS_DEADLINE_S` = 250 s (a class's sx checks must start before it). Any failure, missed deadline or exception leaves every affected class on master's rega, and `_rega_sx_setup` never raises.
2. **Master strings.** `_REGA_HEAD`, `_REGA_BODY`, `_REGA_CPP`, `_rega_build`, `_rega_setup` and `_experts_rega` were not edited. The diff hunks are only in `prepare()` (+6 lines), `_experts` (+7/-1) and the new block after `_rega_setup`.
3. **Barrier counts, hand-verified.** I = work items of this CTA (it = blockIdx.x, +gridDim.x, ... < total). KS = K/128 = 32 for both w13 and w2. Every role (tid 0, helper warps 1-3, consumer warps 4-11) runs the identical `for it / for kb` loop with the same counter n, so per-fill arrivals always match:

| object | init count | arrivals per fill / item | I = 0 | I = 1 | I >= 2 |
|---|---|---|---|---|---|
| full[s] | 1 | tid 0 `expect_tx` arrive + tx bytes, 1 per fill | none; nobody waits | 32 fills, 8 per stage s, parity alternates | 32 I fills |
| empty[s] | 8 | lane 0 of each of 8 consumer warps, 1 per fill | none | 32 x 8 | 32 I x 8 |
| sfull[s] (new) | 3 | lane 0 of warps 1/2/3 after `__syncwarp`, 1 per fill, unconditional (also when dead) | none | 32 x 3 | 32 I x 3 |
| named bar 1 | 256 | 8 consumer warps, 2 per item | 0 | 2 | 2 I |
| `__syncthreads` (bar 0, 384) | all | only before the role split; trip count depends only on nb (uniform) | unchanged from master | unchanged | unchanged |

   - Helpers never touch named barrier 1 or `__syncthreads`. Warp 0 lanes 1-31 do nothing after `setmaxnreg`. `__syncwarp` is reached by all 32 lanes of a helper warp, because loop bounds depend only on total and KS.
   - **No phase overrun.** A helper cannot arrive for fill f+1 on `sfull[s]` before phase f completes. Its `full[s]` wait for f+1 needs the producer's `empty[s]` for f, which needs all consumers past `sfull[s]` phase f. By the same argument no waiter can fall 2 phases behind.
   - **Reuse of sx[s].** Fill f writes `sx[s]` only after `full[s]` phase f. That phase follows `empty[s]` for f-4, whose arrives follow the consumers' `dq_stage_sx` loads and `wgmma_wait<0>`. The raw scale bytes are read before the helpers' `sfull` arrive, which precedes the consumers' `empty` arrive and hence the TMA refill.
   - **Waits stay bounded.** Every wait is `mbar_wait` (2^26-cycle timeout, then error word 1 and dead), so a lost arrive fails the self-check instead of hanging.
4. **Gate.** For each class whose master rega is ON (low at T=753, high at 3012, ft at 16384), in this order:
   - **(a) Exactness at every class size** (low 753 / 1024; high 3012 / 4096, each random / skewed / one; ft 4500 local, 8192 random / local, 16384 random / skewed / one / local). One align + gather is shared by both variants. hs and y must be bit-identical (`torch.equal`, max |diff| printed) between `rega_gemm` and `rega_se_gemm` in zeroed buffers. `_experts_rega_sx` must equal `_experts_rega` bit for bit. Error words must be 0 and each variant must leave its own launch markers.
   - **(b) Two-state check.** States A (prepared) and B (the judge-style tensor-cloning copy: own bank, own error word; tensor maps per call; the module is shared, and there are no module globals). Each state gets a graph of the sx block and one of master's rega block over that state's own static inputs. 8 refilled replays alternate A B / B A over random, skewed, one and random. On each, the sx output must equal master's graph output bit for bit, word 0, and the sx markers must be in THAT state's error tensor.
   - **(c) Timing.** Cold-L2 (256 MB flush), the four graphs interleaved over 11 alternations, empty-graph baseline subtracted, medians pooled over A and B per variant. sx is ON only if sx <= 0.98 x master's rega block.
   - **Kernels alone.** Printed per class: one-launch graphs of w13 and w2 for master and sx, with us, TB/s (touched experts' weight bytes) and TFLOP/s.
5. **Spills / attributes.** Each sx instantiation prints regs, local bytes, static and dynamic smem, CTAs/SM and binaryVersion. Any spill or failed attribute check turns it off: the stage-1 pair turns every class off (ft's T just above 4096 also launches it); the first-turn pair turns ft off.

**Deviations.**
- (a) The module name is `muse_prefill_rega_se`, a separate extension (review and caller), not one `muse_prefill_rega_v3` module.
- (b) The spec's step (2) asked for master's `_rega_replay_check` plan (A / B deepcopy / C clone, fp32 bound). I used the caller's two-state alternating check, compared bit-exactly against master's rega graphs. This is stronger than an fp32 bound, and it skips the expensive fp32 references.
- (c) The helpers skip `decode_item`, since KS is uniform; this is simpler and fits easily in 40 registers.
- (d) The sfull arrive is per warp after `__syncwarp` (count 3), as specified. A per-thread arrive (count 96, as ssdq does) would be the fallback if the exactness gate ever showed a visibility race.

**Review open questions not built.** None of these change master's path.
- The SMSP-balanced helper distribution (warp 0 lanes 1-31).
- The consumer-only bit-trick scale conversion. This is the follow-up if SX is flat.
- A separate first-item latency probe. The kernel-alone lines at each class T are the proxy.

**Biggest risks.**
- First compile of new CUDA in a scored run. A build failure costs only sx; an illegal instruction in the sx canary would poison the context. The sx kernel adds no new PTX form: `__byte_perm`, generic ld/st.shared and `mbarrier.arrive` are all forms master already runs.
- The SX build time on top of master's 124 s. It is bounded by the 185 s start deadline and the 250 s class deadline, with a 300 s prepare limit.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-producer-scale-expand-all-t diff`:
- [x] **CUDA-graph capturable.** `kernel_fn` → `_experts` → `_experts_rega_sx` has no `.item()`, `.cpu()`, `synchronize()` or prints. Allocations are sized by shapes (`block_experts.shape`, T). The only control flow is on the token count and prepare-time bool flags. Graphs are captured in prepare's two-state check.
- [x] **Nothing derived from `data` is cached.** `state['rega_sx']` holds plain bools set in prepare(). `rega_se_ext` is a module. Tensor maps are encoded per call from that call's xs / hs.
- [x] **Weights.** No new weight storage: `rega_kernel_sx` reads master's NVFP4 / E4M3 bank bytes from global memory. The bf16 scale values exist only in per-stage shared memory inside one launch, the same values master builds in registers. Dequant happens in registers, with fp32 accumulation.
- [x] **Activations** are BF16 (xs, hs, y as master). KV cache and queries are untouched (attention not changed).
- [x] **Attention sees every key.** n/a: attention is not touched.
- [x] **No timing patches,** harness frames, I/O, secrets, module edits, reference import, subprocesses or network. Building is in-process `load_inline` only; there is no `subprocess`, `multiprocessing`, `fork` or `spawn`.
- [x] **All work happens inside the timed call;** no output caching or replay. prepare-time graphs and checks are discarded, and `kernel_fn` recomputes every route.
- [x] **Two state copies.** No C++ / static / device globals (audit clean). There are no Python tensor globals. Bank and error word are per state, and the per-state check runs on A and B.
- [x] **sm_90a.** `TORCH_CUDA_ARCH_LIST=9.0a` is set around `load_inline` and restored. wgmma / setmaxnreg bodies sit under `__CUDA_ARCH_FEAT_SM90_ALL` with an `#else` that sets error word 2. -lineinfo is on. Every launch in the sx canary and in the first exactness case of each class is announced, synced and printed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`): `OK with warnings: muse_prefill_layer/submission.py`. There are 81 warnings:
- 78 host-syncs, all outside kernel_fn's call graph and fine in prepare(). The ones at L4640-L4995 are the new sx setup's `synchronize()` calls; the rest are master's.
- 3 uint8 `torch.empty` dtype notes at L3428/3429/3677, on master's bank retile and pad.

There are no errors and no warnings on the kernel_fn path. `python3 -I -m py_compile`: OK. `__pycache__` removed.

## Risks / what could fail
low
Low risk: no new MMA form, no register-A hazard, the new sync is an mbarrier (bounded wait, error word), and correctness reduces to an exact compare with master (max diff 0). Rules: scales stay E4M3 bytes in global memory; the bf16 expansion exists only in per-stage shared memory inside one launch, as master already builds the same values in registers, so it is not a stored BF16 weight copy. Main risk: the helper hop (wait full, convert, arrive sfull) adds latency in front of each stage; with 4 stages it should hide, but if the ring runs dry the kernel is flat or slower and the gate keeps master. Second risk: shared memory at 211968 B dynamic + up to ~14.3 KB static in the small class leaves ~6 KB; print both. Sibling experts-rega-pingpong-consumers-all-t tests the other consumer hypothesis (serialization, not instruction count); each is attributable on its own rega rows; stack only after both are read.

## Result (after the run)
- job `6be2a8d3cf7d489f` (submit), status done, passed True, tests 5/5
- speedup 1.4538x, 95% CI [1.4320, 1.4756], score 266478.14228484535 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4586709689454935, "ci": [1.4367909044113112, 1.4805510334796759], "job": "1cc92a43b8be40f8"}; base result: muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-producer-scale-expand-all-t.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-producer-scale-expand-all-t.patch`; branch `hyp/muse_prefill_layer/experts-rega-producer-scale-expand-all-t`
