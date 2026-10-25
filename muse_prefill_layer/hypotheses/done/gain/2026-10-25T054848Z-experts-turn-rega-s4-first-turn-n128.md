---
problem: muse_prefill_layer
slug: experts-turn-rega-s4-first-turn-n128
created: 2026-10-25T054848Z
base: master
area: "w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM"
change: >
  Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries
  the ported _wg_* module and its per-state scratch). First_turn is the biggest absolute gap: the
  Triton engine runs 8.08-9.17 ms per layer (37-57% of every first_turn shape; 57% of the
  weight-138 sliding ctx32768 shape) against a 1670 us math floor (~16384 local routes x 2 x 50.3
  MFLOP), ~5x, and every Triton knob is closed (packed asm, unmasked loads, persistent grid,
  BLOCK_T 256: REWRITE_READ). The s4 engine dequantizes into the register A fragment and pipelines
  it against async wgmma, which Triton's lowering does not. At T=16384 each local expert gets ~512
  routes, so: (1) in _experts_triton, when state['rega_ft_ok'], align with
  moe_align_block_size(local_ids, WG_BSM, LOCAL_EXPERTS), and give every expert ceil(routes/128)
  consecutive N=128 tiles, ordered expert-major so an expert's weight tile is re-read from L2 by
  its neighbours (print the tile count and the max tiles per expert); (2) size xs/hs scratch per
  state for 16384 x 8 local routes worst case only if memory allows (2 x ~1.07 GB per state; print
  max_memory_allocated), else for the scored all-soft case with a prepare-asserted capacity and
  Triton fallback for any T whose worst case does not fit (shape-based, never value-based); (3)
  the w13 silu*up epilogue removes act_and_mul (289 us) and cache1. If the single-CTA-per-tile
  kernel leaves SMs idle at the 128-tile granularity, try a 2-CTA-per-SM config only if regs <=
  128 (print occupancy). Gate shape T=16384 soft routing: new (align + gather + w13/silu + w2) vs
  master's (moe_align_block_size + count_and_sort + Triton w13 + act_and_mul + Triton w2). T <=
  4096 keeps whatever stage 1 chose. RULES (all stages): kernel_fn stays CUDA-graph capturable
  with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no
  data-dependent host control flow, no prints, no try/except and no per-call mutable Python state;
  dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision
  per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the
  kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16
  activations, fp32 accumulation; every route of every token is computed. No child processes
  (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup
  copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA
  maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either
  encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or
  stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed
  call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at
  bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)).
  prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in
  place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force
  every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound
  calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero
  device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set
  os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after);
  extra_cuda_cflags exactly
  ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg
  asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in
  the C++ bindings never call a templated Tensor::data_ptr<T>(); use
  reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol
  at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes /
  smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag;
  LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved
  alternations against master's exact incumbent block, empty-graph baseline subtracted, at the
  scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win,
  not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST
  lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a
  scored submit.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Triton pair + act_and_mul 8.4-9.5 ms -> ~4-5.5 ms if the engine holds s4's ~300 TFLOP/s at N=128 (2048 routes: 206 GFLOP in 700 us); sliding ctx32768 14265 -> ~10500 us; floor 1670 us"
  - shapes: "*load=turn_*"
    direction: unchanged
    why: "T <= 4096 keeps stage 1's choice"
risk: high
depends_on: experts-turn-rega-s4-port-t4096
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) and act_and_mul at T > 4096
review: proceed
status: done
review_at: 2026-10-25T120230Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T120231Z
base_sha: c0cdf3a56f74658d323e211e760d12b57dcf6772
base_blob: 712edc5ea5be2dde8f558a016d0877b285340ee4
branch: hyp/muse_prefill_layer/experts-turn-rega-s4-first-turn-n128
worktree: .worktrees/muse_prefill_layer/experts-turn-rega-s4-first-turn-n128
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T121653Z
impl_commit: d375dd60d9615046903966bfa1143a876e566b8a
submission_sha256: 12a9680ac1d8b09158917d5cec51dd07e8af044263357248ca997c07eed5b99b
norm_sha256: af4fbff4b62d1424d07b1d7d3689809b5c6e6c3e41115fb908a2b699789c11b9
submission_blob: c2e3e595def2e6b5949e0fd350ce98ed59729147
lint_warnings: 59
attempt_file: 2026-10-25T121653Z-check-experts-turn-rega-s4-first-turn-n128.md
prepared_at: 2026-10-25T121653Z
job: 07a069bc69ba42a8
dispatched_at: 2026-10-25T121702Z
outcome: gain
verdict_detail: gain
speedup: 1.4060131381907617
speedup_ci: [1.3849229411179003, 1.4271033352636229]
harvested_at: 2026-10-25T124810Z
best_at_harvest: 1.3274587526558561
promotion: merged
promotion_commit: a1a5497d7a46aeb3c1aa4a18e70871a23c1b6be3
promotion_attempt: 2026-10-25T124812Z-submit-experts-turn-rega-s4-first-turn-n128.md
---

# experts-turn-rega-s4-first-turn-n128

## Change

Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is the biggest absolute gap: the Triton engine runs 8.08-9.17 ms per layer (37-57% of every first_turn shape; 57% of the weight-138 sliding ctx32768 shape) against a 1670 us math floor (~16384 local routes x 2 x 50.3 MFLOP), ~5x, and every Triton knob is closed (packed asm, unmasked loads, persistent grid, BLOCK_T 256: REWRITE_READ). The s4 engine dequantizes into the register A fragment and pipelines it against async wgmma, which Triton's lowering does not. At T=16384 each local expert gets ~512 routes, so: (1) in _experts_triton, when state['rega_ft_ok'], align with moe_align_block_size(local_ids, WG_BSM, LOCAL_EXPERTS), and give every expert ceil(routes/128) consecutive N=128 tiles, ordered expert-major so an expert's weight tile is re-read from L2 by its neighbours (print the tile count and the max tiles per expert); (2) size xs/hs scratch per state for 16384 x 8 local routes worst case only if memory allows (2 x ~1.07 GB per state; print max_memory_allocated), else for the scored all-soft case with a prepare-asserted capacity and Triton fallback for any T whose worst case does not fit (shape-based, never value-based); (3) the w13 silu*up epilogue removes act_and_mul (289 us) and cache1. If the single-CTA-per-tile kernel leaves SMs idle at the 128-tile granularity, try a 2-CTA-per-SM config only if regs <= 128 (print occupancy). Gate shape T=16384 soft routing: new (align + gather + w13/silu + w2) vs master's (moe_align_block_size + count_and_sort + Triton w13 + act_and_mul + Triton w2). T <= 4096 keeps whatever stage 1 chose. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=first_turn*` faster: Triton pair + act_and_mul 8.4-9.5 ms -> ~4-5.5 ms if the engine holds s4's ~300 TFLOP/s at N=128 (2048 routes: 206 GFLOP in 700 us); sliding ctx32768 14265 -> ~10500 us; floor 1670 us
- `*load=turn_*` unchanged: T <= 4096 keeps stage 1's choice

## Risk

Milestones: M1 correct and replay-safe at T=16384; M2 <= 0.95x Triton (gate ON); M3 <= 5 ms for the replaced block.

## Milestones

Rewrite stage 2 of `_moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) and act_and_mul at T > 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T120230Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is now the best prefill check of the day. first_turn is the weakest load (1.19x against 1.39x/1.42x for turn_1/turn_4) and its Triton block costs 8.3-8.7 ms. The landed rega kernel already measured about 290 TFLOP/s at T=3012 on the same bucket-128 code path, so the replaced block should drop to roughly 4.7-5.9 ms (headline about 1.33 -> 1.41-1.45). Before building, the spec must be rewritten against the parent's real code. The main fix is block/chunk table capacity and shared memory for the worst case at T=16384. Its sub-items (1) and (3) are already implemented, and its per-state scratch and 2-CTA/SM items are wrong.

### Justification
MECHANISM: plausible and largely already demonstrated. The incumbent at T=16384 is _moe_nvfp4_gemm_kernel x2 at 7863/8233/8095/8003 us, plus act_and_mul at about 284 us, plus SGLang moe_align_block_size + count_and_sort at about 137 us. Together that is 8.3-8.7 ms per first_turn shape (54/58/44/37% of the shape) against a 1.67 ms math floor (16384 local routes x 100.6 MFLOP = 1.65 TFLOP). Calibration from the parent's judged profile at turn_4 sliding: 3012 local routes = 303 GFLOP, and rega w13 702 us + w2 339 us = 1041 us, which is 288/298 TFLOP/s. At T=3012 an expert holds about 94 routes, so its chunk is bucket 128 with only 6/8 blocks valid (25% of the MMA N wasted). At T=16384 an expert holds about 512 routes, which is 4 full 128-route chunks, so efficiency should hold or improve: 1.65 TFLOP / 290-350 TF/s = 4.7-5.7 ms for the GEMMs, plus about 50 us for _align_parallel and about 60 us for rega_gather. The w13 silu*up epilogue removes act_and_mul for free. Expected replaced-block time is 4.8-5.9 ms against 8.3-8.7 ms, a gate ratio of about 0.55-0.70, so it should clear 0.95. Weights are 0.86 GB NVFP4 for all 32 experts, which is about 180 us to stream once. decode_item already orders items expert-major with the k chunks of one tile j consecutive, so the 4 chunks re-read each weight tile from L2. The kernel stays compute-bound, and 10240 w13 items / 5120 w2 items over 132 persistent CTAs leave a negligible tail. HEADLINE ARITHMETIC: first_turn load is SGLang 3336810 vs ours 2800766 = 1.191. Saving 2.5-3.4 ms on each of the 186 weighted shapes gives 1.43-1.54, and the geomean with turn_1 1.385 and turn_4 1.418 goes from 1.327 to about 1.41-1.45. No other queued prefill item comes close. The ssdq line is dominated (gate 805 rega vs 1250 ssdq at T=3012), so experts-first-turn-ssdq-s2-large-m, experts-turn1-ssdq-s2-m64-tile and experts-turn-rega-s4-w2-only-t4096 are superseded by the landed parent and should be deprioritised or held. Triton first_turn knobs are closed. SPEC CORRECTIONS (what the builder must do instead of the literal text): (a) Names and structure: the parent has rega_* (REGA_BSM=16, _experts_rega, _rega_setup, _rega_gate, state['rega_ok'] with classes 'low'/'high'), not _wg_*/WG_BSM. Its xs/hs scratch is allocated PER CALL in the graph pool, not per state, and that copy-safe design must stay. Do NOT add per-state 1.07 GB buffers: 2 states x 2.1 GB, plus deepcopy hazards. Per-call memory at T=16384 is xs 1.08 GB + hs 1.08 GB + y 1.07 GB, about the same as the Triton path's cache2 1.07 GB + cache13 2.15 GB. (b) Item (1) is already implemented: _experts_rega aligns with _align_parallel(local_ids, 0, 32, 16), chunks are at most 8 blocks = 128 routes, and decode is expert-major. Do not add a moe_align_block_size variant. The dispatch change is only: in _experts, for MARLIN_MAX_TOKENS < T <= REGA_FT_MAX (=16384) and state['rega_ok']['ft'], call _experts_rega; otherwise _experts_triton unchanged; T > 16384 always goes to Triton (shape-based). (c) Item (3) is already in the parent (G=true epilogue); nothing to build. (d) Drop the '2-CTA per SM if regs <= 128' branch: the kernel is __launch_bounds__(384,1) with 187-204 KB of dynamic shared memory and setmaxnreg 232 on consumers, so 2 CTAs/SM is physically impossible. It is a distraction from the single change. (e) THE REAL WORK, capacity: MAXBLK=2304 and MAXC=512 were sized for T<=4096. At T=16384, _align_parallel gives cap = 131072 + 32*15 = 131552 slots, i.e. n_blocks = 8222. A soft routing needs only about 1060 blocks and about 160 chunks, so it would fit today's tables by luck. A local-heavy routing (every route on one of the 32 local experts) needs up to 8222 blocks and 1060 chunks. The kernel then clamps nb and sets error word 3, and the timed call silently produces truncated output. That is value-dependent incorrectness, which the 'shape-based, never value-based' rule and the held-out rerun forbid. Tables must cover the worst case at the largest T served: MAXBLK >= 8224, MAXC >= 1060 (round to 1088). Shared memory does not fit if s_be stays in shared memory. Static would be 32.9 KB (s_be) + 8.7 KB (chunk tables), about 42 KB; w2 dynamic is 1024 + 4*41984 + 128*272 = 203776 B; the total of about 245.7 KB exceeds 227 KB (w13 at about 229 KB barely fits). Fix: template rega_kernel on a capacity class. The <=4096 instantiation stays byte-identical (MAXBLK 2304 / MAXC 512, s_be in shared memory) so the landed turn rows cannot move. The large instantiation reads block_experts from global with __ldg (it is used only in the per-CTA chunk scan and in 2 loads per item in decode_item) and keeps s_cblk/s_cnb[1088] in shared memory (about 8.7 KB, under the existing 16384 static bound). Fallback, if the template is awkward: STAGES=3 for the large w2 instantiation only. Keep the static_assert on the budget for both instantiations and bump REGA_EXT_NAME to _v2. (f) Self-checks at T=16384: extend _align_par_setup's token list to 16384, because today it validates only up to 4096. Run _rega canary/self-check and the A/B/C replay check at T=16384 with routings random (soft), skewed, one-expert, and LOCAL-SATURATED (top-8 drawn only from the 32 local experts, which forces about 8222 blocks / 1060 chunks and proves capacity with error word 0). Also add one unscored in-between T (e.g. 8192) to exercise the class boundary. (g) Gate: a new class 'ft' at T=16384 soft via the existing _rega_gate (rega vs dispatch with rega flags off, i.e. exactly master's _experts_triton block), 9 alternations, 256 MB cold-L2 flush, ON at <= 0.95. The incumbent is the Triton block only: ssdq does not serve T>4096, and Marlin at 16384 lost to Triton long ago (stage 4 gain). Optionally add Marlin to the gate for information. Print 'PATH CHOSEN T in (4096, 16384]: rega|triton; T > 16384: triton' as the last line along with the existing two. (h) state['zero_nonlocal_w2'] must be False on this path. If it were True, _experts_rega does torch.zeros on a 1.07 GB y, a memset of roughly 250-450 us that eats a tenth of the win. Print it in the gate line. WEAKEST LINK: the 290-350 TF/s extrapolation from T=3012 to T=16384. A 7.9x larger B stream (xs rows 131552 x 8 KB through TMA, hs 134 MB) and 4-way weight sharing through L2 under 132 CTAs have not been measured. If the kernel were weight-stream-bound at small N but drops at large N because of B-side TMA or L2 pressure, the gain shrinks toward a tie. The in-prepare gate protects the headline either way. Second-weakest: prepare memory and time. The replay check at T=16384 builds 3 states x graphs of about 3.2 GB each, so free each T's graphs, call empty_cache before the next T, and print max_memory_allocated. The parent's prepare was 69.3 s of the 200 s budget, plus about 15-20 s more build for the new instantiations. The M3 milestone (<= 5 ms) is borderline. Expect M1 and M2.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 (parent, landed gain 1.3275): this stage reuses its kernel, bank, gather and gate. A template change that alters the <=4096 instantiation could move turn rows and mix stage-1 drift into the stage-2 reading. Keep that instantiation byte-identical, and expect turn rows flat within CI.
- w2-nonlocal-zero-store-off-first-turn (in stack-tonight-five-prefill): state['zero_nonlocal_w2'] decides whether _experts_rega memsets the 1.07 GB y at T=16384. If it is True on the judge, rega pays about 250-450 us that Triton's zero_nonlocal_w2 path also handles differently, which biases the comparison. Verify and print it.
- moe-align-local-routes-parallel-port / align-local-parallel-first-turn (in the five-stack; the latter neutral): rega at T=16384 uses _align_parallel (33 programs each streaming 131072 ids) instead of SGLang's moe_align_block_size + count_and_sort (137 us). The align cost is part of the replaced block, so attribute it via the rega tracer timings or the profile rows, not to the GEMM.
- experts-hybrid-marlin-turn-triton-first-turn / experts-triton-w4a16-first-turn-blockt256 (landed): they define the incumbent (Triton BLOCK_T 256 block). The gate's 'dispatch' alternative is exactly that block, so there is no ambiguity about what is beaten.
- sglang-pipeline-parity: fused_post_expert_norm_weighted_sum is shared by both paths and is not a confounder.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage whose in-prepare gate is itself an interleaved cold-L2 A/B against master's exact incumbent block at the scored T. Its PATH CHOSEN plus the profile rows (rega_kernel<true>/<false> replacing _moe_nvfp4_gemm_kernel x2 + act_and_mul + count_and_sort/moe_align on the four first_turn rows) attribute the result directly. Turn rows are protected by the byte-identical <=4096 instantiation. If the gate turns the path OFF, the run is a measured negative with both medians logged and no headline ambiguity. If it is ON and first_turn rows move by roughly the gate delta, the attribution is clean. An ab-revert of the parent is meaningless here, because the change is built from the parent's kernel. Spend the remaining prefill checks on the next stage, not on a control.

### Open questions
- Does rega hold at least 290 TF/s at T=16384 with full 128-route chunks, or does the B-side TMA/L2 traffic (xs 131552 rows) cap it? Print per-kernel us for w13 and w2 from the tracer at T=16384 in prepare.
- Template (s_be from global in the large instantiation) or STAGES=3 for large w2: which does the builder take? The template is preferred because it leaves the 4-stage pipeline intact. Report regs, smem and CTAs/SM per instantiation.
- Should the 'ft' class serve all of (4096, 16384] or only T=16384? Prefer the whole range with worst-case tables, plus a self-check at an in-between T, because the final review reruns nearby shapes.
- Prepare peak memory with 3 replay states x about 3.2 GB graph pools at T=16384: print max_memory_allocated and release graphs between T classes.
- Queue hygiene: experts-turn-rega-s4-w2-only-t4096, experts-first-turn-ssdq-s2-large-m and experts-turn1-ssdq-s2-m64-tile are superseded by the landed rega gate result (rega 0.644-0.648x of the best alternative). Consider holding them so they do not take the remaining prefill checks.



## Implementation notes

Built 2026-10-25 on worktree base c0cdf3a5. Its muse_prefill_layer/submission.py is byte-identical to master 64d709d1, which carries experts-turn-rega-s4-port-t4096 (gain 1.3275), so no merge was needed. Only `muse_prefill_layer/submission.py` changed: +331/-56.

**What changed.** The adversarial review overrides the spec. Each review item and what was done:

- **(a) Names and scratch.** Uses the parent's names (`REGA_BSM`, `_experts_rega`, `_rega_setup`, `_rega_gate`, `state['rega_ok']`) and adds one class, `'ft'`. No `_wg_*` and no `WG_BSM`. xs/hs/y stay per-call graph-pool allocations, as in the parent. There are no per-state 1.07 GB buffers. The only per-state device data are still `rega_bank` and `rega_err`.
- **(b) Align and dispatch.** The spec's moe_align_block_size/N=128 item is already in the parent: `_align_parallel` at block 16, chunks of at most 8 blocks (128 routes), expert-major decode. Nothing was added there. The one dispatch change is in `_experts`: for `MARLIN_MAX_TOKENS < T <= REGA_FT_MAX (16384)` with `state['rega_ok']['ft']`, call `_experts_rega`. Otherwise `_experts_triton` runs unchanged, and T > 16384 always goes to Triton. The branch reads only the token count and the prepare flag.
- **(c) silu\*up epilogue.** Already in the parent (G=true). Nothing was built.
- **(d) 2-CTA/SM branch.** Dropped.
- **(e) Capacity.** `rega_kernel` is now `template <bool G, bool BIG>`.
  - **Small class (`BIG=false`).** MAXBLK 2304 and MAXC 512, with `s_be[2304]` in shared memory, as in stage 1. The edits are type-level only: `decode_item` is templated on the table type and gets the same `const int*` to `s_be`. The `s_be` fill sits under `if constexpr (!BIG)`. `MAXC` becomes `Cap<false>::C` (=512). The SASS should match stage 1, but no nvcc was available here to diff it. The mangled kernel name changes (`rega_kernel<true, false>`), so profile rows are renamed.
  - **Large class (`BIG=true`).** MAXBLK_FT 8224 and MAXC_FT 1088. block_experts is read from global memory through `GTab` (`__ldg`): in the run-start scan, the chunk scan, the group loop and the 2 loads per item in `decode_item`. Only `s_cblk`/`s_cnb[1088]` (8.7 KB) and the per-expert arrays sit in static shared memory. A static_assert checks both classes against the 16384 B static bound. w2 is 203776 B dynamic plus about 9.3 KB static, under 227 KB, so the 3-stage fallback was not needed and all instantiations keep 4 stages.
  - **Class selection.** Host-side in `rega_gemm`, by `block_experts.numel() > MAXBLK`. The table length depends only on the shape: (T·8 + 480)/16. Every T ≤ 4096 (at most 2078 blocks) gets the stage-1 instantiation. T in (4096, 4548] also stays in the small class: at most 2304 blocks and at most 320 chunks, under 512. The `TORCH_CHECK` bound is now MAXBLK_FT.
  - **Worst case.** prepare prints the worst-case arithmetic: 8222 blocks against a table of 8224, and at most 1059 chunks against 1088.
  - **Other build changes.** `rega_init` now reports 4 instantiations (flag = G + 2·BIG). `REGA_EXT_NAME` is `muse_prefill_rega_v2`.
- **(f) Self-checks at first-turn sizes.**
  - **Align.** `_align_par_setup` now also runs `_align_par_ft_setup`. It checks T 8192 and 16384 with random, skewed and local-saturated routings, at Marlin's block size and at REGA_BSM=16, against moe_align_block_size. It sets only `state['align_par_ft_ok']`, so a miss there cannot turn off master's Marlin-path align or the low/high rega classes.
  - **Eager self-check.** The new `_rega_ft_setup` checks (4500 local), (8192 random/local) and (16384 random/skewed/one/local) against the fp32 reference. The bound is max(2× Triton's own error, 8e-3), with worst row at 5×. Each case also requires error word 0, both launch markers and finite output. Each line prints the block-table length, the class, used blocks, chunks, max chunks per expert and the w13/w2 item counts. The tracer now prints the eager µs per launch (align, gather, w13, w2), which answers the review's per-kernel question at T=16384. The local-saturated cases force about 8192 blocks and about 1030-1059 chunks, so they test capacity: error word 3 would be a MISS.
  - **Copy/replay check.** `_rega_replay_check` now takes a `plan` (default unchanged). It runs A (prepared), B (deepcopy) and C (tensor clone), replayed A/B/C/A/B with inputs refilled in place. The plan is T=8192 (random, local) and T=16384 (random, skewed, one, local), with the bound calibrated on Triton above 4096. Graphs are freed and `empty_cache` is called after every T, and `max_memory_allocated` is printed (peak stats are reset at the start of ft).
- **(g) ft gate.** `_rega_gate` at T=16384 soft. Above 4096 it includes only `dispatch`, which is master's Triton block with the ft flag still False, plus the empty-graph baseline. ssdq and Marlin are excluded above MARLIN_MAX_TOKENS; their presence at 753/3012 is unchanged. It uses the parent's 9 alternations, 256 MB flush, rotated and flipped order, and baseline subtraction. ft is ON at rega/dispatch ≤ 0.95. The gate line prints both medians, the ratio, zero_nonlocal_w2 and max_memory_allocated. The final lines of prepare are the existing PATH CHOSEN line (its tail now reads "T > 4096: next line"), then `[rega] PATH CHOSEN T in (4096, 16384]: rega|triton; T > 16384: triton`.
- **(h) zero_nonlocal_w2.** It is a precondition of ft. If it is True, prepare prints `OFF ft: zero_nonlocal_w2 is True ...` and ft stays OFF. It is printed in the capacity line and in the gate line.
- **Failure isolation.** All ft work runs after the low/high gates, inside its own try. Any ft miss, exception, budget overrun or attribute failure prints one `[rega] OFF ft: <reason>` line and sets only `rega_ok['ft']=False`. The low/high classes and master's Triton path stay as they were. If the first-turn instantiations fail their attribute check (`attrs_ok[True]`), only ft goes off. If the stage-1 instantiations fail, everything goes off, as in the parent.
- **Budget.** ft is skipped if prepare is already past 150 s when it starts. The ft gate is skipped past `REGA_BUDGET_S` = 200 s. Phase timings print per ft step (ft-self-check, ft-copy-check, ft-gate). Expected cost: about 15-20 s more build for 4 instantiations, about 3 s of first-turn align cases, and about 25-40 s for the ft phase, on top of the parent's 69 s. That makes about 115-130 s.

**Not done / deviations.** Marlin is not in the ft gate (the review made it optional). The spec's per-state scratch and 2-CTA branch were dropped per review (a) and (d). Byte-identity of the small instantiation is argued from the source, not proven: nvcc is not on this node. A g++ `-fsyntax-only` pass over the CUDA source with CUDA/torch stubs, `<<<>>>` stripped and all 4 instantiations, compiled clean. The parent's `_rega_audit_cpp` global/static audit returns no findings on the new source, and the templated-accessor audit is clean.

**Biggest risks.**
1. Compile: the new template/`if constexpr`/`GTab` code under nvcc (checked only with g++).
2. Performance: the B-side TMA/L2 traffic at T=16384 (131552 xs rows) may keep rega above 0.95× Triton, in which case the gate keeps Triton.
3. Prepare memory/time at T=16384: 3 replay states of about 3.4 GB graph pool each, plus the 6.4 GB fp32 dequant cache, printed.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-turn-rega-s4-first-turn-n128 diff`:

- [x] CUDA-graph capturable. The only kernel_fn-path change is `_experts`' branch on `tokens` (a shape) and `state['rega_ok']['ft']` (a prepare flag). `_experts_rega` is the parent's graph-safe body: per-call `torch.empty` sized by shape, and the capacity class is picked in C++ from `block_experts.numel()`, a shape. No `.item`/`.cpu`/sync/print, no try/except and no per-call mutable state in the timed path. All new syncs, `.cpu()` calls and prints are in prepare-only helpers (lint: every host-sync is outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls. There are no pointer-equality gates and no module globals holding tensors (the `_rega_audit_globals` scan is kept). TMA maps are still encoded per call from that call's xs/hs.
- [x] Weights: the same lossless NVFP4 retiling as the parent (`_rega_bank` unchanged). There is no BF16 copy of any expert weight. Dequant happens only in registers inside the kernel.
- [x] Activations: BF16 xs/hs/y, fp32 accumulation, silu\*up rounded to bf16 once, unchanged from the parent. KV/attention untouched.
- [x] n/a: attention is untouched. Every key is still attended (FA3 path unchanged).
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses (in-process load_inline only) or network.
- [x] All work happens inside the timed call. No output caching or replay. Every route of every token is computed: tables cover the worst case at T ≤ 16384, overflow sets error word 3, and the local-saturated self-check proves word 0.
- [x] Two judge states: the A/B/C copy/replay check is extended to T 8192/16384. No C++/static/module state was added.
- [x] Build: `TORCH_CUDA_ARCH_LIST=9.0a` set and restored, extra_cuda_cflags `['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']` (unchanged). wgmma/setmaxnreg stay under `__CUDA_ARCH_FEAT_SM90_ALL` with the error-word-2 `#else`. No templated `data_ptr<T>`. Bounded mbarrier waits are unchanged.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):

```
OK with warnings: .worktrees/muse_prefill_layer/experts-turn-rega-s4-first-turn-n128/muse_prefill_layer/submission.py
(warnings only: host-sync lines, all "outside kernel_fn's call graph; fine in prepare()"; dtype uint8 at L2902/L2903
 (rega bank, parent code) and L3151 (replay-check pad, parent code))
```

py_compile: ok. undef.py: `[]` (no unbound names). `__pycache__` removed. Not committed; okc not run.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent, against master muse_prefill_layer/submission.py. **Verdict: dispatch-with-fixes.**

**Fix (build-failure blast radius).** Before this fix, a compile or load failure of `muse_prefill_rega_v2` appended "build failed" and turned off all of rega (`low`, `high` and `ft`). That would drop master's landed T<=4096 classes to Marlin/ssdq and read as a large regression. Now, when the v2 build returns None, `_rega_setup` builds master's exact stage-1 extension (`muse_prefill_rega_v1`).
- `_rega_src_v1()` derives that source from the v2 source by reverting this stage's 26 edits (`_REGA_V1_EDITS`), and uses it only if its sha256 equals master's (`REGA_V1_SHA256`). On this node, the derivation was checked to be byte-identical to master's `_rega_src()`.
- The fallback marks the first-turn class unavailable (`attrs_ok[True] = False`) and sets `ft` OFF with the reason "v2 build failed; running master's muse_prefill_rega_v1". It also prints `[rega] v2 build FAILED; fallback: ...`.
- low/high then go through master's unchanged canary, self-check, replay check and gates.
- `_rega_build` now takes `(name, src)` and its defaults are unchanged. `hashlib` is imported.

**Checked, no change needed.**
- **Small class.** With `BIG=false`, the code is semantically identical to master: `s_be[2304]`, `s_cblk`/`s_cnb[512]`, the same fill, clamp and err-3 logic, and `be` is a plain pointer to `s_be`. Every T<=4548 launches it (blocks_len <= 2304), so every T<=4096 does.
- **Large class.** Every `__ldg` index into block_experts is below nb <= min(blocks_len, 8224), and blocks_len at T=16384 is 8222. The s_cblk/s_cnb writes are guarded by idx < 1088. The worst case is <= 1056 chunks, under 1088. Overflow sets err 3 and clamps, and decode_item stays inside [f, f+k) < nc. Output, xs and gather offsets are size_t.
- **Memory.** The fp32 dequant cache (6.4 GB) is prepare-only and cleared in a `finally` before the gate. Replay-check graphs are freed per T. Estimated peak is about 6.4 + 3 x 3.4 GB + transients, well under 141 GB. No per-state buffers were added.
- **Budget.** ft starts only before 150 s and the gate only before 200 s, against a 300 s prepare limit. The ft phase is about 30-40 s.
- **kernel_fn.** The only change is a shape-plus-flag branch in `_experts`. The capacity class is chosen in C++ from `block_experts.numel()`, which is a shape. There is no try/except, print, sync, pointer gating or module-global device state.
- `undef.py` reports `[]`, py_compile passes, and okc_lint reports warnings only, all in parent code or prepare-only. `__pycache__` was removed.

**Residual risk:** the first-turn instantiation is unproven under nvcc. A device fault in its T=16384 self-check would poison the context, and a fallback cannot cover that. The small instantiation may also compile to different SASS through the templated `be` pointer, which could shift turn rows slightly.

## Result

- job `07a069bc69ba42a8` (check), status done, passed True, tests 5/5
- speedup 1.4060x, 95% CI [1.3849, 1.4271], score 275533.62730202224 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.3274587526558561, "ci": [1.3075468713660183, 1.347370633945694], "job": "2910b2fb4b744865"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-port-t4096.result.json
- **outcome: gain** (gain)
- files: `muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-first-turn-n128.result.json`, `muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-first-turn-n128.patch`; branch `hyp/muse_prefill_layer/experts-turn-rega-s4-first-turn-n128`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 730.17 | 777.02 | 0.9636 | lost |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1105.64 | 1136.06 | 0.9908 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2167.2 | 2251.37 | 1.0151 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3748.07 | 3785.25 | 0.9905 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14585.46 | 12148.27 | 1.21 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14010.59 | 11658.83 | 1.2286 | gained |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17741.73 | 15589.38 | 1.167 | gained |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21874.7 | 19167.43 | 1.1623 | gained |

gained on full first_turn; sliding first_turn; lost on sliding turn_1. Expected hit: 1/2. Noise ±1.5%.

## Follow-ups

- rewrite stage 2 of `_moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) and act_and_mul at T > 4096`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-turn-rega-s4-first-turn-n128 --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.
- candidate for promotion; stack further changes with `depends_on` once it lands.

