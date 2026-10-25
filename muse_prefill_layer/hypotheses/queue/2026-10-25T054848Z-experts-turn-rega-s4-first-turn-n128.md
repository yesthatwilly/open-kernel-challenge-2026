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
status: queued
review_at: 2026-10-25T120230Z
review_companions: null
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

