---
problem: muse_prefill_layer
slug: turn-moe-local-align-sorted-silu
created: 2026-10-24T185232Z
base: master
area: MoE align/dispatch + silu-mul on the Marlin path (T <= 4096)
change: >
  Built on master (635cfee). Only _experts_marlin (T <= 4096: turn_1 T=753, turn_4 T=3012 and
  every test) and its call site change; _experts_triton (T > 4096) is byte-identical. Two parts:
  (1) LOCAL-ROUTES ALIGN exactly as hypothesis moe-align-local-routes-parallel-port specifies, on
  this path only: copy muse_moe_w4a16's _align_parallel_kernel/_align_parallel, read local_start
  from data['local_start'] on device (tl.load, no .item()), local_count 32, block = master's
  block_size_m pick (64 on every prefill shape and test); it replaces _local_expert_ids and
  moe_align_block_size(local_ids, block_size_m, N_EXPERTS) on this path and returns local_ids plus
  (sorted_ids, block_experts, n_padded) with no remote (-1) blocks. Both moe_wna16_marlin_gemm
  calls take those three tensors unchanged in position (is_ep=True stays). kernel_fn passes
  topk_ids and data['local_start'] down to _experts. (2) SORTED SILU: replace
  silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
  with _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots=T*8, N=4096, BLOCK_R=16,
  BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load
  n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange(BLOCK_R); s =
  load(sorted_ids + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots; cols =
  pid1*BLOCK_N + arange(BLOCK_N); g = c1[s, cols], u = c1[s, 4096 + cols]; out = (g_f32 / (1 +
  exp(-g_f32))) * u_f32, rounded once to bf16; store c2[s, cols] under valid. With part 1 every
  valid s is a local slot, so the kernel touches only ~T rows instead of launching over T*8.
  fused_post_expert_norm_weighted_sum(..., slot_expert_ids=local_ids) is unchanged; remote rows of
  the w2 output are never written and that kernel skips them (muse_moe_w4a16 master relies on it).
  prepare() gates, one log line each, any failure keeps master's path for the run
  ('[turn-dispatch] OFF: <reason>' else 'ON'): align vs moe_align_block_size per-expert route sets
  at T = 200, 753, 3012, 4096 with block 64 and local_start 0 and 224; NaN-poison check of the
  weighted sum on remote slots; silu kernel vs silu_and_mul on local rows (<= 1 bf16 ulp);
  _check_experts_e2e adapted to the new signature.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "act_and_mul 52.8-53.5 us -> ~15-20 us (3012 local rows x 24 KB); count_and_sort 17.4-17.7 us + moe_align + remap -> one ~8-10 us launch; Marlin stops walking ~330 remote m-blocks (upside if its is_ep path pays per block): about -1.2 to -1.8%, more if the Marlin row moves"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "act_and_mul 14.0-14.4 us -> ~5 us, count_and_sort 5.2 us + align + remap -> ~4 us, ~83 remote m-blocks gone: about -1.2 to -2%"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: Triton path byte-identical
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T014119Z
review_companions: null
hold_reason: "adversarial review: Hold as a duplicate of landed work: master's _experts_marlin already runs _align_parallel (local routes only, no -1 blocks) and _silu_mul_sorted_kernel at T<=40"
outcome: expired
verdict_detail: "retired: engine runs on no scored shape (rega replaced Triton/Marlin/ssdq; prefill DIAGNOSIS 2026-10-25)"
expired_at: 2026-10-25T211931Z
---

# turn-moe-local-align-sorted-silu

## Change

Built on master (635cfee). Only _experts_marlin (T <= 4096: turn_1 T=753, turn_4 T=3012 and every test) and its call site change; _experts_triton (T > 4096) is byte-identical. Two parts: (1) LOCAL-ROUTES ALIGN exactly as hypothesis moe-align-local-routes-parallel-port specifies, on this path only: copy muse_moe_w4a16's _align_parallel_kernel/_align_parallel, read local_start from data['local_start'] on device (tl.load, no .item()), local_count 32, block = master's block_size_m pick (64 on every prefill shape and test); it replaces _local_expert_ids and moe_align_block_size(local_ids, block_size_m, N_EXPERTS) on this path and returns local_ids plus (sorted_ids, block_experts, n_padded) with no remote (-1) blocks. Both moe_wna16_marlin_gemm calls take those three tensors unchanged in position (is_ep=True stays). kernel_fn passes topk_ids and data['local_start'] down to _experts. (2) SORTED SILU: replace silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots=T*8, N=4096, BLOCK_R=16, BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange(BLOCK_R); s = load(sorted_ids + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots; cols = pid1*BLOCK_N + arange(BLOCK_N); g = c1[s, cols], u = c1[s, 4096 + cols]; out = (g_f32 / (1 + exp(-g_f32))) * u_f32, rounded once to bf16; store c2[s, cols] under valid. With part 1 every valid s is a local slot, so the kernel touches only ~T rows instead of launching over T*8. fused_post_expert_norm_weighted_sum(..., slot_expert_ids=local_ids) is unchanged; remote rows of the w2 output are never written and that kernel skips them (muse_moe_w4a16 master relies on it). prepare() gates, one log line each, any failure keeps master's path for the run ('[turn-dispatch] OFF: <reason>' else 'ON'): align vs moe_align_block_size per-expert route sets at T = 200, 753, 3012, 4096 with block 64 and local_start 0 and 224; NaN-poison check of the weighted sum on remote slots; silu kernel vs silu_and_mul on local rows (<= 1 bf16 ulp); _check_experts_e2e adapted to the new signature.

## Expected

- `*load=turn_4*` faster: act_and_mul 52.8-53.5 us -> ~15-20 us (3012 local rows x 24 KB); count_and_sort 17.4-17.7 us + moe_align + remap -> one ~8-10 us launch; Marlin stops walking ~330 remote m-blocks (upside if its is_ep path pays per block): about -1.2 to -1.8%, more if the Marlin row moves
- `*load=turn_1*` faster: act_and_mul 14.0-14.4 us -> ~5 us, count_and_sort 5.2 us + align + remap -> ~4 us, ~83 remote m-blocks gone: about -1.2 to -2%
- `*load=first_turn*` neutral: Triton path byte-identical

## Risk

Folds the held silu-mul-triton-local-rows-marlin-turn (held as below noise alone) into a larger turn-shape dispatch change, as its hold asked. Its align part is the T <= 4096 half of moe-align-local-routes-parallel-port; if both run, the act_and_mul vs _silu_mul_sorted_kernel row isolates the silu part. Shape-disjoint from first-turn-moe-dead-work-stack, so the two can be restacked by 3-way merge. Composes with any later expert engine at turn shapes (CUDA wgmma stages 2-3, Triton stages 6-7): it touches only align and silu. Risk: _align_parallel at T=3012 streams 24K ids twice per program (~8-10 us est.).

## Adversarial review

Reviewed 2026-10-25T014119Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold as a duplicate of landed work: master's _experts_marlin already runs _align_parallel (local routes only, no -1 blocks) and _silu_mul_sorted_kernel at T<=4096. Both were self-checked ON in the master jobs c4d14d62 and 502f2eab ('[align-local] ON', '[silu-sorted] ON'). Retire it; it does not need to wait for the running checks.

### Justification
Both parts landed with stack-tonight-five-prefill (moe-align-local-routes-parallel-port + silu-mul-triton-local-rows-marlin-turn). See submission.py _experts_marlin: _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) when align_par_ok, and _silu_mul_sorted(...) when silu_sorted_ok. REWRITE_READ_2026-10-25 states that on the new master the parallel align and _silu_mul_sorted_kernel replace count_and_sort plus act_and_mul on the Marlin path. The hypothesis was written against 635cfee (pre-five), so its expected savings (act_and_mul 53->15-20 us, count_and_sort 17.4 us, ~330 remote m-blocks) are already in master's 1.0705/1.0744. The only residual difference is cosmetic. The hypothesis aligns straight from topk_ids plus data['local_start'] and skips _local_expert_ids, but it still returns local_ids for fused_post_expert_norm_weighted_sum, so at most one ~2-5 us remap launch is saved. That is far below the ±1.5% turn noise and not worth a check. WEAKEST LINK: the premise that master still runs moe_align_block_size + silu_and_mul on this path is false.

### Confounders on the current kernel
- stack-tonight-five-prefill: it contains this hypothesis's two parts verbatim, so the run would measure master against master.
- Running experts-triton-turn4-persistent-local-align-control-no-persi: if it wins, turn_4 (T=3012) leaves Marlin entirely (MARLIN_MAX_TOKENS 1024), removing this hypothesis's main target shape anyway.

### Interference and A/B needs
No companion and no waiting are warranted. The change is already on master, so no result of the running first-turn persistent-grid or turn_4 control checks can make it informative. Only a new mechanism on the Marlin path would justify a run, e.g. fusing the _local_expert_ids remap into _align_parallel together with something else measurable.

### Open questions
- Move this to done/rejected (duplicate of stack-tonight-five-prefill) and add it to FINDINGS 'Do not retry', so /hypothesize stops re-minting the Marlin-path align+silu pair.

