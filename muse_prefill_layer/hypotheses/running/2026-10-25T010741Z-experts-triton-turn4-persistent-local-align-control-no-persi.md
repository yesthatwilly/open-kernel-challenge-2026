---
problem: muse_prefill_layer
slug: experts-triton-turn4-persistent-local-align-control-no-persi
created: 2026-10-25T010741Z
base: master
area: w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid
change: >
  Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send
  1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 1024, with a comment); T <= 1024
  keeps _experts_marlin. (3) In _experts_triton, when state['align_par_ok'] is set, replace
  moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS) with _align_parallel(local_ids,
  0, LOCAL_EXPERTS, cfg['BLOCK_T']), which already exists on master for the Marlin path. Extend
  _align_par_setup's self-check to cover block 128 and 256 at T=3012 and 16384 (per-expert route
  sets vs moe_align_block_size). Because no -1 tiles remain, pass zero_nonlocal=False to the w2
  call on this path. Add a prepare() NaN-poison check: fill cache3 with NaN, run the finalize, and
  assert a finite output. Run _check_experts_e2e at T=3012 as well as 8192 (rel L2 <= 2e-2, worst
  row <= 5x). Any failure sets MARLIN_MAX_TOKENS back to 4096 for the run and logs '[turn4-triton]
  OFF: <reason>'. Otherwise log 'ON'. _moe_nvfp4_gemm_kernel's grid stays as on master (no
  persistent loop). Everything else is unchanged.
expected:
  - shapes: "*load=turn_4*"
    direction: flat
    why: ~165 -1 tiles and the 170 MB zero-store removed, est. 50-120 us; kernel row 2.0-2.25 ms vs Marlin 2.12-2.15 ms
  - shapes: "*load=first_turn*"
    direction: faster
    why: local align also removes remote -1 tiles at T=16384; 1-3%
  - shapes: "*load=turn_1*"
    direction: flat
    why: T=753 stays on Marlin
risk: medium
depends_on: experts-triton-turn4-persistent-local-align
priority: 2
kind: control
controls_for: experts-triton-turn4-persistent-local-align
review: proceed
status: running
review_at: 2026-10-25T011030Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T011048Z
base_sha: e530b77b8a03d595f892b57b91b666002026a175
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-triton-turn4-persistent-local-align-control-no-persi
worktree: .worktrees/muse_prefill_layer/experts-triton-turn4-persistent-local-align-control-no-persi
mode: null
restack_of: null
stale_base: null
---

# experts-triton-turn4-persistent-local-align-control-no-persi

## Change

Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 1024, with a comment); T <= 1024 keeps _experts_marlin. (3) In _experts_triton, when state['align_par_ok'] is set, replace moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']), which already exists on master for the Marlin path. Extend _align_par_setup's self-check to cover block 128 and 256 at T=3012 and 16384 (per-expert route sets vs moe_align_block_size). Because no -1 tiles remain, pass zero_nonlocal=False to the w2 call on this path. Add a prepare() NaN-poison check: fill cache3 with NaN, run the finalize, and assert a finite output. Run _check_experts_e2e at T=3012 as well as 8192 (rel L2 <= 2e-2, worst row <= 5x). Any failure sets MARLIN_MAX_TOKENS back to 4096 for the run and logs '[turn4-triton] OFF: <reason>'. Otherwise log 'ON'. _moe_nvfp4_gemm_kernel's grid stays as on master (no persistent loop). Everything else is unchanged.

## Expected

- `*load=turn_4*` flat: ~165 -1 tiles and the 170 MB zero-store removed, est. 50-120 us; kernel row 2.0-2.25 ms vs Marlin 2.12-2.15 ms
- `*load=first_turn*` faster: local align also removes remote -1 tiles at T=16384; 1-3%
- `*load=turn_1*` flat: T=753 stays on Marlin

## Risk

Separates the turn_4 Triton-vs-Marlin question and the -1-tile removal from the persistent-grid change still being measured.

## Adversarial review

Reviewed 2026-10-25T011030Z by an adversarial subagent (model opus).

**Verdict: proceed.** Buildable as written on master (a81d19c) with three small implementation fixes, and a useful control for the parent. Two expectations are wrong: on master the w2 zero-store is already OFF, and removing -1 tiles saves almost nothing under a non-persistent grid. Read it as master's scalar-dequant Triton engine vs Marlin at turn_4, with first_turn expected flat, not faster.

### Justification
SGLang's moe_align_block_size places -1 routes in an extra padded bucket (num_experts+1, ids shifted +1, expert_ids -1); master's '33*(BLOCK_T-1)' capacity comment and _check_expert_gemms' -1 block agree. But _nvfp4_moe_gemm sizes its grid from the static capacity (221 vs 220 token tiles at turn_4) and the kernel early-exits both -1 tiles and past-n_padded tiles, so removing -1 tiles only swaps one early-exit for another; only the parent's n_padded-bounded persistent loop cashes it in. zero_nonlocal_w2 is already OFF on master (every log prints '[w2 zero-store] OFF'). At turn_4 the Marlin path already uses _align_parallel. What remains is Triton (scalar dequant, BLOCK_T 128/BLOCK_W 128/8w/3s) vs Marlin at turn_4: stages 2/3 ran 2100-2318 us vs Marlin 2115-2217; flat expected; ~7% row drift. Weakest link: first_turn 'faster 1-3%': part (3) at T=16384 was measured twice (align-local-parallel-first-turn 0.977-0.990; stack-tonight-six-prefill 0.980-1.007); expect flat to ~-1%. turn_1 stays on Marlin, a clean drift control.

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn (landed): part (1) reverses its turn_4 half; the point of the control; compare to Marlin rows 2148/2115 us.
- w2-nonlocal-zero-store-off-first-turn (landed): zero-store already off; zero_nonlocal=False changes nothing; do not credit it.
- stack-tonight-five-prefill / moe-align-local-routes-parallel-port: if the extended self-check is folded into align_par_ok, a failure also turns off Marlin's parallel align and confounds; use a separate flag.
- moe-align-32-experts-first-turn: one token tile of grid difference, negligible.

### Interference and A/B needs
No companion. Parent turn_4 minus control turn_4 = persistent-grid effect plus the -1-tile removal only a persistent grid can cash in. Implementation fixes: (a) separate flag state['align_par_triton_ok'] for the Triton-path align self-check, leaving align_par_ok untouched; (b) _experts reads state.get('marlin_max_tokens', MARLIN_MAX_TOKENS) instead of the module global (or the fallback writes it with global); (c) warm _align_parallel for BSM 128 and 256 in prepare() before capture. The NaN-poison check largely duplicates _weighted_sum_skips_remote; gating on state['zero_nonlocal_w2'] being False is sufficient. _check_experts_e2e needs a token-count argument for T=3012.

### Open questions
- Optionally print n_padded at T=3012 in prepare() with both aligns to confirm ~165 tiles difference.
- If the control loses at turn_4 by more than ~7%, the parent must win by more than that margin before the gate move is worth keeping.

