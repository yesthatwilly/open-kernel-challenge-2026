---
problem: muse_prefill_layer
slug: sliding-attn-block-triton-gqa-fused-normgate
created: 2026-10-24T183246Z
base: master
area: sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers
change: >
  Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing
  the two in-flight prefill runs touch (full-layer FA3 stays FA3, experts unchanged). In
  kernel_fn, on sliding layers only (cfg['window'] != 0), replace the four steps between
  store_cache and the o projection -- q.view(tokens,16,64).to(FP8), flash_attn_with_kvcache(...)
  (FA3 + its FlashAttnFwdCombine split merge), and state['o_norm_gate'](out, gate) -- with a new
  Triton GQA-packed sliding flash-attention kernel _sliding_prefill_attn_kernel whose epilogue
  applies the norm-gate and writes the bf16 [T, 1024] input of the o GEMM directly. qkv GEMM,
  FusedQKNorm (q/k RMSNorm + interleaved RoPE: its q output is this kernel's input, so RoPE is NOT
  reimplemented and the tl.sin issue in LESSONS does not arise), the k/v FP8 cast and store_cache
  all stay byte-identical and run first, so the step's own K/V rows are read back from the paged
  FP8 cache exactly as FA3 reads them. Full layers, the o GEMM and everything after are untouched.
  KERNEL: one program = one KV head x BLOCK_TOK=16 consecutive tokens of one request x all 8 q
  heads sharing that KV head = 128 query rows (GQA packing: each gathered K/V row feeds 8 heads,
  half the K/V traffic and twice the MMA M of FA3's per-head tiles). Prologue: load q (bf16 from
  FusedQKNorm) and round to FP8 E4M3 then back to bf16 in registers (identical query precision to
  FA3's q.to(FP8); no coarser). Key range per program = [max(0, pos_first-2047), pos_last] in that
  request's positions (window_size (2047,0) semantics: token at p sees keys p-2047..p); walk it in
  BLOCK_N=64 key tiles; per tile gather K and V rows through page_table[b, kpos // PAGE] * PAGE +
  kpos % PAGE (PAGE = cfg['page_size'] constexpr, so the page-64 test works), load FP8, upcast to
  bf16, S = tl.dot(q, tl.trans(k)) in fp32 (try FP8xFP8 tl.dot for S first if it compiles in the
  judge's Triton, else bf16), per-element mask kpos <= qpos and kpos >= qpos-2047 (every in-window
  key is attended; only tiles wholly outside every row's window are not visited), online softmax
  in fp32 with sm_scale*qk_scale_factor*k_scale folded into exp2, P kept in BF16 (finer than FA3's
  FP8 P) for tl.dot(p, v), v_scale applied in the epilogue, num_stages 3 so Triton pipelines the
  K/V gather. Split-KV only where the grid is thin: S = 4 if B*ceil(L/16)*2 < 264 (turn_1: 96
  programs x 4 splits), else S = 1 (turn_4: 384 programs; first_turn: 2048); S is chosen from
  config shapes only. S=1 epilogue: o = acc/l, then per (token, head) row the FusedRMSNormGate
  math (rmsnorm over the 64 dims, eps 1e-5, no weight, times sigmoid(gate) read from qkv[:, 1280 +
  head*64 : ...]) in fp32, store bf16 into out[T, 1024]. S>1: programs write unnormalized acc
  (fp32), m and l into a workspace preallocated in prepare() (sized for T<=4096, 4 splits, 16
  heads), then one small Triton combine kernel merges the splits AND applies the same norm-gate,
  writing out[T,1024]. ROUTING: new path for sliding layers at every T <= 4096 (turn_1, turn_4 and
  all sliding tests); at T > 4096 (sliding first_turn) prepare() times both paths as captured CUDA
  graphs on a synthetic T=16384, ctx=32768 case (cold L2 via a 256 MiB flush write before each
  replay, 20 replays, median) and enables the new kernel there only if it beats FA3+cast+norm-gate
  by >= 8% (else that shape stays byte-identical to master); print the timing table. SAFETY
  (required): (1) build/compile the kernel(s) inside prepare() under try/except; (2) correctness
  self-check in prepare() against the incumbent (master's exact q.to(FP8) + FA3 + o_norm_gate
  sequence on the same synthetic qkv/cache) on: B=1 ctx=80000 L=753; B=4 ctx=80000 L=753; B=1
  ctx=0 L=300; B=2 ctx=2500 L=753; B=1 ctx=4096 L=4096; one page_size=64 case; one kv_scale != 1
  case; and the T=16384 case if it is to be routed; pass rule per case: rel L2 of the [T,1024]
  output vs the incumbent <= 2e-2, worst row <= 6e-2, no NaN/Inf, and on the two smallest cases
  rel L2 vs an fp32 torch reference no worse than 1.25x the incumbent's own; (3) any exception,
  failed case or non-finite output disables the new path for every shape and kernel_fn runs
  master's code unchanged; (4) print loud lines: 'PATH CHOSEN sliding-attn: triton-gqa (S=<splits>
  at T=<T>)' / 'PATH CHOSEN sliding-attn: FA3 fallback (<reason>)' per routed T, plus the
  self-check table (case, rel L2 vs incumbent, worst row, vs-fp32 ratio) and per-case kernel us.
  kernel_fn: no host syncs, no .item(), no branching on values; only torch.empty for the output
  (as master's GEMMs already allocate) and the preallocated split workspace; every launch is
  CUDA-graph capturable (grid from config only; per-request key counts come from seq_lens /
  cu_seqlens_q / positions on device). Precision: KV cache stays FP8 and is read as FP8, queries
  rounded to E4M3 exactly as today, activations BF16, no weight touched. Mechanism (master profile
  ba9c16b9cc6b4605): FA3 on sliding layers costs 25.3 us + 7.1 us FlashAttnFwdCombine at turn_1
  (96 per-head 128-row tiles < 132 SMs, ~17 serial 128-key tiles each, so it splits and pays a
  combine), 63.1 us at turn_4 and 300.6 us at sliding first_turn, plus the separate q FP8 cast and
  FusedRMSNormGate launches (2-5 us each at turn shapes); the compute/exp floor is ~6.4 us, ~25.6
  us and ~139 us (4*T*2048*16*64 FLOP at 989 TFLOP/s; T*2048*16 exps at ~3.9 T/s). GQA packing,
  4-way split at turn_1 (~5 key tiles per program instead of 17), and folding
  cast+combine+norm-gate into the kernel attack the latency chain and the launches that dominate
  the turn shapes.
expected:
  - shapes: "*layer=sliding load=turn_1*"
    direction: faster
    why: "incumbent sliding attention block ~37 us (FA3 25.3 + combine 7.1 + q cast + norm-gate) vs floor ~6.4 us; target <= 15 us: ~20-25 us of 1079-1094 us (~2%), ~1.3% on the turn_1 load (sliding is ~2/3 of it). Judge on the new kernel rows vs FA3+combine+FusedRMSNormGate rows"
  - shapes: "*layer=sliding load=turn_4*"
    direction: faster
    why: "incumbent ~70 us (FA3 63.1 + cast + norm-gate) vs floor ~25.6 us; target <= 40 us: ~30 us of 3335-3381 us (~0.9%)"
  - shapes: "*layer=sliding load=first_turn*"
    direction: faster
    why: "only if prepare() timing enables it (>= 8% margin): incumbent FA3 300.6 us + cast + norm-gate (~320 us) vs floor ~139 us (bf16 MMA and exp both ~139 us); target <= 270 us, ~50 us of 14275-14379 us (0.35%); if not enabled, byte-identical (check the PATH CHOSEN line)"
  - shapes: "*layer=full*"
    direction: flat
    why: full layers keep FA3 and master code byte-identical (control; also keeps this run disjoint from attn-full-prefill-cuda-fp8-exp-offload)
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)"
review: proceed
status: running
review_at: 2026-10-24T183433Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T183433Z
base_sha: c449be5350fc7f87e9daa73bc83715a041cff19e
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/sliding-attn-block-triton-gqa-fused-normgate
worktree: .worktrees/muse_prefill_layer/sliding-attn-block-triton-gqa-fused-normgate
mode: submit
implemented_at: 2026-10-24T184731Z
impl_commit: dd376f045ed7c6df58640da4091967acb44b4e17
submission_sha256: eb3f81db96dd4883127b7d7fe51a28623f0ff3fffd69a024798569ef03b43f04
norm_sha256: 53aaad2a4420b80e3945dfe26ff00789612febcaba26cb3c2e37c9589eddc3c3
submission_blob: 078e0578581a7a44fe8a8e9ba217559c9e4e4cf5
lint_warnings: 11
attempt_file: 2026-10-24T184732Z-submit-sliding-attn-block-triton-gqa-fused-normgate.md
prepared_at: 2026-10-24T184732Z
---

# sliding-attn-block-triton-gqa-fused-normgate

## Change

Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, experts unchanged). In kernel_fn, on sliding layers only (cfg['window'] != 0), replace the four steps between store_cache and the o projection -- q.view(tokens,16,64).to(FP8), flash_attn_with_kvcache(...) (FA3 + its FlashAttnFwdCombine split merge), and state['o_norm_gate'](out, gate) -- with a new Triton GQA-packed sliding flash-attention kernel _sliding_prefill_attn_kernel whose epilogue applies the norm-gate and writes the bf16 [T, 1024] input of the o GEMM directly. qkv GEMM, FusedQKNorm (q/k RMSNorm + interleaved RoPE: its q output is this kernel's input, so RoPE is NOT reimplemented and the tl.sin issue in LESSONS does not arise), the k/v FP8 cast and store_cache all stay byte-identical and run first, so the step's own K/V rows are read back from the paged FP8 cache exactly as FA3 reads them. Full layers, the o GEMM and everything after are untouched. KERNEL: one program = one KV head x BLOCK_TOK=16 consecutive tokens of one request x all 8 q heads sharing that KV head = 128 query rows (GQA packing: each gathered K/V row feeds 8 heads, half the K/V traffic and twice the MMA M of FA3's per-head tiles). Prologue: load q (bf16 from FusedQKNorm) and round to FP8 E4M3 then back to bf16 in registers (identical query precision to FA3's q.to(FP8); no coarser). Key range per program = [max(0, pos_first-2047), pos_last] in that request's positions (window_size (2047,0) semantics: token at p sees keys p-2047..p); walk it in BLOCK_N=64 key tiles; per tile gather K and V rows through page_table[b, kpos // PAGE] * PAGE + kpos % PAGE (PAGE = cfg['page_size'] constexpr, so the page-64 test works), load FP8, upcast to bf16, S = tl.dot(q, tl.trans(k)) in fp32 (try FP8xFP8 tl.dot for S first if it compiles in the judge's Triton, else bf16), per-element mask kpos <= qpos and kpos >= qpos-2047 (every in-window key is attended; only tiles wholly outside every row's window are not visited), online softmax in fp32 with sm_scale*qk_scale_factor*k_scale folded into exp2, P kept in BF16 (finer than FA3's FP8 P) for tl.dot(p, v), v_scale applied in the epilogue, num_stages 3 so Triton pipelines the K/V gather. Split-KV only where the grid is thin: S = 4 if B*ceil(L/16)*2 < 264 (turn_1: 96 programs x 4 splits), else S = 1 (turn_4: 384 programs; first_turn: 2048); S is chosen from config shapes only. S=1 epilogue: o = acc/l, then per (token, head) row the FusedRMSNormGate math (rmsnorm over the 64 dims, eps 1e-5, no weight, times sigmoid(gate) read from qkv[:, 1280 + head*64 : ...]) in fp32, store bf16 into out[T, 1024]. S>1: programs write unnormalized acc (fp32), m and l into a workspace preallocated in prepare() (sized for T<=4096, 4 splits, 16 heads), then one small Triton combine kernel merges the splits AND applies the same norm-gate, writing out[T,1024]. ROUTING: new path for sliding layers at every T <= 4096 (turn_1, turn_4 and all sliding tests); at T > 4096 (sliding first_turn) prepare() times both paths as captured CUDA graphs on a synthetic T=16384, ctx=32768 case (cold L2 via a 256 MiB flush write before each replay, 20 replays, median) and enables the new kernel there only if it beats FA3+cast+norm-gate by >= 8% (else that shape stays byte-identical to master); print the timing table. SAFETY (required): (1) build/compile the kernel(s) inside prepare() under try/except; (2) correctness self-check in prepare() against the incumbent (master's exact q.to(FP8) + FA3 + o_norm_gate sequence on the same synthetic qkv/cache) on: B=1 ctx=80000 L=753; B=4 ctx=80000 L=753; B=1 ctx=0 L=300; B=2 ctx=2500 L=753; B=1 ctx=4096 L=4096; one page_size=64 case; one kv_scale != 1 case; and the T=16384 case if it is to be routed; pass rule per case: rel L2 of the [T,1024] output vs the incumbent <= 2e-2, worst row <= 6e-2, no NaN/Inf, and on the two smallest cases rel L2 vs an fp32 torch reference no worse than 1.25x the incumbent's own; (3) any exception, failed case or non-finite output disables the new path for every shape and kernel_fn runs master's code unchanged; (4) print loud lines: 'PATH CHOSEN sliding-attn: triton-gqa (S=<splits> at T=<T>)' / 'PATH CHOSEN sliding-attn: FA3 fallback (<reason>)' per routed T, plus the self-check table (case, rel L2 vs incumbent, worst row, vs-fp32 ratio) and per-case kernel us. kernel_fn: no host syncs, no .item(), no branching on values; only torch.empty for the output (as master's GEMMs already allocate) and the preallocated split workspace; every launch is CUDA-graph capturable (grid from config only; per-request key counts come from seq_lens / cu_seqlens_q / positions on device). Precision: KV cache stays FP8 and is read as FP8, queries rounded to E4M3 exactly as today, activations BF16, no weight touched. Mechanism (master profile ba9c16b9cc6b4605): FA3 on sliding layers costs 25.3 us + 7.1 us FlashAttnFwdCombine at turn_1 (96 per-head 128-row tiles < 132 SMs, ~17 serial 128-key tiles each, so it splits and pays a combine), 63.1 us at turn_4 and 300.6 us at sliding first_turn, plus the separate q FP8 cast and FusedRMSNormGate launches (2-5 us each at turn shapes); the compute/exp floor is ~6.4 us, ~25.6 us and ~139 us (4*T*2048*16*64 FLOP at 989 TFLOP/s; T*2048*16 exps at ~3.9 T/s). GQA packing, 4-way split at turn_1 (~5 key tiles per program instead of 17), and folding cast+combine+norm-gate into the kernel attack the latency chain and the launches that dominate the turn shapes.

## Expected

- `*layer=sliding load=turn_1*` faster: incumbent sliding attention block ~37 us (FA3 25.3 + combine 7.1 + q cast + norm-gate) vs floor ~6.4 us; target <= 15 us: ~20-25 us of 1079-1094 us (~2%), ~1.3% on the turn_1 load (sliding is ~2/3 of it). Judge on the new kernel rows vs FA3+combine+FusedRMSNormGate rows
- `*layer=sliding load=turn_4*` faster: incumbent ~70 us (FA3 63.1 + cast + norm-gate) vs floor ~25.6 us; target <= 40 us: ~30 us of 3335-3381 us (~0.9%)
- `*layer=sliding load=first_turn*` faster: only if prepare() timing enables it (>= 8% margin): incumbent FA3 300.6 us + cast + norm-gate (~320 us) vs floor ~139 us (bf16 MMA and exp both ~139 us); target <= 270 us, ~50 us of 14275-14379 us (0.35%); if not enabled, byte-identical (check the PATH CHOSEN line)
- `*layer=full*` flat: full layers keep FA3 and master code byte-identical (control; also keeps this run disjoint from attn-full-prefill-cuda-fp8-exp-offload)

## Risk

Not a duplicate: attn-full-prefill-cuda-fp8-exp-offload (in flight) replaces FA3 on FULL layers only; experts-cuda-wgmma-mixed-input-grouped-gemm (in flight) touches only the experts. qk-norm-fp8-cache-store-fusion (neutral) fused the K/V front; this keeps that front as master has it and replaces the back (q cast, FA3, combine, norm-gate). Template: muse_attention_decode master's _sliding_fused_kernel / _online_update / _gate_epilogue (fused sliding decode with gate epilogue, compiles in the judge's Triton) generalised to 16-token x 8-head prefill tiles. Risks: Triton's paged gather at page_size 1 (LESSONS: Triton tops out ~2.1 TB/s on gathers; here per-program K/V is only ~(2048+16)*256 B so it is latency, not bandwidth); FP8xFP8 tl.dot may not compile (fall back to bf16 dot, 2x MMA time, still far from bound at turn shapes); first_turn may not beat FA3 (then it stays on FA3 by the timing gate). Time budget: ~1-1.5 h (kernel ~150 lines from the decode template, combine ~50, self-check ~60). Stage 2: first_turn sliding with a warp-specialized/persistent schedule if stage 1 kept FA3 there.

## Milestones

Rewrite stage 1 of `q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T183433Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound, rules-clean stage-1 rewrite of the sliding-layer attention back end. The turn_1 and turn_4 targets are physically reachable, and the T=16384 timing gate protects first_turn. No landed hypothesis confounds it, so no companions are needed. The main gap: the incumbent's q-cast and FusedRMSNormGate rows are hidden in the profile's '... 11 more', so prepare() should time the incumbent sequence beside the new one at the scored turn shapes.

### Justification
Mechanism is plausible. Master profile ba9c16b9cc6b4605 confirms FA3 at 25.3 us plus FlashAttnFwdCombine at 7.1 us on sliding turn_1, 63.1 us at sliding turn_4 and 300.6 us at sliding first_turn. At turn_1, 96 per-head 128-row tiles do not fill 132 SMs and the work is latency-chain bound. GQA packing gives 16 tokens x 8 heads = 128 MMA rows per program and halves K/V gathers; with the 4-way split at turn_1 that makes 384 programs of about 8 64-key tiles each, against about 17 serial 128-key tiles in FA3. The ~15 us turn_1 target is credible. The window K/V working set (about 0.7 MB cold) is L2-resident and per-program gathers are latency-bound, not bandwidth-bound, so the LESSONS 2.1 TB/s gather ceiling does not bind. Floors check out: 4*753*2048*16*64 = 6.3 GFLOP, which is 6.4 us at 989 TF, with exp count at the same order. The expected directions match TASK weights. turn_1 is sliding 46 + full 16, about 72.4 ms per load, so saving ~22 us x 46 is about 1.4% on that load; turn_4 is about 0.6% of its load; first_turn is flat or gated. The headline gain is therefore about 0.6-0.7%, inside the +/-1.5% CI, as the hypothesis itself says. That is acceptable because, as a rewrite, it is judged on the replaced kernel rows. Weakest link: the incumbent's 'q cast + FusedRMSNormGate 2-5 us each' is an estimate, not in the profile. Both rows fall into '... 11 more' at turn_1 and turn_4, as the new combine kernel likely will too, so milestone 2 (at or below the incumbent kernels' time) cannot be fully read from top_kernels without extra prints. Secondary weak links: (a) the 'combine folded in' saving at turn_1 is partly given back by the stage's own split-combine launch; (b) a Triton bf16-dot FA at first_turn will very likely lose to FA3, so the gate will keep FA3 there; that is expected and fine.

### Confounders on the current kernel
- sglang-pipeline-parity: not a confounder but the incumbent definition. It established the exact q.to(FP8) + paged FA3 (window (2047,0)) + FusedRMSNormGate sequence being replaced and used as the self-check oracle; nothing tunable in it alters this comparison.
- experts-triton-w4a16-first-turn-blockt256 and experts-hybrid-marlin-turn-triton-first-turn: touch only prepare()/_experts after the o GEMM. The timed replay starts with a cold L2 and the attention block runs strictly before the experts on the main stream, so they share no memory traffic, numerical path or launch adjacency with the replaced kernels. A change in attention kernel time is not caused by them. The only weak coupling is that master's prepare() spends time on expert self-checks, which shares the 300 s prepare and 15 min run budget with this hypothesis's T=16384 graph timing and self-checks.

### Interference and A/B needs
No companion is warranted. No landed hypothesis touches the sliding-attention back end, its inputs (qkv GEMM, FusedQKNorm, store_cache stay byte-identical) or its consumer (o GEMM unchanged). A gain or loss on the replaced rows is therefore attributable to the change itself. Full-layer shapes are a built-in control because they stay byte-identical. The first_turn sub-change is self-isolating through the prepare() timing gate and the PATH CHOSEN line. An ab-revert of a landed experts hypothesis would not change the attention rows, and a control splitting 'GQA kernel' from 'fused norm-gate epilogue' would cost a check to attribute about 2-5 us, which the next stage can do more cheaply. For a rewrite, the next stage is the follow-up.

### Open questions
- The incumbent q.to(FP8) and FusedRMSNormGate rows, and the new combine kernel, will likely be hidden in the profile's '... 11 more' at turn_1 and turn_4. Should prepare() also time master's exact sequence and the new path as captured graphs at the scored turn_1 (B=1 ctx=80000 L=753) and turn_4 (B=4) shapes (cold L2, median), printing both, so milestone 2 is readable from the log?
- Is the 2e-2 rel-L2-vs-incumbent pass bar safe, given that the incumbent uses FP8 P and the new kernel uses BF16 P? The difference after per-head RMSNorm and gating should be well under 1e-2, but if the bar trips spuriously the run silently becomes byte-identical to master and wastes a rewrite check. Print the margin.
- At turn_1 the 4-way split adds a combine launch. Was S=2 or S=1 with BLOCK_TOK=16 (96 programs) considered as an alternative recorded in prepare() timing? A small in-prepare sweep over S in {1,2,4} at the turn_1 shape would choose from data at no check cost.
- Does the profile show master's FA3 launching with PDL? Replacing it with non-PDL Triton launches could lose a few us of overlap with the neighbouring qk_norm/store_cache and o GEMM, which would show up as noise in the per-kernel rows.
- Budget: the T=16384 synthetic self-check plus graph timing with a 256 MiB flush x 20 replays for both paths, on top of master's existing expert checks. Confirm prepare() stays well under 300 s and the whole run under 15 min including the Triton compile.


## Implementation notes

All edits are in the worktree's `muse_prefill_layer/submission.py` (+570/-22, uncommitted). Imports added: `statistics`, `time`.

- **Master's back end, moved as is.** Master's `q.to(FP8)` + `flash_attn_with_kvcache` + `o_norm_gate` lines now live in `_attn_fa3(q, gate, data, cfg, state)`. The ops and arguments are unchanged (`k_cache` is read as `data["k_cache"]`). kernel_fn uses it for full layers and for every fallback, and prepare() uses it as the self-check incumbent.
- **New kernels:**
  - `_sliding_prefill_attn_kernel`: grid (B*ceil(L/16), HKV, S). Each program has 128 rows (16 tokens x 8 q heads). q is rounded to E4M3 in registers. K/V are gathered through the page table (`PAGE` is a constexpr). The mask is per element (`kpos <= qpos`, `kpos >= qpos-2047`), with online softmax in exp2 and P in bf16. When S=1 the epilogue applies the norm-gate. When S>1 the program writes fp32 partials to a workspace.
  - `_sliding_combine_kernel`: merges the S>1 partials and applies the norm-gate.
  - Helpers: `_online_update`, `_norm_gate_store`, `_sliding_rows`.
- **QK dot variant.** prepare() tries FP8xFP8 first (`QK_FP8=True`). If it raises or fails any self-check case, the bf16 dot is tried. If both fail, every shape runs master's path.
- **Routing (`_sliding_route`, config only).** Sliding layers with heads (16,2,64) and T <= 4096 take the new path. T > 4096 takes it only if the prepare() graph-timing gate passes (new <= 0.92x incumbent at T=16384, ctx=32768). S=4 when B*ceil(L/16)*2 < 264, else S=1. The split workspace (1056 partials, about 35 MB) is allocated in prepare().
- **kernel_fn fallbacks.** kernel_fn also falls back to `_attn_fa3` when the q rows are non-contiguous or when the launch raises on the host. Each case prints a `PATH CHOSEN` line once per shape (config key only).
- **prepare() self-check** (`_prepare_sliding`, wrapped in try/except, never raises). Synthetic cases (soft FP8 keys, scattered pages): turn_1, turn_4, ctx0 L300, B2 ctx2500 L753, ctx4096 L4096, page-64 (B3 ctx5000 L200), kv_scale=1/64. The T=16384 case is added only if the gate routes it.
  - Each case is compared with master's sequence and with an fp32 SDPA reference.
  - The pass rule needs finite output and rel L2 vs incumbent <= max(2e-2, 2x the incumbent's own fp32 error). The worst row must be <= max(6e-2, 3x the incumbent's worst row). These bars follow the incumbent's own FP8 noise so they cannot trip every time; the margin is printed.
  - On the two smallest cases the error vs the fp32 reference must also be <= 1.25x the incumbent's + 5e-4.
  - The table prints eager us for both paths per case.
- **Timing (reviewer's open question 1).** prepare() times both paths as CUDA graphs: cold L2 from a 256 MiB flush write, 20 replays, median. It prints both rows at turn_1 and turn_4, and the T=16384 gate row. The T=16384 gate is skipped (stays FA3) if prepare() has already taken 180 s.
- **Deviations:**
  - The tolerances are adaptive floors, as above, not fixed numbers.
  - No in-prepare S sweep over {1,2,4} (reviewer's open question 3). The caller asked for fixed configs, so S follows the spec.
  - PDL is not addressed.
  - `pt_stride`, `L` and `NTB` are `do_not_specialize`, so the variants compiled in prepare() cover the judge's shapes.
- **Uncertainties (nothing can run here):**
  - Whether FP8 tl.dot with `tl.trans` compiles in the judge's Triton. The bf16 dot is the fallback.
  - Register pressure with 4 warps on 128x64 tiles.
  - How long prepare() takes in total.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/sliding-attn-block-triton-gqa-fused-normgate diff`:

- [x] CUDA-graph capturable. The new kernel_fn code branches only on config, tensor strides and host exceptions. It allocates only `torch.empty` sized from config. It has no `.item()`, `.cpu()` or synchronize, and the print is a config string, once per shape. All grids come from config.
- [x] Nothing derived from `data` is cached. The module-level `_ANNOUNCED` set holds config tuples only. The `state["sliding"]` workspace is scratch that every call overwrites before reading.
- [x] Weights are untouched. The sliding path reads no weights.
- [x] Activations stay BF16, the KV cache stays FP8 and is read as FP8, and queries are rounded to E4M3 (RNE) exactly as `q.to(FP8)`, no coarser. P is bf16, which is finer than FA3's FP8.
- [x] Every key is attended: the per-element window mask covers p-2047..p and every tile in the program's key range is visited. There is no top-k, sparse, block-skip or weight-based skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. Synthetic cases are built in-file.
- [x] All work happens inside the timed call. There is no output caching or replay; the prepare() timing and self-checks use synthetic data only.
- [x] Full layers are byte-identical to master: `_attn_fa3` holds master's lines verbatim.

Lint (`okc_lint.py ... --problem muse_prefill_layer`):
```
warn  host-sync    L465/L507/L909/L913/L937/L1093/L1099/L1106/L1115: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L896/L921: end.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
py_compile: OK (pycache removed).

## Pre-flight audit

Audited 2026-10-24 by an Opus subagent (read-only review, plus numpy emulation; no GPU here). **Verdict: dispatch.** No fixes were needed. Lint passes (only prepare()-side host-sync warnings) and py_compile is clean.

- **Gates.** A numpy emulation of the synthetic case (soft keys, unit-RMS q, qk_scale 5.47) gives these errors:
  - incumbent (FP8 q + FP8 P) vs fp32: ~6.8%;
  - new kernel (FP8 q + bf16 P) vs fp32: ~6.7%;
  - new kernel vs incumbent: ~1.5%, worst row ~2%.

  The bars are max(2e-2, 2x the incumbent's error) for rel L2 and max(6e-2, 3x) for the worst row, so the margin is about 9x. The enforced fp32 ratio comes out ~0.98, against a bar of 1.25. Those two enforced cases (ctx0 L300 S=4, and the page-64 ctx5000 L200 case with the window edge) are tight enough to catch masking, gate-offset, head-map and k_scale bugs.
- **Fallbacks.** The cascade is FP8 dot, then bf16 dot, then master. A compile error or a self-check miss lands on the next path. Anything in `_prepare_sliding` that raises is caught in prepare(), and kernel_fn falls back to master's verbatim `_attn_fa3` on a route miss or a host-side launch error. `PATH CHOSEN` lines print in both prepare() and kernel_fn.
- **Correctness.** A numpy emulation of the program's row mapping, its lo/hi/chunk split arithmetic, the per-element window mask (`t >= qpos-2047`), online softmax with empty or -inf splits, and the S=4 combine matches direct windowed attention to 1e-15. Cases covered: ctx 0/100/2047/2500/5000; L 17/33/37/200/300/2100; S 1 and 4. Gate offset 1280, cache flat-row indexing, page-table gather and int64 offsets all check out. Loads are masked or clamped, so nothing reads out of bounds.
- **Budget.** prepare() runs once per run (the master log prints once). The sliding work adds about 4 Triton variants (about 8 if FP8 fails), seven small synthetic cases, sub-second fp32 references and graph timings, and the T=16384 gate (skipped past 180 s). Estimate: ~30-100 s on top of master, well inside 300 s and 15 min.
- **Rules.** kernel_fn branches only on config and strides. It allocates only a config-sized `torch.empty`; the workspace comes from prepare() and is overwritten every call. It has no syncs. KV is read as FP8, q is rounded to E4M3 (RNE), P is bf16, and every in-window key is visited.

Residual risk: a hard device fault or compiler crash in the new kernel (not catchable) would fail the run instead of falling back. The judge's real FusedQKNorm q stride differs from the self-check's contiguous q, but the specialization is the same. Its correctness is checked only by the judge's tests.
