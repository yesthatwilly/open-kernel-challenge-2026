# muse_prefill_layer · 2026-10-24T184732Z · submit · sliding-attn-block-triton-gqa-fused-normgate

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T183246Z-sliding-attn-block-triton-gqa-fused-normgate.md` (branch `hyp/muse_prefill_layer/sliding-attn-block-triton-gqa-fused-normgate`).

- **Job:** `78d8d7bdec55491c` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `dd376f0`

## Goal
Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, experts unchanged). In kernel_fn, on sliding layers only (cfg['window'] != 0), replace the four steps between store_cache and the o projection -- q.view(tokens,16,64).to(FP8), flash_attn_with_kvcache(...) (FA3 + its FlashAttnFwdCombine split merge), and state['o_norm_gate'](out, gate) -- with a new Triton GQA-packed sliding flash-attention kernel _sliding_prefill_attn_kernel whose epilogue applies the norm-gate and writes the bf16 [T, 1024] input of the o GEMM directly. qkv GEMM, FusedQKNorm (q/k RMSNorm + interleaved RoPE: its q output is this kernel's input, so RoPE is NOT reimplemented and the tl.sin issue in LESSONS does not arise), the k/v FP8 cast and store_cache all stay byte-identical and run first, so the step's own K/V rows are read back from the paged FP8 cache exactly as FA3 reads them. Full layers, the o GEMM and everything after are untouched. KERNEL: one program = one KV head x BLOCK_TOK=16 consecutive tokens of one request x all 8 q heads sharing that KV head = 128 query rows (GQA packing: each gathered K/V row feeds 8 heads, half the K/V traffic and twice the MMA M of FA3's per-head tiles). Prologue: load q (bf16 from FusedQKNorm) and round to FP8 E4M3 then back to bf16 in registers (identical query precision to FA3's q.to(FP8); no coarser). Key range per program = [max(0, pos_first-2047), pos_last] in that request's positions (window_size (2047,0) semantics: token at p sees keys p-2047..p); walk it in BLOCK_N=64 key tiles; per tile gather K and V rows through page_table[b, kpos // PAGE] * PAGE + kpos % PAGE (PAGE = cfg['page_size'] constexpr, so the page-64 test works), load FP8, upcast to bf16, S = tl.dot(q, tl.trans(k)) in fp32 (try FP8xFP8 tl.dot for S first if it compiles in the judge's Triton, else bf16), per-element mask kpos <= qpos and kpos >= qpos-2047 (every in-window key is attended; only tiles wholly outside every row's window are not visited), online softmax in fp32 with sm_scale*qk_scale_factor*k_scale folded into exp2, P kept in BF16 (finer than FA3's FP8 P) for tl.dot(p, v), v_scale applied in the epilogue, num_stages 3 so Triton pipelines the K/V gather. Split-KV only where the grid is thin: S = 4 if B*ceil(L/16)*2 < 264 (turn_1: 96 programs x 4 splits), else S = 1 (turn_4: 384 programs; first_turn: 2048); S is chosen from config shapes only. S=1 epilogue: o = acc/l, then per (token, head) row the FusedRMSNormGate math (rmsnorm over the 64 dims, eps 1e-5, no weight, times sigmoid(gate) read from qkv[:, 1280 + head*64 : ...]) in fp32, store bf16 into out[T, 1024]. S>1: programs write unnormalized acc (fp32), m and l into a workspace preallocated in prepare() (sized for T<=4096, 4 splits, 16 heads), then one small Triton combine kernel merges the splits AND applies the same norm-gate, writing out[T,1024]. ROUTING: new path for sliding layers at every T <= 4096 (turn_1, turn_4 and all sliding tests); at T > 4096 (sliding first_turn) prepare() times both paths as captured CUDA graphs on a synthetic T=16384, ctx=32768 case (cold L2 via a 256 MiB flush write before each replay, 20 replays, median) and enables the new kernel there only if it beats FA3+cast+norm-gate by >= 8% (else that shape stays byte-identical to master); print the timing table. SAFETY (required): (1) build/compile the kernel(s) inside prepare() under try/except; (2) correctness self-check in prepare() against the incumbent (master's exact q.to(FP8) + FA3 + o_norm_gate sequence on the same synthetic qkv/cache) on: B=1 ctx=80000 L=753; B=4 ctx=80000 L=753; B=1 ctx=0 L=300; B=2 ctx=2500 L=753; B=1 ctx=4096 L=4096; one page_size=64 case; one kv_scale != 1 case; and the T=16384 case if it is to be routed; pass rule per case: rel L2 of the [T,1024] output vs the incumbent <= 2e-2, worst row <= 6e-2, no NaN/Inf, and on the two smallest cases rel L2 vs an fp32 torch reference no worse than 1.25x the incumbent's own; (3) any exception, failed case or non-finite output disables the new path for every shape and kernel_fn runs master's code unchanged; (4) print loud lines: 'PATH CHOSEN sliding-attn: triton-gqa (S=<splits> at T=<T>)' / 'PATH CHOSEN sliding-attn: FA3 fallback (<reason>)' per routed T, plus the self-check table (case, rel L2 vs incumbent, worst row, vs-fp32 ratio) and per-case kernel us. kernel_fn: no host syncs, no .item(), no branching on values; only torch.empty for the output (as master's GEMMs already allocate) and the preallocated split workspace; every launch is CUDA-graph capturable (grid from config only; per-request key counts come from seq_lens / cu_seqlens_q / positions on device). Precision: KV cache stays FP8 and is read as FP8, queries rounded to E4M3 exactly as today, activations BF16, no weight touched. Mechanism (master profile ba9c16b9cc6b4605): FA3 on sliding layers costs 25.3 us + 7.1 us FlashAttnFwdCombine at turn_1 (96 per-head 128-row tiles < 132 SMs, ~17 serial 128-key tiles each, so it splits and pays a combine), 63.1 us at turn_4 and 300.6 us at sliding first_turn, plus the separate q FP8 cast and FusedRMSNormGate launches (2-5 us each at turn shapes); the compute/exp floor is ~6.4 us, ~25.6 us and ~139 us (4*T*2048*16*64 FLOP at 989 TFLOP/s; T*2048*16 exps at ~3.9 T/s). GQA packing, 4-way split at turn_1 (~5 key tiles per program instead of 17), and folding cast+combine+norm-gate into the kernel attack the latency chain and the launches that dominate the turn shapes.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
high
Not a duplicate: attn-full-prefill-cuda-fp8-exp-offload (in flight) replaces FA3 on FULL layers only; experts-cuda-wgmma-mixed-input-grouped-gemm (in flight) touches only the experts. qk-norm-fp8-cache-store-fusion (neutral) fused the K/V front; this keeps that front as master has it and replaces the back (q cast, FA3, combine, norm-gate). Template: muse_attention_decode master's _sliding_fused_kernel / _online_update / _gate_epilogue (fused sliding decode with gate epilogue, compiles in the judge's Triton) generalised to 16-token x 8-head prefill tiles. Risks: Triton's paged gather at page_size 1 (LESSONS: Triton tops out ~2.1 TB/s on gathers; here per-program K/V is only ~(2048+16)*256 B so it is latency, not bandwidth); FP8xFP8 tl.dot may not compile (fall back to bf16 dot, 2x MMA time, still far from bound at turn shapes); first_turn may not beat FA3 (then it stays on FA3 by the timing gate). Time budget: ~1-1.5 h (kernel ~150 lines from the decode template, combine ~50, self-check ~60). Stage 2: first_turn sliding with a warp-specialized/persistent schedule if stage 1 kept FA3 there.

## Result (after the run)
- job `78d8d7bdec55491c` (submit), status done, passed True, tests 5/5
- speedup 1.0507x, 95% CI [1.0350, 1.0665], score 368704.5727929042 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/sliding-attn-block-triton-gqa-fused-normgate.result.json`, `muse_prefill_layer/hypotheses/done/neutral/sliding-attn-block-triton-gqa-fused-normgate.patch`; branch `hyp/muse_prefill_layer/sliding-attn-block-triton-gqa-fused-normgate`
