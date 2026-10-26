---
problem: muse_prefill_layer
slug: boundary-overlap-s1-slice-kernel-ffn-tail
created: 2026-10-25T152554Z
base: master
area: FFN boundary + unpooler GEMM (boundary chunks overlapped under unpooler chunks via a slice-writing boundary kernel)
change: >
  Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two
  FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kernel<8192, 512>: 399-402 us
  EACH at T=16384, 76-78 us at T=3012, 21-22 us at T=753; at T=16384 each moves ~1.61 GB = 335 us
  at 4.8 TB/s, so they already run at ~4.0 TB/s and a faster re-implementation cannot pay:
  norm-boundaries-persistent-multirow's Triton variants were 1.17-1.39x SLOWER) OFF the critical
  path, by overlapping them with the compute-bound nvjet GEMMs next to them. That needs a boundary
  kernel that writes into ROW SLICES of preallocated outputs, which the SGLang module (it
  allocates its outputs) cannot do. STAGE 1 (this hypothesis) = (a) the kernel and (b) the FFN
  tail at T > 4096 only. (a) A CUDA kernel built with torch.utils.cpp_extension.load_inline as its
  own small module (extra_cuda_cflags must include -std=c++20 and -O3, plain sm_90 is enough: no
  wgmma; -lineinfo), one function bnd_rows(stream_in fp32 [T,8192], branch bf16 [T,8192], row0,
  nrows, coef tensors, eps values, out_next bf16 [T,8192], out_stream fp32 [T,8192]) on the
  current stream: one CTA per row, 512 threads, 16 elements per thread, every global load of the
  row issued up front as 16-byte vectors (4 x float4 of stream, 2 x uint4 of branch), warp-shuffle
  + shared-memory block reductions, the same two reductions SGLang's boundary does: post-norm of
  the branch (eps POST_NORM_EPS, with the post_ffn (1 + w) input scale for the FFN boundary, none
  for the attention one), the norm-preserving residual mix with the module's precomputed
  coefficients, then the pre-norm of the new stream ((1 + w), RMS_EPS) into out_next. Take the
  exact formulas and the coefficient tensors from master's modules (state['_modules'],
  state['ffn_boundary'] / state['attn_boundary'] and their residual gates' precomputed
  coefficients) and from the Triton port in
  muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.patch, which
  already reproduced the math (stream err/residual 1.7-2.8e-3, out rel L2 1.7-2.3e-3 vs the
  module). Coefficients are weight-derived: build them in prepare into state, never module-global;
  no device pointers cached across calls. (b) FFN tail at T > MARLIN_MAX_TOKENS (first_turn):
  expert_output -> unpooler -> ffn boundary becomes, for NCH chunks of T/NCH rows (NCH in {2, 4},
  picked in prepare): main stream torch.mm(expert_output[rc], dense['unpooler'].t(),
  out=ffn_output[rc]) (contiguous row slices of a preallocated [T,8192] bf16), then
  bnd_stream.wait_stream(main) and, on a new per-state stream state['bnd_stream'],
  bnd_rows(stream, ffn_output, rc, ..., out_next=next_attn_input, out_stream=stream_out); after
  the loop main.wait_stream(bnd_stream); outputs preallocated once per call with torch.empty
  (shape-sized). So boundary chunk c runs under unpooler chunk c+1 (nvjet tiles are a
  non-persistent grid, so the block scheduler interleaves the two, as it does for the router under
  the pooler: 172 us of overlap in profile 5). Outside capture, record_stream every tensor the
  bnd_stream touches. T <= 4096 and the attention boundary stay master in stage 1. FALLBACK GATE
  in prepare (budget <= 40 s including the ~25 s build; master's prepare is 124 s of its own 200 s
  budget): (1) parity: bnd_rows on random rows at T=753 and 16384 vs state['ffn_boundary'] and
  state['attn_boundary']: residual-update rel err <= 5e-3 and next-input rel L2 <= 3e-3, worst row
  <= 5x that, no NaN; run twice alternating two separately allocated states; (2) kernel speed: one
  bnd_rows over all rows at T=16384 vs the module, graph replay, flushed, interleaved: print the
  ratio (milestone M1 below); (3) the tail: graphs of master's tail (unpooler + ffn boundary) vs
  the chunked tail at NCH 2 and 4 on synthetic bf16 expert_output at T=16384, 15 interleaved
  alternations, 256 MiB flush before each replay, empty-graph baseline subtracted; ON with the
  faster NCH only if <= 0.95x master. Any build failure, exception or a slow ratio leaves master
  byte-identical. Print '[bnd] ...' lines and a '[bnd] PATH CHOSEN' line after rega's. LATER
  STAGES (not this one): stage 2 the attention side at T > 4096 (o GEMM -> attention boundary
  chunks on bnd_stream -> pooler chunks with torch.mm(out=expert_input[rc]) waiting per chunk,
  router waiting on the last chunk), stage 3 extend both to turn_4 (T=3012, NCH 2) if stage 1-2
  show the overlap is real.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "incumbent FFN tail at T=16384 = unpooler (nvjet 192x192 share, ~1400 us) + ffn boundary 399-402 us serial; floor of the tail ~ the unpooler alone plus one exposed chunk of boundary (~100-200 us). Target -100 to -250 us per first_turn row (sliding 11120 us: 0.9-2.2%). Milestones: M1 bnd_rows <= 1.05x the SGLang kernel at T=16384 (prepare line); M2 chunked tail <= 0.92x master's tail in the prepare graph timing; M3 the judge's first_turn rows faster with the nvjet unpooler rows + norm_residual rows summing above the shape-time drop (overlap visible as GPU busy < kernel sum)"
  - shapes: "*load=turn_*"
    direction: flat
    why: "T <= 4096 keeps master's tail byte-identical in stage 1"
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "FusedPostNormResidualPreNorm (muse_spark_norm_residual_kernel) x2 on the critical path after the o and unpooler GEMMs at T > 4096 (stage 1: the FFN boundary)"
review: proceed
status: running
review_at: 2026-10-25T212333Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T212552Z
base_sha: cc874e4f97ac53e4c4a00078d39a8ff28d8316aa
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/boundary-overlap-s1-slice-kernel-ffn-tail
worktree: .worktrees/muse_prefill_layer/boundary-overlap-s1-slice-kernel-ffn-tail
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T011443Z
impl_commit: be87808942711d08b3625dbd3ecd6b39455c3d4f
submission_sha256: 36bea92e834c646f4af8fedc308047c8e68297281cf61b8d82c38904b4df973a
norm_sha256: 93d2e9656ab3256c3ac85f2ffb2f115230d5eb030e59112669246a8c5af6e36b
submission_blob: d24dd7d968891f49c1073544913df9dc158131b9
lint_warnings: 104
attempt_file: 2026-10-26T012540Z-check-boundary-overlap-s1-slice-kernel-ffn-tail.md
prepared_at: 2026-10-26T012540Z
job: 6962f6c6af3d4e5b
dispatched_at: 2026-10-26T012946Z
---

# boundary-overlap-s1-slice-kernel-ffn-tail

## Change

Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kernel<8192, 512>: 399-402 us EACH at T=16384, 76-78 us at T=3012, 21-22 us at T=753; at T=16384 each moves ~1.61 GB = 335 us at 4.8 TB/s, so they already run at ~4.0 TB/s and a faster re-implementation cannot pay: norm-boundaries-persistent-multirow's Triton variants were 1.17-1.39x SLOWER) OFF the critical path, by overlapping them with the compute-bound nvjet GEMMs next to them. That needs a boundary kernel that writes into ROW SLICES of preallocated outputs, which the SGLang module (it allocates its outputs) cannot do. STAGE 1 (this hypothesis) = (a) the kernel and (b) the FFN tail at T > 4096 only. (a) A CUDA kernel built with torch.utils.cpp_extension.load_inline as its own small module (extra_cuda_cflags must include -std=c++20 and -O3, plain sm_90 is enough: no wgmma; -lineinfo), one function bnd_rows(stream_in fp32 [T,8192], branch bf16 [T,8192], row0, nrows, coef tensors, eps values, out_next bf16 [T,8192], out_stream fp32 [T,8192]) on the current stream: one CTA per row, 512 threads, 16 elements per thread, every global load of the row issued up front as 16-byte vectors (4 x float4 of stream, 2 x uint4 of branch), warp-shuffle + shared-memory block reductions, the same two reductions SGLang's boundary does: post-norm of the branch (eps POST_NORM_EPS, with the post_ffn (1 + w) input scale for the FFN boundary, none for the attention one), the norm-preserving residual mix with the module's precomputed coefficients, then the pre-norm of the new stream ((1 + w), RMS_EPS) into out_next. Take the exact formulas and the coefficient tensors from master's modules (state['_modules'], state['ffn_boundary'] / state['attn_boundary'] and their residual gates' precomputed coefficients) and from the Triton port in muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.patch, which already reproduced the math (stream err/residual 1.7-2.8e-3, out rel L2 1.7-2.3e-3 vs the module). Coefficients are weight-derived: build them in prepare into state, never module-global; no device pointers cached across calls. (b) FFN tail at T > MARLIN_MAX_TOKENS (first_turn): expert_output -> unpooler -> ffn boundary becomes, for NCH chunks of T/NCH rows (NCH in {2, 4}, picked in prepare): main stream torch.mm(expert_output[rc], dense['unpooler'].t(), out=ffn_output[rc]) (contiguous row slices of a preallocated [T,8192] bf16), then bnd_stream.wait_stream(main) and, on a new per-state stream state['bnd_stream'], bnd_rows(stream, ffn_output, rc, ..., out_next=next_attn_input, out_stream=stream_out); after the loop main.wait_stream(bnd_stream); outputs preallocated once per call with torch.empty (shape-sized). So boundary chunk c runs under unpooler chunk c+1 (nvjet tiles are a non-persistent grid, so the block scheduler interleaves the two, as it does for the router under the pooler: 172 us of overlap in profile 5). Outside capture, record_stream every tensor the bnd_stream touches. T <= 4096 and the attention boundary stay master in stage 1. FALLBACK GATE in prepare (budget <= 40 s including the ~25 s build; master's prepare is 124 s of its own 200 s budget): (1) parity: bnd_rows on random rows at T=753 and 16384 vs state['ffn_boundary'] and state['attn_boundary']: residual-update rel err <= 5e-3 and next-input rel L2 <= 3e-3, worst row <= 5x that, no NaN; run twice alternating two separately allocated states; (2) kernel speed: one bnd_rows over all rows at T=16384 vs the module, graph replay, flushed, interleaved: print the ratio (milestone M1 below); (3) the tail: graphs of master's tail (unpooler + ffn boundary) vs the chunked tail at NCH 2 and 4 on synthetic bf16 expert_output at T=16384, 15 interleaved alternations, 256 MiB flush before each replay, empty-graph baseline subtracted; ON with the faster NCH only if <= 0.95x master. Any build failure, exception or a slow ratio leaves master byte-identical. Print '[bnd] ...' lines and a '[bnd] PATH CHOSEN' line after rega's. LATER STAGES (not this one): stage 2 the attention side at T > 4096 (o GEMM -> attention boundary chunks on bnd_stream -> pooler chunks with torch.mm(out=expert_input[rc]) waiting per chunk, router waiting on the last chunk), stage 3 extend both to turn_4 (T=3012, NCH 2) if stage 1-2 show the overlap is real.

## Expected

- `*load=first_turn*` faster: incumbent FFN tail at T=16384 = unpooler (nvjet 192x192 share, ~1400 us) + ffn boundary 399-402 us serial; floor of the tail ~ the unpooler alone plus one exposed chunk of boundary (~100-200 us). Target -100 to -250 us per first_turn row (sliding 11120 us: 0.9-2.2%). Milestones: M1 bnd_rows <= 1.05x the SGLang kernel at T=16384 (prepare line); M2 chunked tail <= 0.92x master's tail in the prepare graph timing; M3 the judge's first_turn rows faster with the nvjet unpooler rows + norm_residual rows summing above the shape-time drop (overlap visible as GPU busy < kernel sum)
- `*load=turn_*` flat: T <= 4096 keeps master's tail byte-identical in stage 1

## Risk

Stage 1 of a rewrite of the boundary placement (not of the boundary math): judge it on M1/M2 in the prepare log and the replaced rows (FFN-boundary muse_spark_norm_residual_kernel + unpooler nvjet) at first_turn, not on the headline. Main risks: (1) the overlap may not materialise if the boundary CTAs (512 threads) cannot co-reside with nvjet CTAs and only fill tile-wave tails; the 0.95x gate then keeps master and the stage still delivers M1 (a slice-writing boundary kernel at SGLang speed), which stage 2 needs anyway; (2) chunking the unpooler into M=4096-8192 GEMMs costs wave-quantization tails (cuBLAS may pick another tile); NCH 2 limits that. CUDA build hygiene from LESSONS: -std=c++20 when overriding nvcc flags, grep the log for the build line and PATH CHOSEN, sync+print after each first launch in the self-check so a fault is attributed, first run as a check. No FP8 or precision change: fp32 stream, bf16 outputs as SGLang.

## Milestones

Rewrite stage 1 of `FusedPostNormResidualPreNorm (muse_spark_norm_residual_kernel) x2 on the critical path after the o and unpooler GEMMs at T > 4096 (stage 1: the FFN boundary)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212333Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage 1: moves the FFN boundary (399-402 us at T=16384, ~1.2x its byte floor) off the critical path rather than making it faster (unlike norm-boundaries-persistent-multirow). Its prepare gate is its own A/B, so no companion. Weakest link: whether a 512-thread boundary CTA can share an SM with nvjet 192x192 CTAs at all; if not, overlap is only SM-level interleaving and wave tails, and M2 may gate OFF.

### Justification
Boundary kernels 813 us at first_turn (~1.2x memory floor); ~1.54-1.61 GB per boundary checks out, so a faster rewrite cannot pay (norm-boundaries-persistent-multirow neutral, Do-not-try); hiding is a different mechanism. nvjet GEMMs ~78% of peak (compute-bound), a good pairing. Missing prior field: norm-boundaries-persistent-multirow, qkv-gate-columns-side-stream-under-attention (1.4512 flat), turn-align-remap-on-side-stream-under-pooler, moe-align-side-stream. WEAKEST LINK: nvjet 192x192x64 4-stage coop likely fills smem and registers, so a 512-thread bnd_rows CTA may not co-reside; overlap then relies on the block scheduler handing retiring SMs to boundary CTAs. NCH=2 leaves the last ~200 us chunk exposed; chunking adds wave tails and may change the cuBLAS tile at M=4096/8192. Realistic -100..-250 us per first_turn row (~0.3-0.7% headline), below the 1.24% margin: read on M1/M2 and replaced rows. Prepare budget is 300 s (not 200); base is now f4751fc0 (kernel_fn tail unchanged).

### Confounders on the current kernel
- sglang-pipeline-parity: defines state['ffn_boundary'] and the dispatch unpooler call replaced here; parity tolerances against that module.
- experts-turn-rega-s4-first-turn-n128 / combo-a-master-normgather-naive-stack: rega's L2 footprint precedes the unpooler in the judge, not in the flushed prepare gate (M2 vs M3 may differ).
- router side-stream overlap (stack-tonight-five-prefill lineage): only in-layer precedent for block-scheduler overlap; no direct interaction.

### Interference and A/B needs
No companion: in-process captured-graph A/B (master tail vs chunked tail at NCH 2/4, flushed, interleaved, empty-graph baseline subtracted), M1 print, PATH CHOSEN. Add inside the gate: two interleaved rounds with an IQR bound; a high-priority bnd_stream (torch.cuda.Stream(priority=-1)) and whether capture keeps priority; a capped-grid persistent bnd_rows variant (16-32 CTAs) vs one CTA per row.

### Open questions
- Print cudaOccupancy for bnd_rows and the nvjet kernel's smem/regs in prepare.
- Log chunked GEMM kernel names (does the 192x192 tile survive at M=8192/4096?).
- Add prior (norm-boundaries-persistent-multirow, qkv-gate-columns-side-stream-under-attention); fix base blob f4751fc0 and prepare budget 300 s.
- Stage 2 (attention boundary under the pooler) competes with the router side stream; settle before minting it.


## Implementation notes

Edited only `muse_prefill_layer/submission.py` in the worktree (+~770 lines: one block before `kernel_fn`, a 5-line hook at the end of `prepare()`, a 10-line routed branch in `kernel_fn`).

- **Kernel** `bnd_rows` (own `load_inline` module `muse_prefill_bnd_v1`, kernel symbol `muse_prefill_bnd_rows_kernel`, flags `-O3 -std=c++20 -lineinfo`, `TORCH_CUDA_ARCH_LIST=9.0` around the build, restored after): 512 threads, 16 elements/thread, all row loads (2 x uint4 branch, 4 x float4 stream) issued first, two block reductions (separate smem slots), post-norm (x wpost when given) -> alpha/beta mix -> pre-norm, writes fp32 stream and bf16 next input into ROW SLICES (row0, nrows) of preallocated outputs. `grid` arg: 0 = one CTA per row, else that many persistent CTAs (the review's capped-grid variant). Host checks dtype/shape/contiguity/16-B alignment/row range; prints regs, local (spill) bytes, static smem, occupancy.
- **Coefficients** (per state, never module-global): alpha/beta probed from `state['attn_boundary']`/`['ffn_boundary']` and cross-checked against the reference formula (ported from the neutral norm-boundaries-persistent-multirow patch), wpost/wpre from the modules' effective weights.
- **Tail** `_bnd_tail`: NCH chunks; `torch.mm(expert_output[rc], unpooler.t(), out=ffn_output[rc])` on the main stream, then `bnd_stream.wait_stream(main)` and `bnd_rows` on the per-state side stream; `main.wait_stream(bnd_stream)` after the loop; outputs `torch.empty` per call; `record_stream` only outside capture.
- **Prepare gate** (`_bnd_setup`, runs LAST after rega-lo, start deadline 215 s, phase deadline 270 s, each skip prints `skipped: deadline`): source audit, build (time printed), attributes/occupancy, coefficient probe, parity at T=753 and 16384 on both boundaries (two separately allocated sets alternated over 4 runs, two row chunks with different grids, sentinel check that rows outside a chunk are untouched, whole-matrix and worst-row bounds 5e-3 / 3e-3 / 5x, no NaN), M1 print (kernel vs module, graph, flushed, two rounds), tail gate, a two-state captured-graph replay check with refilled inputs. Compact summary lines are printed again as the last lines of prepare.
- **Tail gate**: master = its ACTUALLY ROUTED tail (`_gemm` unpooler + `ffn_boundary`, no clone) vs 12 candidates (NCH 2/4 x grid 0/32/16 x side-stream priority 0/-1), eager correctness first, then 2 interleaved rounds x 15 alternations, 256 MiB flush, empty-graph baseline subtracted. ON only if the best candidate is <= 0.95x master in BOTH rounds and candidate IQR <= 1.5x master's in both. All rounds/IQRs/kernel names printed even when master is kept. Flag is per state (`state['bnd_ok']`); a miss is master's exact path and touches no other flag.

Review fixes applied: (1) bnd_rows occupancy/regs/spills printed; the nvjet kernel's smem/regs are not queryable in-process (closed cuBLAS binary), so the profiler prints the chunked and master GEMM kernel names instead (does the 192x192 tile survive at M=8192/4096). (2) Two interleaved rounds with the IQR bound. (3) High-priority `bnd_stream` (priority -1) is a candidate dimension, so the A/B shows whether capture keeps priority. (4) Capped-grid persistent bnd_rows (16/32 CTAs) vs one CTA per row is a candidate dimension. (5) Base blob f4751fc0 and the 300 s prepare budget are what the code assumes (deadlines are in the 300 s frame). The stage-2 open question is not build-relevant.

Deviations: routing is T == 16384 exactly (the gated, scored first_turn shape), not every T > 4096, because only 16384 is timed in prepare; other T stay on master's tail. Candidate selection extends the spec's NCH choice with grid and priority per the review.

## Rules check

- [x] CUDA-graph capturable: `_bnd_tail` has no sync/alloc-on-values/branching on values (shape checks only); event waits via `wait_stream`; `record_stream` skipped while capturing; the gate itself captures and replays it.
- [x] Nothing derived from `data` is cached: only weight-derived coefficients, the extension handle and a stream in `state`; no device pointers cached, no module-global tensors.
- [x] Weights: nothing repacked or re-stored; coefficients are fp32 [8192] vectors derived from the norm modules; unpooler used as a transposed view of the bf16 weight.
- [x] Activations: stream fp32, branch/outputs bf16 exactly as SGLang's boundary; no precision change.
- [x] Attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (torch.profiler in prepare only, as muse_bf16_proj does).
- [x] All work inside the timed call; no output caching or replay.
- [x] Prepare budget: own start/phase deadlines; the build and gate fall back to master on any error, never raise.

okc_lint: `OK with warnings` (only host-sync warnings in prepare()-reachable code and pre-existing dtype warnings); `py_compile` ok; no __pycache__ left.

## Pre-flight audit

Verdict: dispatch-with-fixes.
- Capture safety: _bnd_tail forks with bnd_stream.wait_stream(main) per chunk and joins with main.wait_stream(bnd_stream); no host sync, outputs are shape-sized torch.empty, record_stream only outside capture (redundant but harmless since main joins). The side stream is created per prepare() in _bnd_setup and stored per state; graphs only share it sequentially, which is legal.
- Partition: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/bnd_rows_partition_emulation.py (numpy) shows every row written exactly once and every column covered once by the 512x2x8 map, at T=16384 and T=753 (376+377 and 188x3+189 boundary chunks), nch 2/4, grid 0/32/16, and the parity halves.
- Routing: diff vs master is three hunks (hook at end of prepare, the new block, the T == 16384 branch in kernel_fn); every other T/shape runs master's tail unchanged.
- Fix: deadlines tightened (start 215 -> 190 s, gate 270 -> 250 s). Master's prepare ends at ~140 s (archive combo-b log), build+gates ~60 s, so this is still ample, but it keeps the phases inside 300 s if the harness clock also counts the import.
- Parity margin note: the equivalent Triton port measured out rel L2 2.3e-3 (FFN, T=753) vs the 3e-3 gate; deterministic, passes, but thin.
Residual risk: out rel L2 gate (3e-3) sits close to the measured 2.3e-3 bf16-rounding gap vs the module, and the 192x192 nvjet tile may not survive at M=T/NCH, so the tail gate may simply keep master.
