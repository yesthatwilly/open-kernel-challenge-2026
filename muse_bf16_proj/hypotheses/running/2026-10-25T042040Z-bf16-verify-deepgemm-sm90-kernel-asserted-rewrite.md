---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-sm90-kernel-asserted-rewrite
created: 2026-10-25T042040Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
change: >
  Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm
  for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT ever happened), yet every m>=128
  profile row was nvjet: DeepGEMM's own sm90 BF16 kernel never ran. This stage makes it run and
  proves it. STEP A (import/prepare, untimed): for deep_gemm 0.2.1.post1+muse print
  inspect.getsource or the docstring of bf16_gemm_nt and any DG_* env/config getters, then profile
  ONE deep_gemm.bf16_gemm_nt(x, W, out) call per (layer, m in {128, 256}) under torch.profiler
  (in-process, prepare only) and print the launched kernel names. A layer is DG-eligible only if
  the kernel name contains 'sm90_bf16_gemm' (or another deep_gemm kernel symbol) and not
  'nvjet'/'cublas'/'gemmSN'. STEP B, if step A finds cuBLAS: build DeepGEMM's sm90 BF16 GEMM
  directly from the image's header tree (deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh with
  the bundled CUTLASS 4.2.1 / cute) via torch.utils.cpp_extension.load_inline:
  TORCH_CUDA_ARCH_LIST=9.0a set around the build and restored, extra_cuda_cflags = ['-std=c++20'
  (replacing, never appending to, any -std=c++17), '-gencode=arch=compute_90a,code=sm_90a',
  '-lineinfo', '--expt-relaxed-constexpr', '-I<deep_gemm include>', '-I<cutlass include>'], and a
  `#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)` #error guard so a stray
  sm_90 pass fails the build cleanly. Instantiate with the shapes as compile-time constants
  (qkv_gate n=2304 k=8192; pooler n=4096 k=8192; unpooler n=8192 k=4096; m=128 and 256) and
  DeepGEMM's own heuristic block sizes (BLOCK_M 64/128, BLOCK_N chosen to fill 132 SMs, TMA
  multicast 2 on the larger dimension); build the weight TMA descriptors once in prepare; encode
  the x descriptor per call on the host from x.data_ptr() and the cfg shape (a pointer and shape,
  never values; graph capture records it), no allocation, no sync. Print regs/spills/smem
  (cudaFuncGetAttributes); reject any variant with spills. Self-check per (layer, m in {128, 130,
  200, 256}): rel L2 <= 3e-3 vs fp64, one-hot probe, finite, CUDA graph capture+replay; then the
  step-A profiler name assertion on the built kernel. Route in kernel_fn just before the cuBLAS
  fallback: layer DG-ok and 65 <= cfg m <= 256 and full n. o stays on cuBLAS (short K). Distinct
  from bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (our own swap-AB kernel): this tests
  DeepGEMM's shipped kernel; if both run, compare per row. Print '[dg] PATH CHOSEN <layer> m=<m>:
  deepgemm-api|deepgemm-header|master KERNEL=<name>'.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: incumbent nvjet splitK 12.6 + splitKreduce 3.1 (two nodes), row 17.0, floor 8.4; M2 row <= 15.5, M3 <= 13
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 19.5, row 19.7, floor 9.8; M2 <= 18.5
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: nvjet 24.5 at ~2.7 TB/s, row 25.0, floor 14.6; M2 <= 23
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: nvjet 30.7, row 31.2, floor 17.4; M2 <= 29
  - shapes: layer=unpooler load=verify_*
    direction: flat
    why: nvjet 20.3/24.3 already ~700 TFLOP/s at m=256; m=128 may move
  - shapes: layer=o *
    direction: flat
    why: not routed
  - shapes: layer=* load=decode_*
    direction: flat
    why: m<=32 untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256
review: proceed
status: running
review_at: 2026-10-25T042455Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T042455Z
base_sha: e5548a78a3d321973c377dc21431b085c324a412
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite
worktree: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite
mode: null
restack_of: null
stale_base: null
---

# bf16-verify-deepgemm-sm90-kernel-asserted-rewrite

## Change

Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT ever happened), yet every m>=128 profile row was nvjet: DeepGEMM's own sm90 BF16 kernel never ran. This stage makes it run and proves it. STEP A (import/prepare, untimed): for deep_gemm 0.2.1.post1+muse print inspect.getsource or the docstring of bf16_gemm_nt and any DG_* env/config getters, then profile ONE deep_gemm.bf16_gemm_nt(x, W, out) call per (layer, m in {128, 256}) under torch.profiler (in-process, prepare only) and print the launched kernel names. A layer is DG-eligible only if the kernel name contains 'sm90_bf16_gemm' (or another deep_gemm kernel symbol) and not 'nvjet'/'cublas'/'gemmSN'. STEP B, if step A finds cuBLAS: build DeepGEMM's sm90 BF16 GEMM directly from the image's header tree (deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh with the bundled CUTLASS 4.2.1 / cute) via torch.utils.cpp_extension.load_inline: TORCH_CUDA_ARCH_LIST=9.0a set around the build and restored, extra_cuda_cflags = ['-std=c++20' (replacing, never appending to, any -std=c++17), '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '--expt-relaxed-constexpr', '-I<deep_gemm include>', '-I<cutlass include>'], and a `#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)` #error guard so a stray sm_90 pass fails the build cleanly. Instantiate with the shapes as compile-time constants (qkv_gate n=2304 k=8192; pooler n=4096 k=8192; unpooler n=8192 k=4096; m=128 and 256) and DeepGEMM's own heuristic block sizes (BLOCK_M 64/128, BLOCK_N chosen to fill 132 SMs, TMA multicast 2 on the larger dimension); build the weight TMA descriptors once in prepare; encode the x descriptor per call on the host from x.data_ptr() and the cfg shape (a pointer and shape, never values; graph capture records it), no allocation, no sync. Print regs/spills/smem (cudaFuncGetAttributes); reject any variant with spills. Self-check per (layer, m in {128, 130, 200, 256}): rel L2 <= 3e-3 vs fp64, one-hot probe, finite, CUDA graph capture+replay; then the step-A profiler name assertion on the built kernel. Route in kernel_fn just before the cuBLAS fallback: layer DG-ok and 65 <= cfg m <= 256 and full n. o stays on cuBLAS (short K). Distinct from bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (our own swap-AB kernel): this tests DeepGEMM's shipped kernel; if both run, compare per row. Print '[dg] PATH CHOSEN <layer> m=<m>: deepgemm-api|deepgemm-header|master KERNEL=<name>'.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: incumbent nvjet splitK 12.6 + splitKreduce 3.1 (two nodes), row 17.0, floor 8.4; M2 row <= 15.5, M3 <= 13
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 19.5, row 19.7, floor 9.8; M2 <= 18.5
- `layer=pooler load=verify_16 m=128` faster: nvjet 24.5 at ~2.7 TB/s, row 25.0, floor 14.6; M2 <= 23
- `layer=pooler load=verify_32 m=256` faster: nvjet 30.7, row 31.2, floor 17.4; M2 <= 29
- `layer=unpooler load=verify_*` flat: nvjet 20.3/24.3 already ~700 TFLOP/s at m=256; m=128 may move
- `layer=o *` flat: not routed
- `layer=* load=decode_*` flat: m<=32 untouched
- `layer=router *` flat: untouched

## Risk

Milestones: M1 = a deep_gemm kernel name on >= 1 target row in the judge profile (else the run is master: read the [dg] lines and stop). M2 = qkv m=128 row <= 15.5 and pooler m=128 <= 23. Kill: M1 met and no target row beats its incumbent by > 3%. Header build may take 60-150 s; the sm_90 arch bug (LESSONS 2026-10-25) and the c++20 #error (bulk-tma build) are both pre-empted. Run as a check (a trap at self-check would poison the context).

## Milestones

Rewrite stage 1 of `cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042455Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as rewrite stage 1, with three fixes. (1) Replace the torch.profiler kernel-name probe with CUPTI-free evidence: source inspection, a bitwise-vs-cuBLAS output test in prepare, and a host-side route-taken flag printed from kernel_fn. (2) Make M a runtime value (or route only the exact m set that was built and verified) instead of compile-time m=128/256 with a 65..256 gate. (3) Treat STEP B (the header build) as the real deliverable, because the prior run's evidence points to the API being cuBLAS.

### Justification
A. The mechanism is plausible, and FINDINGS supports the targets. On master (b99b1dd) every m=128/256 row of qkv_gate, pooler and unpooler falls through to the cuBLAS fallback, at about 1.0x SGLang and 1.4-2.0x off the floor. qkv_gate m=128 is nvjet splitK 12.6 + splitKreduce 3.1 us (row 17.0, floor 8.4). qkv m=256 is 19.5 (floor 9.8). pooler is 24.5 at m=128 (~2.7 TB/s, floor 14.6) and 30.7 at m=256 (~560 TFLOP/s, floor 17.4). unpooler is 20.3/24.3. I confirmed that the master gates (_pooler_shape_ok, _qkv_route_ok, _unpool_shape_ok, _o_skinny_applies, both swap-AB gates) all stop at m<=32, so the new branch only ever displaces the _bf16_gemm_dispatch_impl fallback. The diagnosis of the prior run (2d522d47) is sound: 'PATH CHOSEN deepgemm' on all 16 pairs, 0.0 s first-call time, and an attribution table where every m=128/256 row is within noise of master (pooler 25.0->24.97, qkv 17.05->17.15). The 0.0 s with no JIT is the stronger clue. Together with nvjet tiles matching master exactly, it points to the '+muse' fork's bf16_gemm_nt wrapping cuBLASLt rather than the route being unreachable. I re-read that patch's _dg_route_ok and generate_input: x is a contiguous bf16 [m,k] with cfg m == x.shape[0], so the gate should have fired. STEP B is therefore the real experiment, and the expected directions are consistent with the floors. pooler (n=4096) and unpooler (n=8192) have enough tiles that a no-split-K persistent kernel can fill 132 SMs at BLOCK_M=64. WEAKEST LINK (two, equal): (a) qkv_gate m=128. DeepGEMM's dense sm90 BF16 GEMM has no split-K, so with one or two M-blocks n=2304 yields 36-72 output tiles at BLOCK_N=64/32, each walking K=8192. That is the same occupancy wall nvjet escaped with splitK, and B-multicast does not cut DRAM bytes with a single M-block. Treat its 'faster' as uncertain, and do not let a qkv loss count as a design kill if pooler/unpooler win. (b) STEP B rests on unverified facts. Nothing in the repo's logs shows that deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh exists in this fork, and the host-side heuristics, TMA-descriptor helpers and launch (cluster dims, dynamic smem attribute) live in DeepGEMM's csrc, not in the header tree. The builder must port them: cuTensorMapEncodeTiled via cudaGetDriverEntryPoint (no -lcuda assumption), cudaLaunchKernelEx with a cluster attribute when multicast=2, and cudaFuncSetAttribute for max dynamic smem. Print the template signature found before instantiating. If the header is missing, print '[dg] header absent' and keep master: the run is VOID, not a design result. Internal inconsistency to fix: the change instantiates 'shapes as compile-time constants (m=128 and 256)' but routes 65 <= m <= 256 and self-checks m=130/200, and the unpooler correctness test runs m=130. Either use SHAPE_M=0 (runtime M, which DeepGEMM templates support) with TMA OOB zero-fill and clipped TMA stores, or gate on the exact m set built and verified, as the prior implementation did. A compile-time M=128 kernel launched at m=130 is a correctness bug.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (b99b1dd, the new base): touches only m<=32 bf16 paths and the m<=32 router. On a cold-L2, graph-replayed, per-shape timing it shares no launch sequence or memory traffic with m=128/256 rows. Not a confounder. Its m=128/256 rows are master's nvjet, which is the incumbent this stage is judged against.
- bf16-m32-swapab-wgmma-triton / bf16-shape-gated-triton-winners-combo / router-* landed items: all gate at m<=32 or on the router. They are unaffected by a branch placed just before the cuBLAS fallback. The only shared resource is prepare-time budget: their Triton warm-ups plus a CUTLASS/cute c++20 header build must fit the 300 s prepare and the 10-minute run.
- sglang-gemm-dispatch-parity: makes master's m>=65 incumbent byte-for-byte SGLang's _bf16_gemm_dispatch_impl. Each target row's paired ratio vs SGLang is therefore a direct DeepGEMM-vs-nvjet measurement. This helps attribution; it does not confound it.
- Not landed but overlapping: bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (being built) routes the same layers at the same m=128/256 rows before the same fallback. Separate runs on separate branches from b99b1dd do not confound each other. Stacking them without disjoint per-(layer, m) gate tables would.

### Interference and A/B needs
No companion. This is a rewrite stage, and every target row is self-attributing once three things hold. (1) The judge profile's top_kernels for that row names a DeepGEMM symbol (sm90_bf16_gemm*) or the extension's kernel, not nvjet. This is M1 and is read at harvest from the judge's own profile, which needs no in-submission CUPTI. (2) kernel_fn prints '[dg] ROUTE TAKEN <layer> m=<m> path=<api|header>' the first time the branch fires per (layer, m). This is a host-side Python flag, which runs during the judge's graph capture, so it proves the timed graph contains our launch. It closes the 'never reached kernel_fn' hypothesis that the prior run could not rule out. (3) The row's paired ratio vs SGLang is compared with master's row from the base result (1e90292c / 60b20fd0), allowing for the stated +-1.5% drift. Profiler probe, per the earlier audit: drop torch.profiler from prepare. It depends on CUPTI, which can fail silently and would then mark a good route ineligible. It can also conflict with the judge's own profiling session and leave the M1 profile empty. And it reads like timing instrumentation to the submission checker. Replace step A with CUPTI-free evidence: print inspect.getsource or the docstring of bf16_gemm_nt and grep it for cublas/cublasLt/nvjet. Run one call per (layer, m in {128, 256}) and compare torch.equal(deep_gemm_out, _bf16_gemm_dispatch_impl(x, W, None)) on synthetic x. Bit-identical output on every pair, or a cuBLAS mention in the source, marks the API as cuBLAS and sends the run to STEP B. The API route is used only if the source shows no cuBLAS AND the outputs differ bitwise AND rel-L2 passes. For the header build there is nothing to assert: the launched function is ours by construction, so cudaFuncGetAttributes on that function pointer (regs, spills, smem) plus the route-taken flag is the proof. Attribution against bf16-verify-swapab-wgmma-cluster-dsmem-rewrite, which shares the qkv/pooler/unpooler m=128/256 rows. (a) Build both from master b99b1dd, never one on the other's branch. (b) Use distinct log tags ('[dg]' vs that run's own) and distinct kernel symbols, so each row's top_kernels names its owner. (c) Compare the two per row through each one's paired ratio vs SGLang in its own job, never raw us across jobs, because cross-job drift is about 1.5% and a raw-us difference under ~3% is not a ranking. (d) Whichever wins a row is restacked later through a separate restack hypothesis with one explicit per-(layer, m) owner table, so neither result is ever read off a combined run.

### Open questions
- Does this fork (deep_gemm 0.2.1.post1+muse.cbe1727) ship impls/sm90_bf16_gemm.cuh at all, and with which template signature? Print the directory listing of deep_gemm/include/deep_gemm/impls and the template header line before building. If it is absent, the stage is VOID.
- Which BLOCK_M/BLOCK_N/stages/multicast does the ported heuristic pick per (layer, m)? Print them with the regs/smem line, so a qkv_gate m=128 loss can be traced to the tile count (36-72 tiles for 132 SMs) rather than to the kernel.
- Build time: one TU with all instantiations (3 layers x the chosen tile configs, runtime M), with a prepare-elapsed guard that skips the build if more than ~150 s have elapsed. A cute c++20 TU with 6+ instantiations can exceed the 60-150 s estimate.
- If STEP A shows the API does launch a genuine DeepGEMM kernel (source has no cuBLAS, outputs not bit-identical) yet the judge profile again shows nvjet, the fault is in the route, not the library. The ROUTE TAKEN flag discriminates that case.


## Implementation notes

Built blind (no GPU/torch here); only the Python parse/codegen/config helpers were exercised against a mock header. All edits are in the worktree `muse_bf16_proj/submission.py` (+739 lines, nothing removed): new imports (hashlib, os, re, time), a `[dg]` block before `prepare()`, one call at the end of `prepare()`, and one branch in `kernel_fn` just before the cuBLAS fallback.

Where the adversarial review overrides the spec (review wins):
- NO torch.profiler anywhere. STEP A is `_dg2_step_a`: prints `inspect.getsource` (or docstring, module, version, getters, DG_* env) of `deep_gemm.bf16_gemm_nt`, greps it for cublas/cublaslt/nvjet/gemmSN, and compares `torch.equal(bf16_gemm_nt(x,W,out), _bf16_gemm_dispatch_impl(x,W,None))` for every layer at m=128 and 256. The API route (`deepgemm-api`) is used only if no cuBLAS word AND no pair is bit-identical; otherwise STEP B.
- STEP B is the deliverable (`_dg2_build`): locates `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` (or an `*sm90*bf16*gemm*.cuh` sibling), prints the impls listing and the parsed template and kernel parameter lists. If the header is absent it prints `[dg] header absent ... VOID RUN` and master runs (this run then measures master). The kernel's template list and argument list are parsed from the header and filled by name; any name the port does not know prints `signature mismatch ... VOID RUN` instead of guessing.
- Build: `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored in `finally`; nvcc flags `-std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr --expt-extended-lambda -lineinfo -gencode=arch=compute_90a,code=sm_90a` plus the `-U__CUDA_NO_*` undefs, `-std=c++20` also in extra_cflags, `-I<deep_gemm include>` (CUTLASS 4.2.1/cute ship there); torch.utils.cpp_extension is never patched. A `#error` guard rejects any non-sm_90a device pass. Flags and arch are printed. Build is skipped if prepare() is already past 110 s.
- M bug fixed both ways: SHAPE_M = 0 (runtime M, TMA OOB zero-fill on A, clipped TMA stores on D) AND routing only the exact (layer, m) pairs built and verified, m in {128, 130, 200, 256} (the m=130 unpooler test included). m=512/1000 and everything else stay on master. Two instantiations per layer: group {128} and group {130,200,256} (no B-multicast, since m % 128 != 0 for 130/200); 6 in one TU.
- Heuristic port (`_dg2_pick_cfg`): BLOCK_K 64; BLOCK_M 64/128; BLOCK_N a divisor of n in 32..256 step 16; multicast 1 or 2 on A (even N-blocks) or on B (even M-blocks for every m in the group); stages from the 232448 B cap; chosen by a bytes/flops per-wave cost model. Each variant's choice and modeled time is printed.
- Launcher is ours: `cuTensorMapEncodeTiled` via `cudaGetDriverEntryPoint` (no -lcuda), weight (B) descriptors encoded once in `dg2_set_weight`, x and out descriptors encoded per call from `data_ptr()` and shape only, `cudaLaunchKernelEx` with a cluster attribute, `cudaFuncSetAttribute` for max dynamic smem, `cudaFuncGetAttributes` printed (regs, local bytes, static smem, max dyn smem, SASS arch, ptx). A variant is rejected if SASS arch != 90 or local bytes > 512 (not 0: printf frames can show as local bytes; the spill number is printed regardless).
- Self-check per (layer, m): flushed print before and after each launch with `torch.cuda.synchronize()`, finite, rel L2 <= 3e-3 vs fp64, one-hot probe <= 1e-4, CUDA graph capture + replay, and a printed `bitwise_equal_to_cuBLAS` (a header kernel should print False). Any failure disables that pair only.
- Proof lines: `[dg] PATH CHOSEN <layer> m=<m>: deepgemm-api|deepgemm-header|master KERNEL=<name>` for all 12 pairs, and `[dg] ROUTE TAKEN <layer> m=<m> path=<api|header>` printed once per pair from kernel_fn (host-side set, shape-only; fires at graph capture). Tag `[dg]` and kernel symbol `sm90_bf16_gemm_impl` are distinct from the swap-AB cluster rewrite.

Deviations / risks:
- "Bound every mbarrier wait" cannot be met: the waits live in DeepGEMM's shipped header, which is not edited. Mitigation: arch guard, signature-name gate, local-bytes/arch rejection, eager per-launch prints, and this must run as `okc check`. A genuine hang would reach the 10-minute limit.
- BLOCK_N=48 (qkv_gate, swizzle-D 32 B) is a legal-looking but untested DeepGEMM tile; if its static_asserts reject it the whole TU fails to build and the run is VOID (the build log tail is printed). Next stage: restrict BLOCK_N to multiples of 32/64.
- kNumLastStages = (K/64) % stages and `deep_gemm::EpilogueIdentity` / `GemmType::Normal` are from memory; `_dg2_build` greps the tree for the real epilogue and enum names and prints them.
- Compile time of 6 cute instantiations may exceed 150 s; prepare's 270 s check cutoff leaves pairs on master if reached.
- qkv_gate m=128 has no split-K (96 tiles of 64x48), the review's weakest link; do not read a qkv loss as a design kill if pooler/unpooler win.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only shape gating, a host-side print-once set, `torch.empty` and a launcher call (descriptor encode + cudaLaunchKernelEx, no sync); the graph capture + replay is self-tested per pair in prepare().
- [x] Nothing derived from `data` is cached: `dg2_seen` holds (layer, cfg m) keys only; descriptors for x/out are re-encoded each call.
- [x] Weights: unchanged bf16, read in place; the B descriptor points at the prepared weight, no repack or precision change.
- [x] Activations BF16 where SGLang keeps BF16 (bf16 in, fp32 accumulate in the wgmma mainloop, bf16 out).
- [x] Attention: n/a (dense projections).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (load_inline compiles in-process via the torch extension API; no subprocess/os.system in the file; no profiler).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Branches in kernel_fn are on cfg and shapes only, never on input values; `o`, router, m<=32 and m outside the verified set are untouched.

okc_lint: `OK with warnings` (6 host-sync warnings, all `torch.cuda.synchronize()` in prepare()-only helpers, flagged by the linter as outside kernel_fn's call graph). py_compile clean, __pycache__ removed.
