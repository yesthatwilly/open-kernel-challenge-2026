---
problem: muse_bf16_proj
slug: bf16-cublaslt-backend-toggle
created: 2026-10-23T225303Z
base: master
area: bf16 GEMM dispatch
change: >
  A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only,
  call torch.backends.cuda.preferred_blas_library('cublaslt') immediately before the
  _bf16_gemm_dispatch_impl(x, W[:n], None) call and restore
  torch.backends.cuda.preferred_blas_library('cublas') immediately after it (both are host-side
  settings, so inside the captured graph only the kernel chosen under the Lt backend is recorded;
  the restore guarantees SGLang's own measurement in the same process is taken with the default
  backend). ATen's torch.mm for bf16 goes through cublasGemmEx under the default backend and
  through cublasLtMatmul with PyTorch's own heuristic query and a 32 MB Lt workspace under
  'cublaslt'; the two can select different nvjet tiles and splitK factors for the same shape.
  Router path and prepare() unchanged. Kernels possibly moved: every nvjet_sm90_tst_* +
  splitKreduce pair on the 20 bf16 shapes; the run's profile shows which shapes changed kernel
  name at all.
expected:
  - shapes: "*layer=pooler load=decode_*"
    direction: faster
    why: the Lt heuristic with a larger workspace may pick a different splitK factor for the 4096x8192 skinny GEMM; the shapes at 1.5x floor have the most to gain
  - shapes: "*layer=unpooler load=decode_16 *"
    direction: faster
    why: single non-splitK nvjet today; Lt may choose a splitK variant
  - shapes: "*layer=router*"
    direction: flat
    why: router path untouched; control
  - shapes: "*layer=o*"
    direction: flat
    why: likely the same kernel either way; a second control
risk: low
depends_on: sglang-gemm-dispatch-parity
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T071318Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T071319Z
base_sha: 036af8be03a756684fc9d199dd2f6b1f05eab3a4
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle
worktree: .worktrees/muse_bf16_proj/bf16-cublaslt-backend-toggle
mode: null
---

# bf16-cublaslt-backend-toggle

## Change

A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') immediately before the _bf16_gemm_dispatch_impl(x, W[:n], None) call and restore torch.backends.cuda.preferred_blas_library('cublas') immediately after it (both are host-side settings, so inside the captured graph only the kernel chosen under the Lt backend is recorded; the restore guarantees SGLang's own measurement in the same process is taken with the default backend). ATen's torch.mm for bf16 goes through cublasGemmEx under the default backend and through cublasLtMatmul with PyTorch's own heuristic query and a 32 MB Lt workspace under 'cublaslt'; the two can select different nvjet tiles and splitK factors for the same shape. Router path and prepare() unchanged. Kernels possibly moved: every nvjet_sm90_tst_* + splitKreduce pair on the 20 bf16 shapes; the run's profile shows which shapes changed kernel name at all.

## Expected

- `*layer=pooler load=decode_*` faster: the Lt heuristic with a larger workspace may pick a different splitK factor for the 4096x8192 skinny GEMM; the shapes at 1.5x floor have the most to gain
- `*layer=unpooler load=decode_16 *` faster: single non-splitK nvjet today; Lt may choose a splitK variant
- `*layer=router*` flat: router path untouched; control
- `*layer=o*` flat: likely the same kernel either way; a second control

## Risk

Honest expectation: most shapes flat, because cublasGemmEx already consults the Lt heuristics internally on Hopper; this run exists to settle whether any PyTorch-level BLAS dispatch lever is left before all remaining bf16 effort goes into custom kernels. Reading: compare kernel names per shape against the parity profile; any shape whose nvjet name did not change and whose time moved is noise. Rule edges: a documented torch API call, not a monkeypatch of a loaded module and not harness state; the restore after each call keeps SGLang's baseline measurement unaffected (if the harness measured SGLang under our setting the paired ratios would be confounded, which is why the toggle is scoped to the single call). If the image's torch lacks preferred_blas_library or ignores it for bf16 mm, the run is a no-op at parity, costing one check. No correctness exposure: same cuBLAS math.

## Adversarial review

Reviewed 2026-10-24T071318Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a cheap, rule-safe one-check probe with a low prior. On Hopper, cublasGemmEx already runs through the cublasLt heuristic: the profile shows cublasLt::splitKreduce_kernel under the default backend. So most shapes should pick the same nvjet kernel. Proceed with no companions, but save and restore the previous backend in try/finally instead of hard-coding 'cublas', and read the result per shape by kernel name, not by the headline speedup.

### Justification
Plausible but weak. Every remaining bf16 shape is a cuBLAS nvjet kernel (plus splitKreduce on the skinny shapes), and all of them sit at parity with SGLang (FINDINGS: qkv_gate 13.8-19.5 us vs SoL 7.9-9.8, pooler 21.2-30.5 vs 14.0-17.4, unpooler decode_16/32 18.4/18.7 vs 14.1, o 6.3-9.5 vs 3.5-4.5). That leaves 1.3-2x of headroom, so the room is there if a different kernel exists. Weakest link: the mechanism. The cublasLt::splitKreduce_kernel in today's profile shows that cublasGemmEx on sm90 is already a front end to the same Lt heuristic. The two backends can only choose differently if the heuristic inputs differ: (a) workspace size, or (b) descriptor and alignment attributes, or the requested algo count, in ATen's bgemm_internal_cublaslt. On recent PyTorch, the Lt workspace defaults to the cuBLAS workspace size (32 MiB on Hopper), which would remove (a). The hypothesis's claim of 'a 32 MB Lt workspace' versus a smaller default is version-dependent and unverified for the judge image. Note also that the CUBLASLT_WORKSPACE_SIZE env var cannot be used, because it is harness or process state. A second weak link: whether this torch build sends bf16 at::mm through Lt at all when preferred_blas_library is 'cublaslt'. Older CUDA builds honoured the setting only for some dtypes or for addmm with bias. If they ignore it, the run is a parity no-op, as the hypothesis admits. The expected entries are internally consistent with TASK.md: pooler decode_* is the largest absolute gap (about 7 us over floor at weight 62 in four loads), router flat is a true control (that path does not move), and o flat is a reasonable second control. The list omits qkv_gate (all six shapes) and unpooler decode_1/verify_1, which the change also touches and which are just as likely to move. The harvest should treat every bf16 shape as a candidate and not score those as unexpected. Magnitude: a 2 us pooler gain is about 3.5% of the decode_1 load (about 55 us), so a single-layer win could show at the aggregate level only if it hits most loads. Per-shape attribution by kernel name is the real readout.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: this is the base the toggle wraps (it supplies the _bf16_gemm_dispatch_impl call). It is not a confounder: reverting it would replace the dispatch with x @ W.T, which reaches the same at::mm and would be an equivalent target. Nothing to disambiguate.
- router-triton-hilo-fused-control-splitk-all-m: it touches only the router branch, and the toggle is scoped to the non-router branch, so the kernels do not interact. One indirect coupling: the router no longer uses cuBLAS on master, so even a buggy, leaked global setting would not move router shapes. A non-flat router row therefore indicates run noise (useful as a noise gauge), not an interaction. Its result also showed qkv_gate decode_1 'lost' with an unchanged kernel, which is the per-shape noise level to expect here.
- router-hilo-at-m8: superseded on master by the Triton router (no torch.mm router path remains at any m), so it is not a confounder.

### Interference and A/B needs
No companion is warranted. ab-revert: no landed hypothesis touches the bf16 cuBLAS path except the parity dispatch, and reverting that reaches the same ATen mm, so the run would measure nothing new. ab-tune: the only landed tunables are router Triton configs, which do not reach these shapes. control: the change has one sub-change, the backend toggle around one call. Applying it per layer is not needed to disambiguate, because the profile already attributes per shape. A shape whose nvjet or splitKreduce kernel names changed and whose time moved is attributable to the toggle. A shape whose kernel names are unchanged is noise, whichever way it moved. A run where no kernel name changed anywhere is a clean negative: the PyTorch BLAS-backend lever is dead on this image, and later bf16 work goes to custom kernels. One caveat on readability: an unchanged nvjet name does not prove an identical launch, because the name encodes tile, stages and splitK presence but not the splitK factor or the reduce shape. A sub-CI move with an identical name pair should still be read as noise. If a gain appears on some layers and a loss on others, the follow-up is a stacked hypothesis that applies the toggle per layer only where it won. That is a new hypothesis, not a companion of this run.

### Open questions
- Does the judge image's torch send bf16 at::mm (via F.linear with no bias) through bgemm_internal_cublaslt when preferred_blas_library is 'cublaslt'? A one-line print in prepare() of torch.__version__ and torch.backends.cuda.preferred_blas_library() before and after the set would confirm that the API exists and what the default backend is, and the profile's kernel names would confirm the routing.
- What are the cuBLAS and Lt workspace sizes in that image? If both are 32 MiB, the heuristic inputs are nearly identical and the expected outcome is all-flat.
- If a shape does switch kernels, check that SGLang's paired measurement in the same run still shows its old kernel name. That proves the restore worked and the baseline was not perturbed.

