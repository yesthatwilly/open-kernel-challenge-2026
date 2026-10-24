---
problem: muse_bf16_proj
slug: pooler-splitk2-tma-weight-stream
created: 2026-10-24T092648Z
base: master
area: pooler GEMM
change: >
  From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32:
  _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2, ticket-semaphore reduce;
  harvested neutral at 20.2-20.7 us per pooler row, ~3.3 TB/s, the same rate as every
  plain-tl.load Triton weight stream so far), change only how the weight tile reaches shared
  memory: load it with a Hopper TMA tensor descriptor instead of per-thread tl.load (cp.async).
  Build the descriptor once in prepare() from the pooler weight alone (shape [4096, 8192], strides
  [8192, 1], block [BLOCK_N=64, BLOCK_K=256], bf16): with Triton >= 3.4 use
  triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [64, 256]) passed as a kernel
  argument and w = w_desc.load([n0, k]) in the K loop; on older Triton use
  triton.tools.experimental_descriptor.create_2d_tma_descriptor and
  tl._experimental_descriptor_load(desc, [n0, k], [64, 256], tl.bfloat16).
  print(triton.__version__) and the chosen API in prepare(). The tile arrives [BLOCK_N, BLOCK_K]
  (K-contiguous, as stored); feed acc = tl.dot(x_tile, tl.trans(w), acc) (the transpose is a
  smem-descriptor/ldmatrix.trans view, not a copy). BLOCK_K 256 (16 K-steps per 4096 K-slice) and
  num_stages 4: 32 KB of weight per stage, 128 KB of weight in flight per CTA, moved by one
  elected thread per stage on an mbarrier instead of ~64 cp.async.16 per thread, which leaves the
  warps free to issue the next stage and the MMA. x stays a plain masked tl.load ([BLOCK_M, 256],
  L2-resident). Smem 4 x (32 KB + BLOCK_M x 512 B) = 160 KB at BLOCK_M=16, 192 KB at BLOCK_M=32
  (drop to num_stages 3 for the BLOCK_M=32 specialization only if it does not fit). Everything
  else stays as the parent: routing (pooler m <= 32 only), grid, SPLIT_K, the fp32 workspace,
  barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, warm-up and counter
  re-zero in prepare(). No autotune; the descriptor depends only on the weight.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: "21.2 us on master (parent 20.2) vs floor 14.0: the largest per-shape gap on the four decode-type loads (pooler is ~38% of each). Every Triton stream so far tops out at 3.3-3.7 TB/s with cp.async; TMA bulk loads with 128 KB in flight per CTA should reach ~4.0-4.2 TB/s. Target <= 17.5 us kernel time"
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: same stream at m=8 (parent 19.9 us); target <= 17.5 us
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: same stream at m=16 (parent 20.7 us); target <= 18 us
  - shapes: layer=pooler load=decode_32 m=32
    direction: flat
    why: parent 22.2 us; the m=32 penalty is in the x path and smem, not the weight stream, so TMA may only recover part of it
  - shapes: layer=pooler load=verify_16 m=128
    direction: flat
    why: "control: cuBLAS"
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
risk: high
depends_on: pooler-triton-splitk2-blockn64-semaphore
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T104016Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T104051Z
base_sha: cee590d58c6c34ded84fdb734c38ca129e327fee
base_blob: 66ba1a39e2aaa5cedb2ce3ddfb809f67f86b37f9
branch: hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream
worktree: .worktrees/muse_bf16_proj/pooler-splitk2-tma-weight-stream
mode: check
implemented_at: 2026-10-24T104412Z
impl_commit: 954d39338b5e6e0be9df9f550b6ba7ff10a7a6e7
submission_sha256: 253d30c05c8eec77961865b4ee1be97c9705eac5c91fbb94e6b152a414c396b1
norm_sha256: 1e415f07907b4d6c9b704d4cf61f58bd7d8e8528e2ede19f65fd7900bf5878bd
submission_blob: c7d9d9d449e9412239f90ab6648b4d6f6bd658e7
lint_warnings: 0
attempt_file: 2026-10-24T104413Z-check-pooler-splitk2-tma-weight-stream.md
prepared_at: 2026-10-24T104413Z
---

# pooler-splitk2-tma-weight-stream

## Change

From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32: _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2, ticket-semaphore reduce; harvested neutral at 20.2-20.7 us per pooler row, ~3.3 TB/s, the same rate as every plain-tl.load Triton weight stream so far), change only how the weight tile reaches shared memory: load it with a Hopper TMA tensor descriptor instead of per-thread tl.load (cp.async). Build the descriptor once in prepare() from the pooler weight alone (shape [4096, 8192], strides [8192, 1], block [BLOCK_N=64, BLOCK_K=256], bf16): with Triton >= 3.4 use triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [64, 256]) passed as a kernel argument and w = w_desc.load([n0, k]) in the K loop; on older Triton use triton.tools.experimental_descriptor.create_2d_tma_descriptor and tl._experimental_descriptor_load(desc, [n0, k], [64, 256], tl.bfloat16). print(triton.__version__) and the chosen API in prepare(). The tile arrives [BLOCK_N, BLOCK_K] (K-contiguous, as stored); feed acc = tl.dot(x_tile, tl.trans(w), acc) (the transpose is a smem-descriptor/ldmatrix.trans view, not a copy). BLOCK_K 256 (16 K-steps per 4096 K-slice) and num_stages 4: 32 KB of weight per stage, 128 KB of weight in flight per CTA, moved by one elected thread per stage on an mbarrier instead of ~64 cp.async.16 per thread, which leaves the warps free to issue the next stage and the MMA. x stays a plain masked tl.load ([BLOCK_M, 256], L2-resident). Smem 4 x (32 KB + BLOCK_M x 512 B) = 160 KB at BLOCK_M=16, 192 KB at BLOCK_M=32 (drop to num_stages 3 for the BLOCK_M=32 specialization only if it does not fit). Everything else stays as the parent: routing (pooler m <= 32 only), grid, SPLIT_K, the fp32 workspace, barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, warm-up and counter re-zero in prepare(). No autotune; the descriptor depends only on the weight.

## Expected

- `layer=pooler load=decode_1 m=1` faster: 21.2 us on master (parent 20.2) vs floor 14.0: the largest per-shape gap on the four decode-type loads (pooler is ~38% of each). Every Triton stream so far tops out at 3.3-3.7 TB/s with cp.async; TMA bulk loads with 128 KB in flight per CTA should reach ~4.0-4.2 TB/s. Target <= 17.5 us kernel time
- `layer=pooler load=verify_1 m=8` faster: same stream at m=8 (parent 19.9 us); target <= 17.5 us
- `layer=pooler load=decode_16 m=16` faster: same stream at m=16 (parent 20.7 us); target <= 18 us
- `layer=pooler load=decode_32 m=32` flat: parent 22.2 us; the m=32 penalty is in the x path and smem, not the weight stream, so TMA may only recover part of it
- `layer=pooler load=verify_16 m=128` flat: control: cuBLAS
- `layer=qkv_gate *` flat: untouched

## Risk

High risk: the judge image Triton version is not recorded anywhere in the repo. Both API branches are named; if neither exists the implementer stops and records it (no run). It changes two things relative to the parent (TMA, and BLOCK_K 128 -> 256, which TMA needs for a 32 KB stage); the held pooler-triton-splitk2-blockn64-semaphore-control-blockk256 is the BLOCK_K-only companion if the reviewer wants the split. Device-side TMA with mbarriers is graph-capturable; the host descriptor is built once from the weight pointer (not input-derived). Readout: the kernel us per pooler row against the parent 20.2/19.9/20.7/22.2 us; a stream above 4 TB/s would carry over to qkv_gate and unpooler at m <= 32.

## Adversarial review

Reviewed 2026-10-24T104016Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** TMA for the pooler weight stream is a reasonable next lever and breaks no rule. But the change also moves BLOCK_K from 128 to 256, and that alone has a measured precedent (the unpooler program at 18.0 us, 3.73 TB/s) close to the hypothesis's own 17.5 us target. So one run cannot credit any gain to TMA. Run it alongside the cp.async BLOCK_K=256 control, which already exists, held, as pooler-triton-splitk2-blockn64-semaphore-control-blockk256.

### Justification
Mechanism. At m<=32 the pooler is a pure 67.1 MB weight stream. The parent's _bf16_splitk_kernel took 19.5/20.0/19.9/22.7 us of kernel time at m=1/8/16/32. That is 3.44/3.36/3.37/2.96 TB/s against a 14.0-14.1 us floor (4.8 TB/s). Shape times were 20.2/19.9/20.7/22.2 us. That leaves 5-6 us of headroom on four rows, each about 31-38% of its load at equal weight 62, so a real gain would show in the geomean (about +2-3% if 2 us per row). On H200, TMA plus mbarrier bulk copies do cut issue pressure compared with 4 warps issuing 64 cp.async.16 per thread per 32 KB stage. 128 KB in flight per CTA times 128 CTAs is about 16 MB in flight, far above the roughly 5 MB that Little's law needs, so the target can be reached in principle. Shared memory is 160/192 KB, which fits within 227 KB. The [64,256] box is within TMA's 256-element per-dimension limit. Directions are consistent with TASK.md: m=1/8/16 faster, m=128/256 and other layers flat as in-run controls. decode_32 marked flat is defensible. BLOCK_K=256 at BLOCK_M=32 on the unpooler ran 19.8 us of kernel time against the parent's 22.7, so decode_32 may even come out faster, and a TMA effect there is equally unattributable. WEAKEST LINK: the premise that 'every plain-tl.load Triton stream tops out at 3.3 TB/s' is contradicted by LESSONS and by unpooler-full-triton-skinny-m32. That kernel has the same per-CTA geometry this change produces (BLOCK_N=64, BLOCK_K=256, 4 stages, 128 KB in flight, 64 rows x 4096 K) and reached 18.0 us at m=16 (3.73 TB/s) with cp.async. The parent's 3.4 TB/s is explained by its BLOCK_K=128 (64 KB in flight), which the parent's own reviewer flagged. So most of the predicted 20.2 -> 17.5 us is already expected from the bundled BLOCK_K change, and TMA's own increment is the remaining ~0.5 us, below the ~1% Triton reproducibility only marginally. A second weak point is that TMA itself is no guarantee of more than 4 TB/s. cuBLAS's nvjet already uses TMA and moves this same weight at 3.67 TB/s (18.3 us at m=1).

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: different layer and kernel, no shared workspace or counters, cold L2 per replay. It cannot move pooler rows. Router rows are only a reproducibility check.
- sglang-gemm-dispatch-parity: defines the cuBLAS path that pooler m>=128 and all other layers keep. Those rows are flat controls, not an interaction.
- router-hilo-at-m8: router-only, now subsumed by the Triton router, with no shared path. Not a confounder.
- Not landed but structural: the base is the neutral parent branch (pooler-triton-splitk2-blockn64-semaphore, 1.166x), not master (1.143x). The headline delta against BEST therefore includes the parent's own roughly +2% pooler effect. Read the gain from the per-row kernel us against the parent's 19.5/20.0/19.9/22.7 us, not from the headline.

### Interference and A/B needs
No landed hypothesis touches the pooler path, its traffic or its numerics, so ab-revert and ab-tune are not warranted. The ambiguity comes from the hypothesis bundling two sub-changes: BLOCK_K 128->256 (doubling bytes in flight) and cp.async->TMA. A gain to about 18 us is exactly what the BLOCK_K change alone predicts from the unpooler precedent, so it cannot be credited to TMA. A flat or lost result is also ambiguous: it could mean TMA does not help, or that TMA's mma.sync operand path costs more than the extra stage bytes buy. Either way the follow-up decision (port TMA to qkv_gate and unpooler, or just use BLOCK_K=256 everywhere) depends on the split. One control resolves both cases: the parent's branch with BLOCK_K=256 and plain tl.load. That code already exists as pooler-triton-splitk2-blockn64-semaphore-control-blockk256. Its hold reason was that a control isolating a sub-change of a neutral parent cannot change a decision, and that no longer applies, because it is now the cp.async arm of this A/B. Re-queue that file rather than minting a duplicate (identical code is never re-run). The slug below is only nominal. One companion is enough. A TMA run at BLOCK_K=128 would be redundant once both 256 runs exist.

### Open questions
- Which Triton version does the judge image run? If neither API exists, the hypothesis records that and makes no run. The control is still worth running on its own, because it answers the BLOCK_K question.
- Box inner extent: 256 bf16 is 512 B, beyond the 128 B swizzle span. Newer Triton splits such a box into several TMA messages per stage. The old experimental create_2d_tma_descriptor path may reject it or fall back to a non-swizzled layout, which causes bank-conflicted ldmatrix. If so, use a [64,64] box with four loads per K-step, or record the fallback.
- At BLOCK_M=16 with 4 warps, Triton lowers tl.dot to mma.sync (MMAv2), not wgmma. Confirm that tl.trans(w) on the TMA NVMMA-shared tile becomes an ldmatrix(.trans) view and not a convert_layout through registers. Print n_regs/n_spills and grep 'cp.async.bulk.tensor' and 'ldmatrix' in kernel.asm['ptx'] in prepare(), since this node cannot compile.
- Does Triton's pipeliner keep 4 TMA stages in flight while x is still a masked cp.async load in the same loop? Mixed async groups sometimes collapse the pipeline depth. The PTX mbarrier count, printed in prepare(), answers this.
- nvjet already uses TMA and reaches only 3.67 TB/s on this weight. If the TMA arm lands at 18 us rather than 17.5, the per-SM or HBM efficiency ceiling, not the copy mechanism, may be the limit for one 128-CTA wave plus the reduce tail.
- churnsubs: the proposed cpasync-blockk256 companion duplicates the held pooler-triton-splitk2-blockn64-semaphore-control-blockk256, which was unheld instead of minting a new file


## Implementation notes

Worktree file: `.worktrees/muse_bf16_proj/pooler-splitk2-tma-weight-stream/muse_bf16_proj/submission.py` (base cee590d, the parent pooler-triton-splitk2-blockn64-semaphore). Diff: 1 file, +155/-8. Pooler path only; router, cuBLAS dispatch and routing (pooler m <= 32) untouched.

- New `_bf16_splitk_tma_kernel`: a copy of `_bf16_splitk_kernel` in which only the weight load changed. `w_desc` replaces `w_ptr`/`stride_wn`; per K-step `w_tile = w_desc.load([n0, k])` (constexpr `NEW_DESC_API`) or `tl._experimental_descriptor_load(w_desc, [n0, k], [64, 256], tl.bfloat16)`; `acc = tl.dot(x_tile, tl.trans(w_tile), acc)`. x is still a masked `tl.load` of [BLOCK_M, 256]. Workspace store, debug_barrier, acq_rel gpu ticket, fixed-order last-CTA reduce, bf16 store and atomic_xchg reset are byte-for-byte the parent's. Grid (64, 2), SPLIT_K 2, BLOCK_N 64, num_warps 4, BLOCK_K 256 (`POOLER_TMA_BLOCK_K`), num_stages 4.
- API selection by feature at import: `triton.tools.tensor_descriptor.TensorDescriptor` if importable (the >= 3.4 API, `from_tensor(W, [64, 256])`), else `triton.tools.experimental_descriptor.create_2d_tma_descriptor(W.data_ptr(), 4096, 8192, 64, 256, 2)`. The descriptor is built once in prepare() from the pooler weight alone and kept in `state["pooler_w_desc"]`. prepare() prints `triton.__version__` and the chosen API.
- Deviation 1 (fallback): if neither API imports, `POOLER_TMA_API = None` and the parent's cp.async kernel (BLOCK_K 128) runs unchanged, so the run does not crash. The md says "no run" in that case, but this node has no Triton and the judge's Triton version is recorded nowhere in the repo (grep of attempts/hypotheses/logs found nothing), so it cannot be known before a run. SGLang on H200 needs torch >= 2.5 (Triton >= 3.1), where at least the experimental API exists, so the fallback should never trigger. If the prepare() log says `none (cp.async fallback)`, the run is the parent and must be read as "no TMA API", not as a TMA result.
- Deviation 2 (stages at BLOCK_M=32, per the md's "only if it does not fit"): prepare() compiles BLOCK_M=16 and 32 first; if the BLOCK_M=32 launch raises `triton.runtime.errors.OutOfResources` at 4 stages, it retries at 3 and records that in `state["pooler_stages"]` (a constant from compile resources, not input-derived). The m=32 log line says which.
- Diagnostics (from the review's open questions): prepare() prints per BLOCK_M `n_regs`, `n_spills` and PTX counts of `cp.async.bulk.tensor`, `ldmatrix`, `.trans`, `mbarrier.`, non-bulk `cp.async`, `wgmma.mma_async` and `mma.sync` from the launch handle's `asm['ptx']`. It is wrapped in try/except so it can never fail prepare().
- `_pooler_forward` now returns `(out, handle)`; kernel_fn takes `[0]`. The warm-up over m = 1..32 and the counter re-zero are as in the parent.
- Box [64, 256]: inner extent 512 B, more than the 128 B swizzle span. Triton's host fill for the experimental path clamps the box to 128 B and the codegen issues several copies. The new API's lowering splits it too. Not verifiable offline. The PTX `cp.async.bulk.tensor` count shows how many copies were issued per stage.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/pooler-splitk2-tma-weight-stream diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn. Prints happen only in prepare(), and they print strings and ints, never tensors. The output `torch.empty` is the parent's per-call allocation. The branches are on shapes (`_pooler_shape_ok`, `_pooler_block_m`) and on prepare()-fixed state (descriptor present, stages), never on input values. The TMA loads are device-side and the descriptor is a fixed host param.
- [x] Nothing derived from `data` is cached across calls: the descriptor is built from the pooler weight only. `pooler_stages` comes from compile resources. The workspace is rewritten every call, and the counters self-reset as in the parent.
- [x] Weights: the pooler weight is unchanged bf16. The descriptor is a view of it, not a copy. The router hi+lo split is unchanged.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16, the accumulation is fp32 and the output is bf16, as in the parent.
- [x] n/a: attention. This problem has no attention, KV cache or queries.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new imports are triton.tools.* only.
- [x] All work happens inside the timed call; no output caching or replay. The warm-up runs on synthetic x in prepare() and its outputs are discarded.
- [x] No triton.autotune. The configs are static constants.
- [x] First line `# okc: problem=muse_bf16_proj` kept.

okc_lint output:

```
CLEAN: submission.py
```

`python3 -I -m py_compile`: OK, and the `__pycache__` it created in the worktree was removed.
