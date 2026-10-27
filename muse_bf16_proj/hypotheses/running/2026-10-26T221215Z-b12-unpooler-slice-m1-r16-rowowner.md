---
problem: muse_bf16_proj
slug: b12-unpooler-slice-m1-r16-rowowner
created: 2026-10-26T221215Z
base: master
area: unpooler GEMM (sliced n=1024, m=1, bulk12 12-bit row-owner)
change: >
  Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024
  rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. CUDA (_B12_CUDA): two new
  instances of the unchanged b12_kernel template, B12Inst<16, 4096, 8, 1, 4> (inst 6) and
  B12Inst<16, 4096, 8, 1, 8> (inst 7): CKK 512, 8 chunks of 12 KB per CTA, all 8 in flight from
  the start, grid 1024/16 = 64 CTAs, dynamic smem 108,800 B (every static_assert holds: CHUNK
  12288, NS 32 divisible by CW, RING 10496 a multiple of 128), plus their b12_run, b12_unpack
  (inst 6/7 -> B12Inst<16,4096,...>::unpack) and b12_setup switch cases; bump B12_EXT_NAME.
  Python: B12_CELLS['unpooler'] = ('U12s', 1024, 4096, 16, ((6, 8, 1, 4), (7, 8, 1, 8)),
  'b12_unpool_ok'); in _b12_prepare and in the second state's re-encode take w =
  state['unpooler'][:1024] (a contiguous view) for this cell instead of state[layer]; _b12_forward
  already allocates (1, spec n); B12_FLAGS / NEW_CELL_FLAGS / the A/B inheritance pick the cell up
  from B12_CELLS, and _ab_component already maps the AB case (unpooler, 1024, 4096, 1) to the b12
  flag when it is on. kernel_fn: move the _b12_route_ok check above the _ouns_route_ok branch (b12
  routes the unpooler only with its own flag on, cfg m == 1 and n == 1024). Gate: the incumbent is
  today's route, the ouns cluster / DSMEM split-K kernel (ouns_kernel<8,64,512,8,4>, 4.4 us, row
  4.97 us in 3834ceec). Because the slice fits in L2, add flush_mode='read' to _cell_gate
  (flush.view(torch.int32).sum() in place of flush.zero_(), the judge's kind of flush; default
  unchanged for every other caller), gate each instance under both flushes and route the faster
  one only if both its write and read ratios are <= 0.98, then the usual profiler kernel-name
  assertion. One PATH CHOSEN line with both ratios. m=2..8 stay on ouns; the pooler / qkv / o
  cells are untouched.
expected:
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: "6.3 MB of 12-bit weights on 64 whole-K row-owner CTAs (no DSMEM reduce) vs ouns's 8.4 MB cluster split-K: 4.97 -> ~4.0-4.5 us if the read-flush gate routes it"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: m=8 stays on ouns
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "restack-unpooler-slice-cluster-dsmem-m8: that cluster/DSMEM split-K kernel (ouns) is today's m<=8 slice route; this replaces only its m=1 cell with the judge-proven bulk12 row-owner on 12-bit weights (6.3 MB instead of 8.4 MB, whole K per CTA, no DSMEM reduce), gated under a judge-like read flush"
review: proceed
status: running
review_at: 2026-10-27T095240Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-27T095240Z
base_sha: c59f4fea68a3a426326191184157123641faca72
base_blob: d51c745a0e43d20ebf2cbb3d582f3f8ead72f039
branch: hyp/muse_bf16_proj/b12-unpooler-slice-m1-r16-rowowner
worktree: .worktrees/muse_bf16_proj/b12-unpooler-slice-m1-r16-rowowner
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-27T095656Z
impl_commit: cdf7101a444e56e30d6babe1c037b5b250b3027f
submission_sha256: 78bba9d11ceb48cb8afcb1cf3378ae592168d20fdf9c7eb90377c9699f1e36f6
norm_sha256: 00782f3ccf0ac5c7b8470d5d2aa945eadfddd02feff3c8fd929a1d9f76a2194e
submission_blob: 5141523a4c58a84f31a3beef1c28a663e34ea225
lint_warnings: 45
attempt_file: 2026-10-27T095656Z-check-b12-unpooler-slice-m1-r16-rowowner.md
prepared_at: 2026-10-27T095656Z
job: a642424734f54365
dispatched_at: 2026-10-27T095706Z
---

# b12-unpooler-slice-m1-r16-rowowner

## Change

Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024 rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. CUDA (_B12_CUDA): two new instances of the unchanged b12_kernel template, B12Inst<16, 4096, 8, 1, 4> (inst 6) and B12Inst<16, 4096, 8, 1, 8> (inst 7): CKK 512, 8 chunks of 12 KB per CTA, all 8 in flight from the start, grid 1024/16 = 64 CTAs, dynamic smem 108,800 B (every static_assert holds: CHUNK 12288, NS 32 divisible by CW, RING 10496 a multiple of 128), plus their b12_run, b12_unpack (inst 6/7 -> B12Inst<16,4096,...>::unpack) and b12_setup switch cases; bump B12_EXT_NAME. Python: B12_CELLS['unpooler'] = ('U12s', 1024, 4096, 16, ((6, 8, 1, 4), (7, 8, 1, 8)), 'b12_unpool_ok'); in _b12_prepare and in the second state's re-encode take w = state['unpooler'][:1024] (a contiguous view) for this cell instead of state[layer]; _b12_forward already allocates (1, spec n); B12_FLAGS / NEW_CELL_FLAGS / the A/B inheritance pick the cell up from B12_CELLS, and _ab_component already maps the AB case (unpooler, 1024, 4096, 1) to the b12 flag when it is on. kernel_fn: move the _b12_route_ok check above the _ouns_route_ok branch (b12 routes the unpooler only with its own flag on, cfg m == 1 and n == 1024). Gate: the incumbent is today's route, the ouns cluster / DSMEM split-K kernel (ouns_kernel<8,64,512,8,4>, 4.4 us, row 4.97 us in 3834ceec). Because the slice fits in L2, add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of flush.zero_(), the judge's kind of flush; default unchanged for every other caller), gate each instance under both flushes and route the faster one only if both its write and read ratios are <= 0.98, then the usual profiler kernel-name assertion. One PATH CHOSEN line with both ratios. m=2..8 stay on ouns; the pooler / qkv / o cells are untouched.

## Expected

- `layer=unpooler load=decode_1 m=1` faster: 6.3 MB of 12-bit weights on 64 whole-K row-owner CTAs (no DSMEM reduce) vs ouns's 8.4 MB cluster split-K: 4.97 -> ~4.0-4.5 us if the read-flush gate routes it
- `layer=unpooler load=verify_1 m=8` flat: m=8 stays on ouns

## Risk

Medium. New template instances of a judge-proven kernel, so a device fault is unlikely, but a compile error would drop the whole b12 module and with it the landed pooler m=1 win (decode_1 pooler 14.8 -> 17.7 us): run it as a check, never a direct submit, after a quick read-only audit of the instance geometry. Only 64 CTAs stream (96 KB each, one per SM): it wins only if one SM pulls its 96 KB faster than ouns's 128 CTAs pull 64 KB each and reduce through DSMEM; the read-flush gate keeps ouns otherwise. Sizing: decode_1 is 0.40 %/us, so -0.5 to -1 us is +0.2-0.4%, below the 1.24% one-run margin alone. Decode-portable: decode runs the same slice at T=1.

## Adversarial review

Reviewed 2026-10-27T095240Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is plausible and the run can be read in isolation. The in-run read-flush gate against ouns, the PATH CHOSEN line and the decode_1 unpooler profile row (b12_kernel<16,4096,...> vs ouns 4.4-4.6 us) attribute the result without any companion. Part of the change paragraph is out of date against today's master and must be rebuilt as described below, or the cell silently stays off or the whole import fails.

### Justification
Mechanism plausible: the unpooler slice (n=1024, k=4096, 8.4 MB raw, floor 1.7 us) runs on ouns_kernel<8,64,512,8,4> at 4.4-4.6 us; bulk12 shrinks the stream to 6.3 MB and drops the DSMEM reduce. Geometry checked against the b12_kernel static_asserts: CKK 512, NCHUNK 8, NS 32, CHUNK 12288, RING 10496, SMEM 108,800 B; 64 CTAs. Weakest link: the landed o cell with the same 96 KB per CTA reaches only ~2.7 TB/s (per-CTA ramp and ingest bound), so the slice may land near o's ~4.6 us and tie ouns. A no-route result is a real possibility. Out of date against master 8f233e88: (1) do not re-add flush_mode='read' to _cell_gate (CELL_FLUSH_MODES, B12_GATE_FLUSH='read' exist); (2) B12_CELLS variants are 5-tuples (inst, S, MINB, CW, R): ('U12s', 1024, 4096, 16, ((6, 8, 1, 4, 16), (7, 8, 1, 8, 16)), 'b12_unpool_ok'); (3) B12_INST_R must gain 6: 16 and 7: 16 (the import-time assert loop raises KeyError otherwise and kills the submission); (4) use master's _b12_pick unchanged (home-R variant on read <= 0.98), not the hypothesis's write AND read rule; (5) bump B12_EXT_NAME. kernel_fn must test _b12_route_ok before _ouns_route_ok.

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (ouns at unpooler slice m<=8) is the incumbent the gate compares against: read the result as 'b12 beats ouns at m=1'.
- bulk12 m=1 landing (5325ea40): new instances go into the same load_inline module; a compile error drops all three b12 cells, so check the '[b12] build' line first.
- read-flush gate landing (8f233e88): qkv_gate and o b12 cells route near the 0.98 threshold, so gate noise can flip other decode_1 m=1 rows; attribute only through the unpooler decode_1 profile row and its gate line.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive: no shared kernel or path.

### Interference and A/B needs
No companion needed: inert on every shape except unpooler cfg m==1 (n==1024) with its own flag on; measured in-run by the gate (flag on vs ouns, read flush), the profiler's kernel assertion and the judged decode_1 unpooler row.

### Open questions
- If the read gate lands near 1.0, is a 2-CTA cluster K-split of the b12 kernel the follow-up, or does that rebuild ouns on 12-bit bytes?
- Print the per-variant read-flush cand us next to the incumbent us for this cell, and include it in the end-of-prepare re-print.
- Does the judged decode_1 unpooler row track the read-flush gate for an L2-sized weight? Expect roughly half the gate's gain (o gated 0.92-0.93, judged 0.97).


## Implementation notes

Implemented in the worktree's `muse_bf16_proj/submission.py` only (uncommitted; base c59f4fea, blob d51c745a). Diff: 1 file, +39 / -17. Built against today's master and following the adversarial review's corrections, not the out-of-date parts of the change paragraph:

- **CUDA (`_B12_CUDA`)**: added two instances of the unchanged `b12_kernel` template, `B12Unp4 = B12Inst<16, 4096, 8, 1, 4>` (inst 6) and `B12Unp8 = B12Inst<16, 4096, 8, 1, 8>` (inst 7). They are wired into `b12_run` (cases 6 and 7), `b12_unpack` (inst 6/7 -> `B12Unp4::unpack`, the same `b12_unpack_kernel<16, 4096>` for both) and `b12_setup` (cases 6 and 7). Kernel template unchanged. I checked every static_assert by hand and with a short script: CKK 512, NCHUNK 8, NS 32 (divisible by CW 4 and 8; KSW 8 and 4), MT 1, CHUNK 12288, XOFF 2304, RING 10496 = 82 x 128, epilogue scratch 512 B <= ring 98304 B, row-offset header 34 B <= 256 B, S 8 in [2, 8]. Dynamic smem is 108,800 B, grid 64, 160 or 288 threads, MINB 1, so no carveout call is needed.
- **`B12_EXT_NAME`**: `muse_bf16_bulk12_m1_v1` -> `muse_bf16_bulk12_m1_v2`.
- **`B12_CELLS`**: added `"unpooler": ("U12s", 1024, 4096, 16, ((6, 8, 1, 4, 16), (7, 8, 1, 8, 16)), "b12_unpool_ok")`. Variants are 5-tuples matching the existing entries, home R is 16 and both variants are home-R, so no challenger logic applies.
- **`B12_INST_R`**: gained `6: 16, 7: 16`. I ran the import-time assert loop without torch (exec of the `B12_CELLS`..loop slice) and it passes for all four cells. The trailing comment moved to its own line to keep line length.
- **New `_b12_weight(state, layer)`**: returns `state[layer]`, except for the unpooler, where it returns `state["unpooler"][:1024]` (a contiguous row-slice view). Both the judge-state loop (`w = _b12_weight(state, layer)`, which feeds the shape check, encode, `_cell_check` and gates) and the second-state (A/B) re-encode path (`_b12_encode_into(..., _b12_weight(state, layer), ...)`) use it. As a result, b12_run's `wc.numel() == n * K * 3 / 2` TORCH_CHECK sees n = 1024 in both states.
- **`kernel_fn`**: the `_b12_route_ok` / `_b12_forward` pair now comes before the `layer == "unpooler" and _ouns_route_ok` branch. This only changes behaviour for the unpooler, because the ouns branch is unpooler-only. b12 routes the unpooler only with `b12_unpool_ok` on, cfg m == 1, n == 1024, k == 4096 and x a contiguous aligned bf16 [1, 4096]. m = 2..8 still reaches ouns.
- **Gate / pick**: no change. Master's `_cell_gate` with `B12_GATE_FLUSH = "read"` plus the write diagnostic, master's `_b12_pick` (home-R best read ratio <= `_b12_max_ratio(1)` = CELL_MAX_RATIO 0.98), and the profiler kernel-name assertion (exactly one `b12_kernel`). The md's own flush_mode / write-AND-read rule was deliberately not implemented (review corrections 1 and 4). The incumbent the gate times with the flag off is today's unpooler m=1 route (ouns when `ouns_cl` is set, because `_ouns_prepare` runs before `_b12_prepare`).
- **Open-question guidance (print)**: no new code was needed. The generic per-variant `_decision` gate line already prints `read <ratio> (<cand us> / <incumbent us>) write <ratio> (<cand> / <incumbent> us)` for the new cell, and so does the `[prepare-time] gates ... read-flush cand X us ... vs incumbent Y us` line. The gate lines and the `PATH CHOSEN U12s unpooler m=1` line go through `_decision`, so the end-of-prepare `[prepare] DECISIONS` re-print includes them, the same as for the other b12 cells.
- **Cosmetic**: the `_b12_prepare` docstring and the section-(10) tick label/comment now name U12s, and the docstring was reflowed to line length.
- **A/B**: `_ab_component("unpooler", 1, st)` already returns `b12_unpool_ok` when the flag is on, through the generic B12_CELLS branch. `_ab_flags` / `NEW_CELL_FLAGS` / `fl["_b12_inst"]` pick the cell up from B12_CELLS. The AB case `("unpooler", 1024, 4096, 1)` already exists, and its reference is `wf["unpooler"][:1024]`.
- **CUDA syntax check**: `/home/user/git/musespark/rewrite/bin/cuda_syntax.py` does not exist on master. I ran the identical copy (27 worktree copies, one md5) at `.worktrees/b12p-fix/rewrite/bin/cuda_syntax.py`, which takes any `<problem>/submission.py`. `_B12_CUDA: ok` and `_BULK_CUDA: ok` on the modified file. `_OUNS_CUDA` and `_RC_CUDA` FAIL identically on the unmodified base: the stub lacks `cudaOccupancyMaxActiveClusters`, so these are stub gaps, not code errors, and this change does not touch them. The syntax check does not evaluate the device-only static_asserts, because the kernel body is under `__CUDA_ARCH__`. Those were checked by hand.
- **Deviation**: none from the review's binding corrections. The change paragraph's flush_mode addition and write-AND-read routing were dropped as instructed.
- **Uncertainty**: prepare time grows by one cell (two variants x check + two gates + encode of 4M elements). That is likely a few seconds, but the 300 s prepare budget was not re-measured here. The new instances compile into the same load_inline module, so a compile error would drop all b12 cells. Read the `[b12] build` line first.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/b12-unpooler-slice-m1-r16-rowowner diff`:

- [x] CUDA-graph capturable: kernel_fn only gains a reordered branch on cfg / shapes / prepare()-set state flags (`_b12_route_ok`). `_b12_forward` is one launch plus a `torch.empty` output, as for the existing b12 cells. No `.item()` / `.cpu()` / sync / tensor print in kernel_fn's call graph.
- [x] Nothing derived from `data` is cached across calls: the new state entries (`b12_wc_unpooler`, `b12_esc_unpooler`, base, inst) are derived from the weight in prepare() only.
- [x] Weights: lossless 12-bit repack of the bf16 unpooler slice in prepare() only. The host and device (`b12_unpack`, the kernel's own decode) round trips are checked bit-exact, and escapes carry the true bf16 bits. BF16 stays BF16 losslessly.
- [x] Activations BF16 where SGLang keeps BF16: x is read as bf16, accumulation is fp32 via mma.sync bf16 and the output is bf16. The kernel is unchanged.
- [x] n/a: attention / KV cache / queries rules: this problem is dense projections, with no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds none. The `load_inline` build is the existing one under a new name.
- [x] All work happens inside the timed call; no output caching or replay: the unpooler m=1 output is computed per call by `b12_run`.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept. No triton.autotune was added.
- [x] Lint run, py_compile OK, `__pycache__` removed.

```
$ python3 -I .claude/skills/churnsubs/bin/okc_lint.py .worktrees/muse_bf16_proj/b12-unpooler-slice-m1-r16-rowowner/muse_bf16_proj/submission.py --problem muse_bf16_proj
(45 warn lines, all pre-existing prepare()-only host-sync / uint8-buffer notes at lines outside kernel_fn's call graph:
 L1443..L4159 host-sync "fine in prepare()", dtype warns on flush / workspace / b12 encode uint8 buffers)
OK with warnings: .worktrees/muse_bf16_proj/b12-unpooler-slice-m1-r16-rowowner/muse_bf16_proj/submission.py
lint exit 0
$ python3 -I -m py_compile .../submission.py   -> exit 0 (__pycache__ removed)
```
