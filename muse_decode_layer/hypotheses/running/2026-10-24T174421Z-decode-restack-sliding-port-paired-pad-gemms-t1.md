---
problem: muse_decode_layer
slug: decode-restack-sliding-port-paired-pad-gemms-t1
created: 2026-10-24T174421Z
base: master
area: restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs)
change: >
  Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t)
  by a 3-way merge of their branches, resolving conflicts by hand (the code paths are disjoint).
  (1) sliding-fused-triton-decode1-port as is: sliding decode_1 +3.4-4.3%, _sliding_fused_kernel
  13.7 us replaces FA3 main + combine + glue. (2) paired-heads-pad-unpad-triton as is: full
  decode_16/32 and verify_16 +1-3%, elementwise glue launches gone. (3)
  bf16-shape-gated-triton-gemms-small-t, but gate its Triton split-K GEMMs to tokens == 1 ONLY. At
  T=8/16 they lost 2-4.5% despite faster kernel sums, so T>1 keeps master's GEMM path exactly.
  Known conflicts: (1) and (2) both insert helpers at the same spot and edit the kernel_fn
  attention block around qk_norm; (3) also touches the module top, the helper block and the qkv =
  _gemm(...) line where (1) hooks in. Keep each piece's own self-check. Nothing else changes.
expected:
  - shapes: "*layer=sliding*load=decode_1*"
    direction: faster
    why: sliding fused port plus the T=1 GEMMs
  - shapes: "*layer=full*load=decode_16*"
    direction: faster
    why: paired pad/unpad
  - shapes: "*load=verify_1*"
    direction: flat
    why: GEMMs gated off at T=8
  - shapes: "*layer=full*load=decode_1*"
    direction: faster
    why: full decode_1 faster (T=1 GEMMs)
  - shapes: "*layer=full*load=verify_16*"
    direction: faster
    why: full verify_16 faster (paired)
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T174912Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T174912Z
base_sha: c8af06125b23fb943b7a8f7c26fc6b9a677dd1e0
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1
worktree: .worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1
mode: null
implemented_at: 2026-10-24T175146Z
impl_commit: 62dae0d15ab7bbba9bbeece153ca2507bb23ba54
submission_sha256: 72144e40a5588f9fef68d5f5da0390f04691c0e1d5b98fbf39879a5e81536fae
norm_sha256: 7e9bd9b4762741435e4d182bf2d80b3db0a8acfba775a24bcc79c6174d0cf06e
submission_blob: a37cb6c8e76c4e703a2e0c45e61f2b0a5f8418ea
lint_warnings: 3
---

# decode-restack-sliding-port-paired-pad-gemms-t1

## Change

Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand (the code paths are disjoint). (1) sliding-fused-triton-decode1-port as is: sliding decode_1 +3.4-4.3%, _sliding_fused_kernel 13.7 us replaces FA3 main + combine + glue. (2) paired-heads-pad-unpad-triton as is: full decode_16/32 and verify_16 +1-3%, elementwise glue launches gone. (3) bf16-shape-gated-triton-gemms-small-t, but gate its Triton split-K GEMMs to tokens == 1 ONLY. At T=8/16 they lost 2-4.5% despite faster kernel sums, so T>1 keeps master's GEMM path exactly. Known conflicts: (1) and (2) both insert helpers at the same spot and edit the kernel_fn attention block around qk_norm; (3) also touches the module top, the helper block and the qkv = _gemm(...) line where (1) hooks in. Keep each piece's own self-check. Nothing else changes.

## Expected

- `*layer=sliding*load=decode_1*` faster: sliding fused port plus the T=1 GEMMs
- `*layer=full*load=decode_16*` faster: paired pad/unpad
- `*load=verify_1*` flat: GEMMs gated off at T=8
- `*layer=full*load=decode_1*` faster: full decode_1 faster (T=1 GEMMs)
- `*layer=full*load=verify_16*` faster: full verify_16 faster (paired)

## Risk

Restack of measured pieces; expected headline about 1.18 vs 1.165 (harvest analysis). Run as a scored experiment.

## Adversarial review

Reviewed 2026-10-24T174912Z by an adversarial subagent (model opus).

**Verdict: proceed.** Each piece is gated off at shapes where another piece runs, except sliding decode_1, and all three were built on master's exact submission blob (cc466fce), so their single runs already serve as per-piece controls. Projected headline is ~1.177 (T=1 pieces only) to ~1.181 (with paired). That is credible but sits inside the best's CI, so the readable signal is per-shape: sliding decode_1 should reach ~1.27x and full decode_1 ~1.20x.

### Justification
Base check: hyp branches sliding-fused-triton-decode1-port, paired-heads-pad-unpad-triton and bf16-shape-gated-triton-gemms-small-t all fork from a commit whose muse_decode_layer/submission.py blob is cc466fce, the same as master (post-expert-norm-triton-small-t, BEST job a017e10aec544c55). Their archived per-shape speedups (paired vs SGLang in the same run) can therefore be compared directly with master's two runs (a017e10a submit 1.1665, 9dcaec31 check 1.1635), and those two runs agree per shape to within 1.4%. Per-shape deltas vs the master geomean: port, sliding decode_1 +3.85% (profile total 149.2 -> 141.1 us: FA3 10.2 + combine 3.6 + qk-norm/cast/store/norm-gate glue replaced by _sliding_fused_kernel 13.7 us); GEMMs at T=1, sliding decode_1 +3.6% and full decode_1 +3.5% (cuBLAS nvjet qkv+pooler 32.6 + o 7.2 + unpool 5.1 + splitKreduce 5.0 = ~50 us of kernel time -> Triton 25.3 + 13.7 + 6.5 = ~45.5 us; profile total 149.2 -> 144.5). The two sliding decode_1 changes replace disjoint kernels (GEMMs vs the attention chain between them), so their savings should add. If anything the sum is slightly conservative, because each piece separately paid for losing cuBLAS/PDL overlap at the qkv->attention and attention->o boundaries, and stacked those boundaries are paid once. Additive model: sliding decode_1 1.178 -> ~1.27x (yours ~134-135 us), full decode_1 1.164 -> ~1.205x. Plugging those plus the paired run's full batch>1 rows into the TASK.md scoring (per-load weighted sums, geomean over 6 loads, SGLang times from master): restack with T=1 pieces only = 1.1774, with paired on all four full batch>1 shapes = 1.1813, master modelled = 1.165. The stated ~1.18 is credible. Almost all of the gain (~+1.0%) is the decode_1 load. Gating to tokens == 1 is right on the data. The original T<=16 gate hit decode_1 (T=1), verify_1 (T=8) and decode_16 (T=16), and those rows were +3.6/+3.5, -4.0/-3.4 and -4.1/-2.2 respectively, consistent across both layers. Scored T are only {1, 8, 16, 32, 128, 256}, so ==1 is equivalent to any gate below 8. The gains the GEMM run shows at T=32/128 (full decode_32 +8.2%, full verify_16 +3.2%) are on shapes where its code never runs, so they are noise; full decode_32 has bimodal per-rep ratios (1.12 vs 1.36) in master too. WEAKEST LINK: the paired pad/unpad piece. Its rows (full decode_16 +2.0%, decode_32 +1.0%, verify_16 +3.1%, verify_32 -0.4%) are no larger than the spread on untouched shapes in sibling runs (port run sliding verify_16 -2.5%, sliding verify_32 +2.2%; GEMM run full verify_16 +3.2% with no code active). It is worth ~+0.35% on the headline and cannot be attributed from one run. A second weak link is the headline itself: +1.0-1.3% over 1.1665 sits inside the best's CI [1.149, 1.184] and inside the run's own >= +/-1.5% interval, so a headline-only classifier will likely call it within-ci/unproven again even if every piece works. Judge it on the sliding decode_1 and full decode_1 rows. The `expected` list is incomplete. It omits full decode_1 faster (T=1 GEMMs, ~+3.5%), the largest-weight full-layer move, and full verify_16 faster (paired), the only shape FINDINGS credits the paired piece with. 'verify_1 flat' is right relative to master, not relative to the GEMM run.

### Confounders on the current kernel
- landed list is empty for this problem's lineage tracking, but master already carries post-expert-norm-triton-small-t (SMALL_T_POST_NORM path, _post_norm_combine_kernel at small T): all three pieces were measured on that exact blob (cc466fce), so it is a shared base, not a confounder.
- Within the restack, sliding decode_1 is the only shape where two pieces both run (sliding port + T=1 GEMMs). They replace disjoint kernels and the qkv tensor the fused kernel reads is the same contiguous [1, 2304] bf16 output either way (fused kernel takes qkv.stride(0); Triton qkv GEMM writes contiguously). Expect additivity, measured by sliding decode_1 vs ~1.27x.
- paired pad/unpad is gated window == 0 and batch > 1, so it never runs on a shape where the port (sliding, batch 1, q_len 1) or the T==1 GEMMs run: no runtime interaction. Its _paired_norm_gate_kernel reads the gate from qkv with qkv.stride(0) and assumes unit inner stride. That holds on master's _gemm path, which is the only path it sees at batch > 1.

### Interference and A/B needs
No companion warranted. The three pieces' own runs on the identical base blob already isolate each sub-change: port alone (12198572363b49a4), paired alone (a92b64be4c6547d3), and GEMMs alone at T<=16 (6076ae1f250143aa, whose T=1 rows are the T==1-gate control). The restack's result decomposes by shape. Sliding decode_1 tests additivity of port + GEMMs against two measured singles. Full decode_1 should reproduce the GEMM run's +3.5%. Full batch>1 should reproduce the paired run. verify_1 and decode_16 should return to master (confirming the gate). A further control would only re-measure code already measured, and identical code returns its cached result anyway. The paired piece's ambiguity is noise, not interference: a companion cannot fix it; only stacking a larger full-layer change on top can.

### Open questions
- Why did the Triton GEMMs lose wall time at T=8/16 while their kernel sums were lower? Candidates: lost PDL overlap between cuBLAS and the next sglang kernel, or the 128-CTA Triton pooler starving the side-stream router that overlaps it. The same mechanism is present at T=1 and is netted into the +3.5%. If it is the router overlap, a T=1 restack is unaffected; if it is PDL, the stacked Triton->Triton chain at sliding decode_1 might be slightly super-additive.
- If the headline lands inside the CI (likely, ~1.177-1.181 vs best 1.1665 [1.149, 1.184]), should promotion use the per-shape decode_1 rows (expected well outside noise: sliding ~+7.5%, full ~+3.5%) rather than the headline? That decision belongs to the harvester.
- Should the T=1 GEMM warmup in _prepare_small_t_gemms be trimmed to m=1 now that only T==1 routes to it? It is harmless as is (untimed prepare), but leaving the 1..16 workspaces makes the code claim more than it uses.

## Implementation notes

Worktree file: `.worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1/muse_decode_layer/submission.py` (base blob cc466fce; uncommitted). Diff stat: 1 file, +756 / -72.

How it was merged: `git merge-file` of port + paired over cc466fce conflicted in two hunks (the helper insertion point and the whole re-indented attention block), as the review predicted. Resolved by hand: the file is the port branch blob (e33a32aa) verbatim, then the paired branch's two Triton kernels (`_pair_pad_q_kernel`, `_paired_norm_gate_kernel`) inserted after `_sliding_forward`, then the GEMM branch's helper block (`_bf16_splitk_kernel` through `_prepare_small_t_gemms`) inserted before `prepare`, then the kernel_fn edits of the paired and GEMM branches re-applied.

- Port (1): unchanged from its branch (docstring bullet, `_INV_FREQ`/`_counters`/`_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue`, `_sliding_fused_kernel`, `_sliding_forward`, the `if cfg["window"] and batch == 1 and length == 1: o = _sliding_forward(...) else:` re-indent).
- Paired (2): both kernels byte-identical to the branch. Its two kernel_fn edits (the `_pair_pad_q_kernel` launch with its `assert qh.stride(2) == 1` replacing torch.zeros + slice copies, and the `if paired: _paired_norm_gate_kernel ... else: o = o_norm_gate(...)` epilogue) sit inside the port's `else:` branch, one indent deeper. `diff -w` of the merged kernel_fn against the paired branch's kernel_fn shows only the port and GEMM hunks.
- GEMMs (3): kernels, shape guards, `_fallback` and its once-per-shape log kept unchanged. Changes from the branch: `SMALL_T_TRITON_GEMM = 1` and `small_t = tokens == SMALL_T_TRITON_GEMM` (was `1 <= tokens <= 16`); the unpooler-slice route is now gated `small_t and _unpool_shape_ok(...)` too (in the branch it ran at every T <= 8, i.e. also at verify_1 T=8, so leaving it would have broken "T>1 takes master's path"). The T>1 unpooler call is `_gemm(expert_output, unpool_w)` with `unpool_w = dense["unpooler"][:UNPOOL_SLICE]`, the same call as master's. Warmup trimmed to m=1 (`for m in (SMALL_T_TRITON_GEMM,)`), as the review allowed; the workspaces keep their 16-row size, since the kernels are verbatim and use WS_M=16 / BLOCK_M=16. Comments and docstring bullet updated to T == 1. `_prepare_small_t_gemms(state)` is called in prepare after `del freqs`, as in the branch.
- `o` is defined on all three routes: sliding batch-1 decode (`_sliding_forward`), paired (`torch.empty` + `_paired_norm_gate_kernel`), plain (`o_norm_gate`). The o / pooler / unpooler GEMM selection follows after the attention block at the outer indent, so it sees `o` from any route.
- T>1: `small_t` is False, so every GEMM is `_gemm(...)` with master's arguments. The only T>1 differences from master are the paired-path kernels (full layer, batch > 1), as intended.
- Self-checks kept: paired `assert qh.stride(2) == 1`, GEMM `_*_shape_ok` guards with `_fallback` logging, port's config-only gate and `_num_splits`.

Deviations: none beyond the specified T == 1 gate, the unpooler gate that it requires, and the warmup trim.

Uncertainties: none tested (no GPU on this node). The port's `_counters` allocates its zeroed ticket buffer lazily on the first eager call (unchanged from the port branch, which passed); it is keyed on (batch, hkv, device), never on data values.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1 diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's call graph (the three synchronize calls are in prepare / `_prepare_small_t_gemms`; `_fallback` prints shapes only, never values); routing on `tokens`, `batch`, `q_len`, `window`, `rope_theta` and tensor shapes/strides only.
- [x] Nothing derived from `data` is cached across calls: `_INV_FREQ` is keyed on (theta, d, device) from config, `_counters` on (batch, hkv, device) and holds only self-reset tickets, the GEMM split-K workspaces are fully rewritten for rows < M every launch, and `_FALLBACK_SEEN` stores shapes only.
- [x] Weights: no new repack; the Triton GEMMs read the original BF16 `dense` weights in place (the unpooler slice by stride); NVFP4 experts untouched.
- [x] Activations BF16 where SGLang keeps BF16 (GEMMs bf16 in / fp32 accumulate / bf16 out; paired epilogue stores bf16); KV cache FP8; queries rounded to FP8 E4M3 (paired pad kernel, `tl.float8e4nv`), and the port's kernel as in its branch.
- [x] Attention sees every key: FA3 full context / whole window unchanged; the port's sliding kernel attends the full 2048-token window in 16 splits (unchanged from its branch).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: imports unchanged from master.
- [x] All work happens inside the timed call; no output caching or replay: prepare only allocates workspaces/counters and warms kernels on synthetic random inputs.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept; no `triton.autotune`.

okc_lint output:

```
warn  host-sync    L881: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L885: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L928: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1/muse_decode_layer/submission.py
```

`python3 -I -m py_compile`: OK; `__pycache__` removed.
