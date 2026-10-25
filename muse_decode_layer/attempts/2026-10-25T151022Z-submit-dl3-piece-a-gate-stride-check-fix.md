# muse_decode_layer · 2026-10-25T151022Z · submit · dl3-piece-a-gate-stride-check-fix

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T145938Z-dl3-piece-a-gate-stride-check-fix.md` (branch `hyp/muse_decode_layer/dl3-piece-a-gate-stride-check-fix`).

- **Job:** `933d3f825a044d36` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `b7cda2f9`

## Goal
Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed job 34a9be06918d448b: its log prints 'sk gate plumbing n=4097: separate-gate output identical to qkv-gate output: True' (and the same for n=30000) yet PATH CHOSEN says 'piece A ... (fallback: self-check: sk gate plumbing n=4097: outputs differ (or gate stride 2304))'. The outputs matched; the failure is the stride clause. In _dl3_check_a the test gate is built as gate_c = sqkv[:, SK_QKV_GATE_OFF:].contiguous() from a ONE-row sqkv [1, 2304]; PyTorch treats a size-1 leading dim as contiguous whatever its stride, so .contiguous() returns the view unchanged with stride(0) == 2304, and 'gate_c.stride(0) != DL3_GATE_STRIDE' (1024) is always true. The fix, in _dl3_check_a only: build the test gate as a fresh tensor with real [rows, 1024] strides, gate_c = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device) followed by gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:]), so the separate-gate launch uses row stride 1024 exactly as dl2_qkv's live gate does (that live gate already printed 'gate stride 1024' in the Q check). Keep the stride clause and the torch.equal comparison as they are. Nothing else changes: kernel_fn, the CUDA modules, piece A's gate (tokens == 1, full layer, sk variant 3, state flag), the layer-level eager and alternating-state checks (_dl3_layer_checks with A=1), and selector A (new chain must beat the parent chain by DL3_MARGIN 1.02 or the incumbent stays) all run unmodified and still decide whether A goes live. Print one extra line after the plumbing check with gate_c.stride() so the log shows the corrected stride. If any later check or selector A fails, the run measures master exactly (fallback per piece).

## What changed, and where
Base 2f268b59 has the same submission.py blob as master 9716bda4, so no merge was needed. Single change in `_dl3_check_a` (muse_decode_layer/submission.py, 4 insertions, 1 deletion): the test gate is now `torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device)` plus `gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:])`, replacing `.contiguous()`. The stride clause and torch.equal comparison are unchanged. Added one print after the plumbing line with `gate_c.stride()`. The per-piece `PATH CHOSEN: dl3 piece A/B/C` lines already exist in `_dl3_layer_checks` (one per piece, each with w2_mode tail and the fallback reason), so nothing was added there. Review items: all confounder and open-question items are reading guidance for the harvest (check the B/C PATH CHOSEN lines and w2_mode before attributing decode_1 movement); no build action needed. Deviations: none.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: change is prepare-only (self-check); kernel_fn untouched.
- [x] Nothing derived from `data` is cached across calls: n/a, no state change.
- [x] Weights: unchanged.
- [x] Activations/KV/queries precision: unchanged.
- [x] Attention sees every key: unchanged.
- [x] No timing patches, harness I/O, module edits, reference import, child processes: checked in the git diff.
- [x] All work inside the timed call: unchanged.

okc_lint: OK with warnings (only host-sync warnings in prepare(), pre-existing). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Risk: this turns on code that has never run in the judge's timed path. Its kernels passed their own checks in the landed job (Q code gaps 0/1/0, O rel L2 1e-5, sk gate output identical), but the A=1 layer-level checks and selector A have not run yet. Both still gate it, and a miss falls back to master. A scored submit that fails a test costs one submit and not the board. Upside is about 1.4% on the decode_1 load (one weight-16 shape), about 0.25% on the headline, so judge it on the full decode_1 row and the PATH CHOSEN line, not the headline. Reviewer: confirm that _dl3_layer_checks with A=1 exercise kv_scale 0.25 and 1.0, the alternating two-state graph replays and the tickets (cnt_q, cnt_o, sk) reset. Also confirm that kernel_fn's piece-A gate reads only config and the state flag.

## Result (after the run)
- job `933d3f825a044d36` (submit), status done, passed True, tests 13/13
- speedup 1.3080x, 95% CI [1.2883, 1.3276], score 17294.37614869084 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.3017519911209459, "ci": [1.2822257112541318, 1.32127827098776], "job": "34a9be06918d448b"}; base result: muse_decode_layer/hypotheses/done/neutral/dense-chain-s3-t1-q-sk2-o-route-unpool.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.result.json`, `muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.patch`; branch `hyp/muse_decode_layer/dl3-piece-a-gate-stride-check-fix`
