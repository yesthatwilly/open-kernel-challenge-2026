---
problem: muse_decode_layer
slug: dl3-piece-a-gate-stride-check-fix
created: 2026-10-25T145938Z
base: master
area: full decode_1 attention side (dl3 piece A enable)
change: >
  Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel
  Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet
  qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed
  job 34a9be06918d448b: its log prints 'sk gate plumbing n=4097: separate-gate output identical to
  qkv-gate output: True' (and the same for n=30000) yet PATH CHOSEN says 'piece A ... (fallback:
  self-check: sk gate plumbing n=4097: outputs differ (or gate stride 2304))'. The outputs
  matched; the failure is the stride clause. In _dl3_check_a the test gate is built as gate_c =
  sqkv[:, SK_QKV_GATE_OFF:].contiguous() from a ONE-row sqkv [1, 2304]; PyTorch treats a size-1
  leading dim as contiguous whatever its stride, so .contiguous() returns the view unchanged with
  stride(0) == 2304, and 'gate_c.stride(0) != DL3_GATE_STRIDE' (1024) is always true. The fix, in
  _dl3_check_a only: build the test gate as a fresh tensor with real [rows, 1024] strides, gate_c
  = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device) followed by
  gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:]), so the separate-gate launch uses row stride 1024
  exactly as dl2_qkv's live gate does (that live gate already printed 'gate stride 1024' in the Q
  check). Keep the stride clause and the torch.equal comparison as they are. Nothing else changes:
  kernel_fn, the CUDA modules, piece A's gate (tokens == 1, full layer, sk variant 3, state flag),
  the layer-level eager and alternating-state checks (_dl3_layer_checks with A=1), and selector A
  (new chain must beat the parent chain by DL3_MARGIN 1.02 or the incumbent stays) all run
  unmodified and still decide whether A goes live. Print one extra line after the plumbing check
  with gate_c.stride() so the log shows the corrected stride. If any later check or selector A
  fails, the run measures master exactly (fallback per piece).
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "piece A replaces the attention side around the sk kernel at full decode_1 (nvjet qkv 128x8 splitK 12.4 us + splitKreduce + FusedQKNorm + 2-3 cast/store nodes + nvjet o 64x8 6.8 us) with Q (8.3 us measured in s2) and O (6.5 us); s2 measured custom Q+O 18.6 vs 27.3 us and full decode_1 -3.3% (z 5.8) on an older base; expect about -4 to -8 us of 130 us. Read the dl2_qkv / dl2_o kernel rows and the absence of the nvjet qkv/o rows in the full decode_1 profile, and the 'PATH CHOSEN: dl3 piece A' line"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: "control: piece A is gated to full layers; sliding decode_1 keeps _sliding_fused_kernel and master's GEMMs, byte-identical path"
  - shapes: "*load=decode_16 *"
    direction: flat
    why: "control: T > 1 untouched"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: T > 1 untouched"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "control: T = 8 untouched"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "control: untouched"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# dl3-piece-a-gate-stride-check-fix

## Change

Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed job 34a9be06918d448b: its log prints 'sk gate plumbing n=4097: separate-gate output identical to qkv-gate output: True' (and the same for n=30000) yet PATH CHOSEN says 'piece A ... (fallback: self-check: sk gate plumbing n=4097: outputs differ (or gate stride 2304))'. The outputs matched; the failure is the stride clause. In _dl3_check_a the test gate is built as gate_c = sqkv[:, SK_QKV_GATE_OFF:].contiguous() from a ONE-row sqkv [1, 2304]; PyTorch treats a size-1 leading dim as contiguous whatever its stride, so .contiguous() returns the view unchanged with stride(0) == 2304, and 'gate_c.stride(0) != DL3_GATE_STRIDE' (1024) is always true. The fix, in _dl3_check_a only: build the test gate as a fresh tensor with real [rows, 1024] strides, gate_c = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device) followed by gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:]), so the separate-gate launch uses row stride 1024 exactly as dl2_qkv's live gate does (that live gate already printed 'gate stride 1024' in the Q check). Keep the stride clause and the torch.equal comparison as they are. Nothing else changes: kernel_fn, the CUDA modules, piece A's gate (tokens == 1, full layer, sk variant 3, state flag), the layer-level eager and alternating-state checks (_dl3_layer_checks with A=1), and selector A (new chain must beat the parent chain by DL3_MARGIN 1.02 or the incumbent stays) all run unmodified and still decide whether A goes live. Print one extra line after the plumbing check with gate_c.stride() so the log shows the corrected stride. If any later check or selector A fails, the run measures master exactly (fallback per piece).

## Expected

- `*layer=full load=decode_1 *` faster: piece A replaces the attention side around the sk kernel at full decode_1 (nvjet qkv 128x8 splitK 12.4 us + splitKreduce + FusedQKNorm + 2-3 cast/store nodes + nvjet o 64x8 6.8 us) with Q (8.3 us measured in s2) and O (6.5 us); s2 measured custom Q+O 18.6 vs 27.3 us and full decode_1 -3.3% (z 5.8) on an older base; expect about -4 to -8 us of 130 us. Read the dl2_qkv / dl2_o kernel rows and the absence of the nvjet qkv/o rows in the full decode_1 profile, and the 'PATH CHOSEN: dl3 piece A' line
- `*layer=sliding load=decode_1 *` flat: control: piece A is gated to full layers; sliding decode_1 keeps _sliding_fused_kernel and master's GEMMs, byte-identical path
- `*load=decode_16 *` flat: control: T > 1 untouched
- `*load=decode_32 *` flat: control: T > 1 untouched
- `*load=verify_1 *` flat: control: T = 8 untouched
- `*load=verify_16 *` flat: control: untouched
- `*load=verify_32 *` flat: control: untouched

## Risk

Risk: this turns on code that has never run in the judge's timed path. Its kernels passed their own checks in the landed job (Q code gaps 0/1/0, O rel L2 1e-5, sk gate output identical), but the A=1 layer-level checks and selector A have not run yet. Both still gate it, and a miss falls back to master. A scored submit that fails a test costs one submit and not the board. Upside is about 1.4% on the decode_1 load (one weight-16 shape), about 0.25% on the headline, so judge it on the full decode_1 row and the PATH CHOSEN line, not the headline. Reviewer: confirm that _dl3_layer_checks with A=1 exercise kv_scale 0.25 and 1.0, the alternating two-state graph replays and the tickets (cnt_q, cnt_o, sk) reset. Also confirm that kernel_fn's piece-A gate reads only config and the state flag.
