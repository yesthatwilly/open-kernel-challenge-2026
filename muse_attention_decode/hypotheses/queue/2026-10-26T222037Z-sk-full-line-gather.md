---
problem: muse_attention_decode
slug: sk-full-line-gather
created: 2026-10-26T222037Z
base: master
area: full-layer attention (decode 1<B<=64, stream-K kernel K/V gather mapping)
change: >
  Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master; that branch
  leaves _SK_CUDA byte-identical to master, so this applies cleanly on top). ONE change, in
  _SK_CUDA's splitkv_attn_sk_kernel, inside the issue_tile lambda (the K/V copy of one 64-token
  tile into ring stage nn % STAGES): the thread-to-copy mapping. Today tt = tid >> 2 and chunks c
  = (tid & 3) * 2 + j for j in {0, 1}, so one warp-wide cp.async covers 8 tokens x half of each
  32-byte sector (64 bytes of each 128-byte K or V line) and every line is requested by two
  instructions. New, the mapping splitkv-full-line-gather-mapping measured on stage-1 kernel A: c
  = tid & 7 per thread, and for j in {0, 1}: tt = (tid >> 3) + 32 * j; ok = lt * TILE + tt <
  seq_s[p_b]; row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE +
  tt] : 0; dst = tt * ROW + ((c ^ (tt & 7)) << 4); cp_async16(smem_u32(stp + dst), kc + row * ROW
  + c * 16, ok ? 16 : 0) and cp_async16(smem_u32(stp + TILE * ROW + dst), vc + row * ROW + c * 16,
  ok ? 16 : 0). Each thread still issues 4 x 16-byte cp.async per tile, each warp instruction now
  covers 4 whole 128-byte lines, and the swizzled stage image is byte-identical, so the q-slot
  copy (tid < QBYTES / 16 at a segment's first tile), commit/wait counts, compute, tickets and
  every instantiation (variants 0..7: single and two-level ticket, PSEUDO TOK=8 sliding verify
  B=1, PSEUDO TOK=1 sliding decode B=17..32) are otherwise unchanged. Do not touch kernel A
  (_SPLITKV_CUDA, the stage-1 fallback), the skF copy in _SKF_CUDA or the c320 module. Every
  import-time self-check (_sk_validate, _sk_validate_b1, the PSEUDO / sd checks and two-state
  replays) and band gate (_ps_gate_band, _sd_gate_band, the skF gate) re-runs unchanged and
  decides routes as before. Readout: the profile's splitkv_attn_sk_kernel time at full
  decode_16/32 (v3chain b4b20425: 108.3 / 226.3 us) and the [sk-diag] TB/s lines.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: kernel A with this mapping fell 111.8 -> 106.5 us in the profile (cfa8c89e vs 698d4cd3); sk kernel 108.3 us -> ~103 us; 0.107%/us
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: kernel A fell 245.9 -> 210.4 us (-14%); sk kernel 226.3 us -> ~205-215 us; 0.060%/us
  - shapes: "*layer=sliding load=verify_1 *"
    direction: flat
    why: PSEUDO TOK=8 route shares issue_tile; latency-bound at B=1, small move at most; its band gate re-decides
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: PSEUDO TOK=1 sk-window route shares issue_tile; _sd_gate_band re-decides
  - shapes: "*load=verify_16 *"
    direction: flat
    why: FA3 paths untouched
risk: low
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 1
kind: primary
controls_for: null
prior: "splitkv-full-line-gather-mapping: same remap, but that run changed stage-1 kernel A, which the stream-K kernel replaced before it could land (splitkv_attn_sk_kernel kept the half-line mapping); A's profile time fell 111.8->106.5 us (B=16) and 245.9->210.4 us (B=32). This ports it to the stream-K kernel on the v3chain base."
review: required
status: queued
---

# sk-full-line-gather

## Change

Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master; that branch leaves _SK_CUDA byte-identical to master, so this applies cleanly on top). ONE change, in _SK_CUDA's splitkv_attn_sk_kernel, inside the issue_tile lambda (the K/V copy of one 64-token tile into ring stage nn % STAGES): the thread-to-copy mapping. Today tt = tid >> 2 and chunks c = (tid & 3) * 2 + j for j in {0, 1}, so one warp-wide cp.async covers 8 tokens x half of each 32-byte sector (64 bytes of each 128-byte K or V line) and every line is requested by two instructions. New, the mapping splitkv-full-line-gather-mapping measured on stage-1 kernel A: c = tid & 7 per thread, and for j in {0, 1}: tt = (tid >> 3) + 32 * j; ok = lt * TILE + tt < seq_s[p_b]; row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE + tt] : 0; dst = tt * ROW + ((c ^ (tt & 7)) << 4); cp_async16(smem_u32(stp + dst), kc + row * ROW + c * 16, ok ? 16 : 0) and cp_async16(smem_u32(stp + TILE * ROW + dst), vc + row * ROW + c * 16, ok ? 16 : 0). Each thread still issues 4 x 16-byte cp.async per tile, each warp instruction now covers 4 whole 128-byte lines, and the swizzled stage image is byte-identical, so the q-slot copy (tid < QBYTES / 16 at a segment's first tile), commit/wait counts, compute, tickets and every instantiation (variants 0..7: single and two-level ticket, PSEUDO TOK=8 sliding verify B=1, PSEUDO TOK=1 sliding decode B=17..32) are otherwise unchanged. Do not touch kernel A (_SPLITKV_CUDA, the stage-1 fallback), the skF copy in _SKF_CUDA or the c320 module. Every import-time self-check (_sk_validate, _sk_validate_b1, the PSEUDO / sd checks and two-state replays) and band gate (_ps_gate_band, _sd_gate_band, the skF gate) re-runs unchanged and decides routes as before. Readout: the profile's splitkv_attn_sk_kernel time at full decode_16/32 (v3chain b4b20425: 108.3 / 226.3 us) and the [sk-diag] TB/s lines.

## Expected

- `*layer=full load=decode_16 *` faster: kernel A with this mapping fell 111.8 -> 106.5 us in the profile (cfa8c89e vs 698d4cd3); sk kernel 108.3 us -> ~103 us; 0.107%/us
- `*layer=full load=decode_32 *` faster: kernel A fell 245.9 -> 210.4 us (-14%); sk kernel 226.3 us -> ~205-215 us; 0.060%/us
- `*layer=sliding load=verify_1 *` flat: PSEUDO TOK=8 route shares issue_tile; latency-bound at B=1, small move at most; its band gate re-decides
- `*layer=sliding load=decode_32 *` flat: PSEUDO TOK=1 sk-window route shares issue_tile; _sd_gate_band re-decides
- `*load=verify_16 *` flat: FA3 paths untouched

## Risk

Expected about +1.0-1.9% over v3chain's 1.5360 (FD16 -5 us x 0.107%/us, FD32 -12..-20 us x 0.060%/us), enough to beat the 1.5347 board in one check. Weakest link: s6's DIAG reached 3.97 TB/s gather-only with the old half-line mapping, so lines per instruction may not be the sk kernel's limiter and the gain may be smaller than on kernel A; judge it on the sk kernel's profile time, not the headline. If _sk_validate fails, full decode B>=2 falls back to stage-1 kernels A+B (profile shows splitkv_attn_kernel): an indexing bug, not a result. Decode-portable: decode_layer runs the same splitkv_attn_sk_kernel at decode_16/32 (110.5/224.0 us).
