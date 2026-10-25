---
problem: muse_attention_decode
slug: fused-sliding-preattn-into-window-kernel
created: 2026-10-23T223802Z
base: master
area: attention boundary
change: >
  Stacked on triton-window-attention-sliding (and on fused-preattention-triton's norm/RoPE code,
  which it absorbs): on sliding layers, move the pre-attention work into the window-attention
  kernel so the sliding path is two launches (attention, rms_norm_gate) instead of three. Regrid
  the window kernel to (B, Hkv): one program per (request, kv head) handles all L query tokens of
  the block and its 8 query heads. Prologue: load the L new key rows and value rows of head kvh
  from qkv (offsets (Hq+kvh)*64 and (Hq+Hkv+kvh)*64), RMS-normalize and RoPE the keys with pos =
  seq_lens[b]+j, divide by k_scale/v_scale, store them FP8 into k_cache/v_cache at
  cache_loc[b*L+j] (the cache write the judge checks), and keep the fp32 copies in registers; load
  the 8 x L query heads from qkv (offset h*64, h in the group), RMS-normalize and RoPE them (no
  FP8 round trip: bf16-exact fp32 queries, finer than FA3's E4M3 and allowed). Main loop:
  positions [max(0, p_first - window + 1), seq_lens[b] - 1] from the cache as before (the old
  entries), masked per query token j by t <= p_j and t >= p_j - window + 1; then the L in-register
  new keys with the causal-in-block mask t <= p_j. Epilogue: bf16 [B*L, Hq, 64] output as before.
  The sliding-layer code no longer launches the separate pre-attention kernel; full layers keep
  the separate pre-attention kernel + FA3 from the landed base.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: one launch fewer on a ~21 us launch-bound shape (3 -> 2 launches after the window kernel lands)
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same; the 8 new keys are attended from registers
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: one launch fewer; 32 programs
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: same, 64 programs
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same at q_len 8
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: high
depends_on: triton-window-attention-sliding
priority: 4
kind: primary
controls_for: null
review: required
status: done
hold_reason: "moot under master 1.269x (sweep by the attention minter): stacks on triton-window-attention-sliding (held); same measured outcome"
outcome: expired
verdict_detail: "hold expired after 33 h: moot under master 1.269x (sweep by the attention minter): stacks on triton-window-attention-sliding (held); same measure"
expired_at: 2026-10-25T211327Z
---

# fused-sliding-preattn-into-window-kernel

## Change

Stacked on triton-window-attention-sliding (and on fused-preattention-triton's norm/RoPE code, which it absorbs): on sliding layers, move the pre-attention work into the window-attention kernel so the sliding path is two launches (attention, rms_norm_gate) instead of three. Regrid the window kernel to (B, Hkv): one program per (request, kv head) handles all L query tokens of the block and its 8 query heads. Prologue: load the L new key rows and value rows of head kvh from qkv (offsets (Hq+kvh)*64 and (Hq+Hkv+kvh)*64), RMS-normalize and RoPE the keys with pos = seq_lens[b]+j, divide by k_scale/v_scale, store them FP8 into k_cache/v_cache at cache_loc[b*L+j] (the cache write the judge checks), and keep the fp32 copies in registers; load the 8 x L query heads from qkv (offset h*64, h in the group), RMS-normalize and RoPE them (no FP8 round trip: bf16-exact fp32 queries, finer than FA3's E4M3 and allowed). Main loop: positions [max(0, p_first - window + 1), seq_lens[b] - 1] from the cache as before (the old entries), masked per query token j by t <= p_j and t >= p_j - window + 1; then the L in-register new keys with the causal-in-block mask t <= p_j. Epilogue: bf16 [B*L, Hq, 64] output as before. The sliding-layer code no longer launches the separate pre-attention kernel; full layers keep the separate pre-attention kernel + FA3 from the landed base.

## Expected

- `*layer=sliding load=decode_1*` faster: one launch fewer on a ~21 us launch-bound shape (3 -> 2 launches after the window kernel lands)
- `*layer=sliding load=verify_1*` faster: same; the 8 new keys are attended from registers
- `*layer=sliding load=decode_16*` faster: one launch fewer; 32 programs
- `*layer=sliding load=decode_32*` faster: same, 64 programs
- `*layer=sliding load=verify_*` faster: same at q_len 8
- `*layer=full*` flat: untouched; control

## Risk

High risk (a fused kernel), but the mechanism is only launch count, so a flat result means the window kernel's own time, not launches, now dominates the sliding shapes. Hazards: the cache write and the in-register new keys must agree bit-for-bit with what the judge reads back (write the FP8-rounded value and attend the dequantized FP8 value, not the fp32 one, so the output is checked against the entries actually stored: the judge re-attends the reference with YOUR new entries, so using the un-rounded key would differ by the FP8 rounding, inside tolerance but needless). Grid (B, Hkv) gives only 2 programs at batch 1: 2048 x 2 rows x 128 B = 512 KB streamed by two SMs; if that is slower than the previous grid (B, Hkv, L) at verify_1, split the window across programs with an in-kernel atomic-free two-pass (second-stage combine) or keep grid (B, Hkv, L) and let each program redundantly recompute the L new keys (cheap: 8 rows x 64) while only program j == 0 stores them. Depends on both triton-window-attention-sliding and fused-preattention-triton having landed; the single depends_on names the window kernel because its grid and masks are what this edits.
