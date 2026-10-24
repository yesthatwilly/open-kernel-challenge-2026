---
problem: muse_attention_decode
slug: fused-sliding-normgate-epilogue
created: 2026-10-23T223803Z
base: master
area: output norm-gate
change: >
  Stacked on fused-sliding-preattn-into-window-kernel: on sliding layers, fold rms_norm_gate into
  the fused window-attention kernel's epilogue so the whole sliding path is ONE launch. After the
  online softmax finishes for query token j and its 8 query heads, the program has acc[8, 64]/l in
  fp32 per head; compute per head r = rsqrt(mean(acc^2) + 1e-5), load the head's 64 gate logits
  from qkv row b*L+j at offset (Hq+2*Hkv)*64 + h*64 (bf16 -> fp32), and store bf16 (acc * r *
  sigmoid(gate)) directly into the final [B, L, Hq*64] output tensor at [b*L+j, h*64:(h+1)*64].
  kernel_fn on sliding layers returns that tensor without calling rms_norm_gate; full layers keep
  SGLang's rms_norm_gate. Semantics match reference: _rms over the 64-wide head of the fp32
  attention output, times sigmoid(gate.float()), cast to bf16 at the end (SGLang's rms_norm_gate
  takes a bf16 attention output; this epilogue skips that intermediate bf16 rounding, which is
  finer, and allowed since bf16 is kept wherever SGLang keeps it: the returned tensor).
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "2 -> 1 launches on a launch-bound shape: the last fixed ~2-3 us"
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: fused-sliding-preattn-into-window-kernel
priority: 5
kind: primary
controls_for: null
review: required
status: hold
hold_reason: "moot under master 1.269x (sweep by the attention minter): stacks on the held sliding Triton chain; same measured outcome"
---

# fused-sliding-normgate-epilogue

## Change

Stacked on fused-sliding-preattn-into-window-kernel: on sliding layers, fold rms_norm_gate into the fused window-attention kernel's epilogue so the whole sliding path is ONE launch. After the online softmax finishes for query token j and its 8 query heads, the program has acc[8, 64]/l in fp32 per head; compute per head r = rsqrt(mean(acc^2) + 1e-5), load the head's 64 gate logits from qkv row b*L+j at offset (Hq+2*Hkv)*64 + h*64 (bf16 -> fp32), and store bf16 (acc * r * sigmoid(gate)) directly into the final [B, L, Hq*64] output tensor at [b*L+j, h*64:(h+1)*64]. kernel_fn on sliding layers returns that tensor without calling rms_norm_gate; full layers keep SGLang's rms_norm_gate. Semantics match reference: _rms over the 64-wide head of the fp32 attention output, times sigmoid(gate.float()), cast to bf16 at the end (SGLang's rms_norm_gate takes a bf16 attention output; this epilogue skips that intermediate bf16 rounding, which is finer, and allowed since bf16 is kept wherever SGLang keeps it: the returned tensor).

## Expected

- `*layer=sliding load=decode_1*` faster: 2 -> 1 launches on a launch-bound shape: the last fixed ~2-3 us
- `*layer=sliding load=verify_1*` faster: same
- `*layer=sliding load=decode_16*` faster: same
- `*layer=sliding load=decode_32*` faster: same
- `*layer=sliding load=verify_*` faster: same
- `*layer=full*` flat: untouched; control

## Risk

Medium risk: the epilogue is 64-wide per head and adds a gate load per (token, head); correctness is a per-row rmsnorm, checked by the row bound (5x) as well as the global one. The one hazard is the near-zero-row rule: rows whose attention output is near zero are measured against a tenth of the average row, so eps placement (inside the sqrt, 1e-5, over the mean of 64 squares) must match reference._rms exactly. If the window kernel of the previous step ended up grid (B, Hkv, L) with redundant key recompute, the epilogue is per program anyway and nothing changes.
