"""Emulate _b12_encode's base choice and per-CTA escape counts for each layer (numpy, bf16 RNE)."""
import numpy as np

LAYERS = {"pooler": (4096, 8192, 32), "qkv_gate": (2304, 8192, 16), "o": (8192, 1024, 64), "unpooler": (8192, 4096, 64)}
CAP = (2048 - 256) // 8


def to_bf16_bits(f32):
    u = f32.view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint32) & 0xFFFF


for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    for name, (n, k, r) in LAYERS.items():
        w = (rng.standard_normal((n, k), dtype=np.float32) / np.float32(np.sqrt(k))).astype(np.float32)
        bits = to_bf16_bits(w)
        hi = bits >> 8
        e71 = hi & 0x7F
        hist = np.bincount(e71.ravel(), minlength=128)
        win = np.convolve(hist, np.ones(8, dtype=np.int64), mode="valid")
        base = int(win.argmax())
        esc = (e71 < base) | (e71 > base + 7)
        per = esc.reshape(n // r, r * k).sum(1)
        print(f"seed {seed} {name:9s} base {base} escapes {int(esc.sum())} ({esc.mean():.2e}) max/CTA {per.max()} "
              f"mean/CTA {per.mean():.1f} cap {CAP}")
