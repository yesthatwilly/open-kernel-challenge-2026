"""CPU model of the decode judge's routing (reference.py structure, our own seeds) vs the X0a surrogate basis.

Prints busiest-range (local routes, distinct, max, hot>16) per load for:
  judge  - router rows with a component along the true mean (ROUTER_ALIGN), stream on that mean
  sur(b) - the same router, stream on a surrogate mean r + b * (random mix of router rows), as _cb_basis builds it
Attention branch: an independent unit-RMS row per token (alpha/beta from a vec(H, 0.3) residual gate).
Also prints moe dist-normal routing stats at T=128/256 (reference generate_input structure).
"""
import sys

import numpy as np

H, E, K, LOC = 8192, 256, 8, 32
LOADS = {(1, 1): 2.5, (1, 8): 7.5, (16, 1): 7.5, (32, 1): 9.0, (16, 8): 18.4, (32, 8): 21.3}
S = 8
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-8)


mean_true = rng.standard_normal(H).astype(np.float32)
shared_true = rng.standard_normal((4, H)).astype(np.float32)
gate = 0.3 * rng.standard_normal(H).astype(np.float32)
w = gate / 0.3
beta1 = sig(w)
alpha1 = np.sqrt(np.clip(sig(-w) * (1 + beta1), 1e-3, None))
pre_ffn = 0.1 * rng.standard_normal(H).astype(np.float32)


def router():
    rows = 0.045 * rng.standard_normal((E, H)).astype(np.float32)
    rows += (0.07 * rng.standard_normal((E, 1)).astype(np.float32)) * mean_true[None, :] / np.sqrt(H)
    return rows, 0.055 * rng.standard_normal(E).astype(np.float32)


def stream(mean, shared, b, l):
    t = b * l
    coef = rng.standard_normal((t, 4)).astype(np.float32)
    scale = np.ones((b, l, 1), np.float32)
    if 1 < l <= 8:
        scale[:, 1:] = 0.6
    scale = scale.reshape(t, 1)
    return (scale * (0.7 * mean + np.sqrt(0.9 / 4) * coef @ shared)
            + np.sqrt(1 - 0.9 * scale * scale) * rng.standard_normal((t, H)).astype(np.float32))


def load_stats(rows, bias, mean, shared, b, l):
    out = []
    for _ in range(S):
        st = stream(mean, shared, b, l)
        a = rms(rng.standard_normal(st.shape).astype(np.float32))
        h = rms(alpha1 * st + beta1 * a) * (1 + pre_ffn)
        sc = sig(0.1767766952966369 * (h @ rows.T)) + bias
        ids = np.argpartition(-sc, K, axis=1)[:, :K].reshape(-1)
        pairs = np.bincount(ids // LOC, minlength=E // LOC)
        g = int(pairs.argmax())
        loc = ids[(ids // LOC) == g] - g * LOC
        cnt = np.bincount(loc, minlength=LOC)
        out.append((cnt.sum(), (cnt > 0).sum(), cnt.max(), (cnt > 16).sum()))
    return np.mean(np.array(out, np.float64), 0)


def report(tag, rows, bias, mean, shared):
    errs, parts = [], []
    for (b, l), tgt in LOADS.items():
        r, d, m, hot = load_stats(rows, bias, mean, shared, b, l)
        errs.append(abs(d / tgt - 1))
        parts.append(f"{b}x{l} {r:.0f}/{d:.1f}/{m:.0f}/{hot:.1f}")
    print(f"{tag:14s} worst {max(errs):5.1%}  " + "  ".join(parts), flush=True)
    return max(errs)


for ri in range(3):
    rows, bias = router()
    print(f"-- router {ri} (routes/distinct/max/hot per load; targets distinct "
          + " ".join(f"{b}x{l}:{t}" for (b, l), t in LOADS.items()) + ")")
    report("judge", rows, bias, mean_true, shared_true)
    best = None
    for draw in range(3):
        r = rng.standard_normal(H).astype(np.float32)
        u = rng.standard_normal(E).astype(np.float32) @ rows
        sh = rng.standard_normal((4, H)).astype(np.float32)
        for beta in (0.0, 0.3, 0.6, 0.8):
            m = r + beta * u / u.std()
            m = m / m.std()
            e = report(f"sur d{draw} b{beta}", rows, bias, m, sh)
            best = min(best or (9, None), (e, (draw, beta)))
    print("   best surrogate", best)

print("-- moe dist normal (reference generate_input structure): touched / max / local routes / v3 passes(16)")
for t in (128, 256):
    rows = []
    for _ in range(15):
        sc = rng.random((t, E)) + 0.03 * rng.standard_normal(E) / np.sqrt(-np.log(np.clip(rng.random(E), 1e-12, None)))
        ids = np.argpartition(-sc, K, axis=1)[:, :K].reshape(-1)
        pairs = np.bincount(ids // LOC, minlength=E // LOC)
        g = int(pairs.argmax())
        loc = ids[(ids // LOC) == g] - g * LOC
        cnt = np.bincount(loc, minlength=LOC)
        rows.append(((cnt > 0).sum(), cnt.max(), cnt.sum(), int(np.ceil(cnt / 16).sum())))
    rows.sort(key=lambda x: x[3])
    print(f"T={t}: " + "  ".join(f"{a}/{b}/{c}/{d}" for a, b, c, d in rows))
