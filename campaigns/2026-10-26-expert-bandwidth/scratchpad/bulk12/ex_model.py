"""Epilogue EX-read wavefronts: current exs[e*NT + tok] vs tok-major exs[tok*225 + e], escapes ~ Poisson per row."""
import numpy as np

rng = np.random.default_rng(0)
CASES = {"pooler m=32 (R32)": (32, 32, 58), "pooler m=16 (R32)": (32, 16, 58), "unpooler m=32 (R64)": (64, 32, 43),
         "unpooler m=16 (R64)": (64, 16, 43), "o m=32 (R64)": (64, 32, 16)}


def wav(addrs):
    banks = {}
    for a in addrs:
        banks.setdefault(a % 32, set()).add(a)
    return max((len(v) for v in banks.values()), default=0)


for name, (R, m, n_esc) in CASES.items():
    NT = m
    rows = np.sort(rng.integers(0, R, n_esc))
    cnt = np.bincount(rows, minlength=R)
    roff = np.concatenate([[0], np.cumsum(cnt)])
    cur = new = 0
    for w0 in range(0, R * m, 32):
        idx = list(range(w0, min(w0 + 32, R * m)))
        it = 0
        while True:
            act = [(i // R, i % R) for i in idx if roff[i % R] + it < roff[i % R + 1]]
            if not act:
                break
            cur += wav([(roff[r] + it) * NT + t for t, r in act])
            new += wav([t * 225 + roff[r] + it for t, r in act])
            it += 1
    print(f"{name}: escapes/CTA {n_esc}: EX-read wavefronts current {cur} (~{cur / 1.9e3:.2f} us), tok-major/225 {new}")
