"""Emulate eak2::build_map (ea2_kernels.cuh) for random routings and check: positions are a stable sort of the local
routes by expert, passes cover every local route exactly once in <= 8-route pieces of one expert, n_v <= 64, and the
stream-K partition over W = n_v * NG * 8 with G CTAs gives every chunk one owner and every partial group a ticket
target equal to its number of writers."""
import numpy as np

rng = np.random.default_rng(7)


def build_map(ids, start, count, warps=5):
    R = ids.size
    le = np.where((ids - start >= 0) & (ids - start < count), ids - start, -1)
    nchunk = (R + 31) // 32
    chunk_cnt = np.zeros((8, 32), dtype=np.int64)
    rank = np.zeros(R, dtype=np.int64)
    for c in range(nchunk):
        lanes = le[c * 32:(c + 1) * 32]
        for lane, v in enumerate(lanes):
            same = [i for i in range(len(lanes)) if lanes[i] == v]
            rank[c * 32 + lane] = sum(1 for i in same if i < lane)
            if v >= 0 and rank[c * 32 + lane] == 0:
                chunk_cnt[c, v] = len(same)
    cnt = np.zeros(32, dtype=np.int64)
    for e in range(32):
        run = 0
        for c in range(nchunk):
            v = chunk_cnt[c, e]
            chunk_cnt[c, e] = run
            run += v
        cnt[e] = run
    base, passes, b = np.zeros(32, dtype=np.int64), [], 0
    for e in range(32):
        base[e] = b
        for p in range(0, cnt[e], 8):
            passes.append((e, b + p, min(cnt[e] - p, 8)))
        b += cnt[e]
    pos_r = np.full(b, -1)
    for r in range(R):
        if le[r] >= 0:
            c = r // 32
            pos_r[base[le[r]] + chunk_cnt[c, le[r]] + rank[r]] = r
    return le, passes, pos_r


def owner(q, W, G):
    return ((q + 1) * G + W - 1) // W - 1


def check_partition(nv, NG, G):
    W = nv * NG * 8
    b = [(c * W) // G for c in range(G + 1)]
    writers = {}
    for c in range(G):
        if b[c] >= b[c + 1]:
            continue
        for grp in range(b[c] // 8, (b[c + 1] - 1) // 8 + 1):
            g0 = grp * 8
            if not (g0 >= b[c] and g0 + 8 <= b[c + 1]):
                slot = 0 if grp == b[c] // 8 else 1
                writers.setdefault(grp, []).append((c, slot))
    for grp, ws in writers.items():
        g0 = grp * 8
        cf, cl = owner(g0, W, G), owner(g0 + 7, W, G)
        assert len(ws) == cl - cf + 1, (grp, ws, cf, cl)
        assert [c for c, _ in ws] == list(range(cf, cl + 1))
        for c, slot in ws:  # the finisher reads slot (grp == b_c // 8 ? 0 : 1)
            assert slot == (0 if grp == b[c] // 8 else 1)
    slots = {}
    for grp, ws in writers.items():
        for c, slot in ws:
            assert (c, slot) not in slots, ("slot reused", c, slot)
            slots[(c, slot)] = grp


worst_v = 0
for trial in range(3000):
    T = int(rng.integers(1, 33))
    start = int(rng.choice([0, 96, 224]))
    hot = rng.random() < 0.3
    ids = np.zeros((T, 8), dtype=np.int64)
    for t in range(T):
        if hot:
            ids[t] = rng.choice(np.arange(start, start + 32) % 256, size=8, replace=False)
        else:
            ids[t] = rng.choice(256, size=8, replace=False)
    ids = ids.reshape(-1)
    le, passes, pos_r = build_map(ids, start if start + 32 <= 256 else start, 32)
    loc = [r for r in range(ids.size) if le[r] >= 0]
    want = sorted(loc, key=lambda r: (le[r], r))
    assert list(pos_r) == want, "positions are not a stable sort by expert"
    covered = []
    for e, beg, n in passes:
        assert 1 <= n <= 8
        assert all(le[pos_r[beg + j]] == e for j in range(n))
        covered.extend(range(beg, beg + n))
    assert covered == list(range(len(loc)))
    worst_v = max(worst_v, len(passes))
    assert len(passes) <= 64
for nv in range(1, 65):
    for NG in (128, 64):
        for G in (264, 132):
            check_partition(nv, NG, G)
print(f"build_map ok over 3000 routings (max passes {worst_v}); stream-K partition ok for n_v 1..64, NG 128/64, G 264/132")
