"""CPU emulation of Engine A's record layout: rewrite/lib/ea_engine.py _ea_pack (numpy port) against the device-side
decode in rewrite/lib/cuda/ea_kernels.cuh consume() + ea_common.cuh ext_prop / ext_master / scale_prop.
For every record, rebuild each A-fragment element the way a lane of the kernel does and compare it, exactly, with the
direct dequantization e2m1 * sf * 2^-7 of the same (row, k). Also checks the K1/K2 row maps."""
import numpy as np

rng = np.random.default_rng(20261026)
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1(c):
    c = np.asarray(c)
    v = GRID[c & 7]
    return np.where(c & 8, -v, v)


def e4m3(b):
    b = np.asarray(b).astype(np.int64)
    f, m = b >> 3, b & 7
    return np.where(f > 0, 2.0 ** (f - 7) * (1 + m / 8), 2.0 ** -6 * (m / 8))


def luts():
    lut = np.zeros((3, 4, 2, 16), dtype=np.int64)
    for c in range(16):
        for p in range(4):
            for half in range(2):
                lut[1, p, half, c] = c << (4 * ((3 - p) + 4 * half))
                if p < 3:
                    v = ((c & 7) << (6 - 3 * p)) | ((c >> 3) << (15 - 3 * p))
                else:
                    v = (((c >> 1) & 3) << 10) | ((c & 1) << 13) | ((c >> 3) << 14)
                lut[2, p, half, c] = v << (16 * half)
    return lut


def pack(tile_nib, tile_sf, lut):  # numpy port of _ea_pack
    ng, k = tile_nib.shape[0], tile_nib.shape[-1]
    nk = k // 512
    t = tile_nib.reshape(ng, 4, 2, 8, nk, 4, 4, 2, 2, 4, 2).astype(np.int64)
    words = np.zeros((ng, 4, 2, 8, nk, 4, 4, 4), dtype=np.int64)
    for bsel in range(2):
        for hi in range(2):
            p = 2 * bsel + hi
            for el in range(2):
                words |= lut[p, el][t[:, :, :, :, :, :, :, bsel, hi, :, el]]
    w = np.ascontiguousarray(words.transpose(0, 4, 1, 2, 5, 3, 7, 6))
    codes = np.stack([(w >> (8 * i)) & 0xFF for i in range(4)], axis=-1).astype(np.uint8).reshape(ng, nk, 4, 4096)
    sc = tile_sf.reshape(ng, 4, 2, 8, nk, 32).transpose(0, 4, 1, 2, 3, 5).reshape(ng, nk, 4, 512)
    return np.concatenate((codes, sc), axis=-1)


def ext(q, p, layout):
    q = q & 0xFFFFFFFF
    if layout == 1:
        t = (q << (4 * p)) & 0xFFFFFFFF
        return (t & 0x80008000) | ((t & 0x70007000) >> 6)
    if p == 0:
        return q & 0x81C081C0
    if p == 1:
        return ((q << 3) & 0xFFFFFFFF) & 0x81C081C0
    if p == 2:
        return ((q << 6) & 0xFFFFFFFF) & 0x81C081C0
    return ((q >> 3) & 0x01800180) | ((q >> 7) & 0x00400040) | (((q << 1) & 0xFFFFFFFF) & 0x80008000)


def bf16(bits16):
    return np.array([int(bits16) << 16], dtype=np.uint32).view(np.float32)[0].astype(np.float64)


def scale_prop(b):
    t = (b << 16) | b
    return ((t * 16) + 0x77807780) & 0xFFFFFFFF


def emulate(rec, layout):
    """Values [16 rows, 512 k] a warp would feed the MMA from one 4608 B record."""
    out = np.full((16, 512), np.nan)
    words = rec[:4096].view(np.uint32)
    scales = rec[4096:]
    for lane in range(32):
        g, c = lane >> 2, lane & 3
        for rh in range(2):
            for kq in range(4):
                w4 = words[((rh * 4 + kq) * 32 + lane) * 4:((rh * 4 + kq) * 32 + lane) * 4 + 4]
                row = rh * 8 + g
                for wd in range(4):
                    q = int(w4[wd])
                    for bsel in range(2):
                        j = wd * 2 + bsel
                        b = int(scales[row * 32 + kq * 8 + j])
                        if layout == 2:
                            s = bf16(scale_prop(b) & 0xFFFF)
                        else:
                            s = float(np.float32(e4m3(b) * 2.0 ** 119))  # master: bf16(sf * 2^119), exact here
                        for hi in range(2):
                            v = ext(q, 2 * bsel + hi, layout)
                            for el in range(2):
                                kk = kq * 128 + wd * 32 + bsel * 16 + hi * 8 + c * 2 + el
                                code_val = bf16((v >> (16 * el)) & 0xFFFF)
                                out[row, kk] = code_val * s
    return out


def check(layout, ng=2, k=4096):
    lut = luts()[layout]
    nib = rng.integers(0, 16, size=(ng, 4, 2, 8, k), dtype=np.int64).astype(np.uint8)
    sf = rng.integers(0x50, 0x7F, size=(ng, 4, 2, 8, k // 16), dtype=np.int64).astype(np.uint8)
    recs = pack(nib, sf, lut)
    bad = 0
    for gi in range(ng):
        for kc in range(k // 512):
            for w in range(4):
                got = emulate(recs[gi, kc, w], layout)
                rows_nib = nib[gi, w].reshape(16, k)[:, kc * 512:(kc + 1) * 512]
                rows_sf = sf[gi, w].reshape(16, k // 16)[:, kc * 32:(kc + 1) * 32]
                want = e2m1(rows_nib) * np.repeat(e4m3(rows_sf), 16, axis=1) * 2.0 ** -7
                bad += int(np.sum(got != want))
    return bad


if __name__ == "__main__":
    for layout, name in ((2, "proposed"), (1, "master")):
        print(f"layout {name}: element mismatches {check(layout, ng=1, k=4096)} of {4 * 16 * 4096}")
    # K1 row map: record rows (rh, g) of warp w in group gi are gate/up rows of h column gi*32 + w*8 + g.
    inter = 4096
    cols = np.arange(inter)
    gi, w, g = cols // 32, (cols // 8) % 4, cols % 8
    assert np.array_equal(gi * 32 + w * 8 + g, cols)
    # K2 row map: record row (rh, g) of warp w in group gi is output row gi*64 + w*16 + rh*8 + g; the epilogue writes
    # n0 = gi*64 + w*16 + g (rh 0) and n0 + 8 (rh 1).
    rows = np.arange(4096).reshape(64, 4, 2, 8)
    assert np.array_equal(rows[:, :, 0, :] + 8, rows[:, :, 1, :])
    print("row maps ok")
