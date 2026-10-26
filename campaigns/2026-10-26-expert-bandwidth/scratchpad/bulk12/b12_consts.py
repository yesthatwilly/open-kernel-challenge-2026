# layer -> (N, K, rows per CTA R): one encoding per layer, shared by its cells.
B12_LAYERS = {
    "pooler":   (4096, 8192, 32),
    "qkv_gate": (2304, 8192, 16),
    "o":        (8192, 1024, 64),
    "unpooler": (8192, 4096, 64),   # the full weight (m > 8); the m <= 8 slice stays on its own cells
}
# cells: (source tag, layer, m, ((instance id, ring stages S, min CTAs/SM, consumer warps CW), ...), state flag).
# m == 1: b12_kernel (x resident); m in {8, 16, 32}: b12x_kernel (x streamed with each chunk, NT == m). Routed at
# cfg m equal to the cell's m only. qkv_gate stays off m >= 16 (its 2 CTAs/SM cannot hold the larger x slots).
B12_CELLS = (
    ("A12", "pooler", 1, ((0, 8, 1, 8), (1, 8, 1, 4)), "b12_pooler_ok"),
    ("A2_12", "qkv_gate", 1, ((2, 6, 2, 4), (3, 6, 2, 8)), "b12_qkv_ok"),
    ("A3_12", "o", 1, ((4, 8, 1, 8), (5, 8, 1, 4)), "b12_o_ok"),
    ("A12x8", "pooler", 8, ((6, 8, 1, 8), (7, 8, 1, 4)), "b12_pooler_m8_ok"),
    ("A2_12x8", "qkv_gate", 8, ((8, 4, 2, 4), (9, 4, 2, 8)), "b12_qkv_m8_ok"),
    ("A3_12x8", "o", 8, ((10, 8, 1, 8), (11, 8, 1, 4)), "b12_o_m8_ok"),
    ("A12x16", "pooler", 16, ((12, 8, 1, 8),), "b12_pooler_m16_ok"),
    ("A12x32", "pooler", 32, ((13, 6, 1, 8),), "b12_pooler_m32_ok"),
    ("U12x16", "unpooler", 16, ((14, 8, 1, 8),), "b12_unpooler_m16_ok"),
    ("U12x32", "unpooler", 32, ((15, 8, 1, 8),), "b12_unpooler_m32_ok"),
    ("A3_12x16", "o", 16, ((16, 8, 1, 8),), "b12_o_m16_ok"),
    ("A3_12x32", "o", 32, ((17, 8, 1, 8),), "b12_o_m32_ok"),
)
B12_BY_KEY = {(cell[1], cell[2]): cell for cell in B12_CELLS}
B12_FLAGS = tuple(cell[4] for cell in B12_CELLS)
