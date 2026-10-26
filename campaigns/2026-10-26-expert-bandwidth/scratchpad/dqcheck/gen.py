# Builds a host C++ harness from the REAL source text (ea_common.cuh, ea_ladder.cuh) and the REAL _ea_luts (ea_core.py).
import ast, re, sys, types
import numpy as np
R = "/home/user/git/musespark/.worktrees/expert-bandwidth/rewrite/"
common = open(R + "lib/cuda/ea_common.cuh").read()
ladder = open(R + "lib/cuda/ea_ladder.cuh").read()
core = open(R + "lib/ea_core.py").read()

def cut(text, start_pat, end_pat):
    a = text.index(start_pat); b = text.index(end_pat, a)
    return text[a:b]

ext_master = cut(common, "template <int I>\n__device__ __forceinline__ uint32_t ext_master", "// Four E4M3")
ext_prop = cut(common, "__device__ __forceinline__ uint32_t rotl32", "// Byte j (0..3)")
place = cut(ladder, "__device__ __forceinline__ uint32_t place_prop", "__global__ void dq_exact_kernel")

# Real _ea_luts with a numpy stand-in for torch
tree = ast.parse(core)
src = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EA_PROP_ROT" for t in node.targets):
        src.append(ast.get_source_segment(core, node))
    if isinstance(node, ast.FunctionDef) and node.name == "_ea_luts":
        src.append(ast.get_source_segment(core, node))
class Arr(np.ndarray):
    def to(self, dev): return self
torch = types.SimpleNamespace(int64=np.int64, zeros=lambda shape, dtype=None: np.zeros(shape, dtype=np.int64).view(Arr))
ns = {"torch": torch}
exec("\n".join(src), ns)
lut = np.asarray(ns["_ea_luts"]("cpu"))
assert lut.shape == (3, 4, 2, 16)
with open(sys.argv[1], "w") as f:
    f.write('#include <cstdint>\n#include <cstdio>\n#define __device__\n#define __forceinline__ inline\n'
            'static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {\n'
            '  const uint64_t v = (static_cast<uint64_t>(hi) << 32) | lo;  // CUDA Math API: hi:lo << (shift & 31), high 32 bits\n'
            '  return static_cast<unsigned>((v << (shift & 31)) >> 32);\n}\n')
    f.write("namespace ea {\n" + ext_master + ext_prop + "}\nusing namespace ea;\n" + place)
    f.write("static const int64_t LUT[3][4][2][16] = {")
    f.write(",".join("{" + ",".join("{" + ",".join("{" + ",".join(str(int(lut[d, p, h, c])) for c in range(16)) + "}"
                                                     for h in range(2)) + "}" for p in range(4)) + "}" for d in range(3)))
    f.write("};\n")
print("extracted:", len(ext_master), len(ext_prop), len(place), "LUT ok")
