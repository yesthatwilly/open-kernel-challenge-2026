# okc: problem=muse_bf16_proj
"""Parity with SGLang's dispatch (hypothesis sglang-gemm-dispatch-parity).

The four bf16 layers go through the same cuBLAS dispatch SGLang uses on H200
(`_bf16_gemm_dispatch_impl`). The fp32 router runs F.linear in fp32 up to 8 tokens and, above
that, two bf16 GEMMs on a hi + lo split of the weight with fp32 accumulation, as the rules allow
("the router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation"). The split is
built once in prepare(); nothing derived from the inputs is kept between calls.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# Above this many tokens the router runs as the bf16 hi + lo pair instead of fp32 F.linear.
ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights: Weights) -> dict:
    """Called once, untimed. Keep the bf16 weights as they are (contiguous) and add the router's
    lossless-for-our-purpose hi/lo bf16 split, transposed for torch.mm."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer != "router":
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
