#!/usr/bin/env python3
"""Run one decode submission on the 12 judge cases, one profiled call per case, for Nsight Compute.

  ncu --profile-from-start off --graph-profiling node --metrics smsp__inst_executed.sum \
      --csv --page raw --log-file <out>.csv  python profile_one.py <problem_dir> <submission.py> <cases.json>

prepare() runs outside the profiled range, so its timing races see an unperturbed GPU. Each case is
generated as the judge does (router_seed from weights_fields, local_start from case_fields), warmed
up twice, then called once between cudaProfilerStart/Stop inside an NVTX range "case<i>", so the
ncu rows can be split per case.
"""
import importlib.util, inspect, json, sys, time
import torch

prob, sub, cases_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, prob)
import reference  # noqa: E402

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

raw = json.load(open(cases_path))
cases = [{k: v for k, v in c.items() if k not in ("weight", "repeats", "load", "router_load", "router_distinct")} for c in raw]
WSEED = 5353
W = reference.generate_weights(seed=WSEED)
if hasattr(reference, "weights_fields"):
    wf = reference.weights_fields(W, raw, WSEED)
    if wf.get("router_seed") is not None:
        W = reference.generate_weights(seed=WSEED, router_seed=wf["router_seed"])
mod = load(sub, "cand")
stateful = len(inspect.signature(mod.kernel_fn).parameters) == 2
torch.cuda.synchronize(); t0 = time.time()
state = mod.prepare(W) if stateful else None
torch.cuda.synchronize(); print(f"prepare {time.time()-t0:.1f}s", flush=True)
call = (lambda d: mod.kernel_fn(d, state)) if stateful else (lambda d: mod.kernel_fn(d))

for i, c in enumerate(cases):
    cc = dict(c)
    if hasattr(reference, "case_fields"):
        cc.update(reference.case_fields(cc, W))
    d = reference.generate_input(**cc)
    for _ in range(2):
        call(d)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"case{i}")
    call(d)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"case{i} done", flush=True)
print("ok", flush=True)
