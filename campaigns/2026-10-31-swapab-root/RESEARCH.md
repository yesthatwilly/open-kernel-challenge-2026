# Decode swap-AB copy-check fault (cb182920): root cause and fix

Branch `fix/swapab-root`, file `muse_decode_layer/submission.py`. Base: master `d6a0d421`, decode blob `b5f84e27`
(sha256 `96998c20eb90`). Fixed file: sha256 `93bcc687d38f`, 997,498 bytes (limit 999,000). Swap-AB stays ON
(`CAMPAIGN_DENSE_SWAPAB = True`) and the copy check stays ON. `kernel_fn` is unchanged.

All GPU work was on H100 gpunode06 GPUs 5 and 6, through `~/git/takemyjob/prototypes/hopper-probe` (bundles `sr-fx` to
`sr-fx4` and `sr-fx-nj`, package `probes/sr-fx/`, nodejudge config `probes/nodejudge/configs/sr-fx-val.json`). Run
ids below are `runs/<id>-gpunode06/` there.

## Verdict

**Root cause (shown on the H100).** Prepare's graph checks replay CUDA graphs whose cuBLAS calls write into a cuBLAS
workspace that has already been freed and unmapped. The faulting kernel is cuBLAS's own
`dot_kernel<float,...cublasGemvTensorStridedBatched...>`, the FP32 router GEMM (`_dc0_lin` -> `F.linear`). It is not
the swap-AB kernel, a ticket, or the deep copy.

The use-after-free comes from torch 2.14.1's cuBLAS workspace handling interacting with how the submission captures
and frees graphs:

1. **One workspace per stream.** torch caches one cuBLAS workspace (32 MiB) per (handle, stream). Every
   `torch.cuda.graph` capture runs on one shared stream, `torch.cuda.graph.default_capture_stream`. So a capture
   reuses whatever workspace is cached for that stream. That workspace may have been allocated in an earlier graph's
   private pool. It may also be in the default pool: an eager warm-up allocates it there when its
   `torch.cuda.Stream()` happens to draw the capture stream, which is one of the 32 pooled streams.
2. **Destroying any graph frees it.** `CUDAGraph::reset()`, run when any graph is destroyed, calls
   `at::cuda::clearCublasWorkspacesForStream(capture_stream)`. I confirmed this by disassembling the node's
   `libtorch_cuda.so`. The workspace is freed even though graphs captured later still have its address baked into
   their kernels.
3. **The next capture unmaps it.** The next capture's `torch.cuda.graph.__enter__` calls `empty_cache()`, which
   releases the freed 32 MiB segment with `cudaFree`.
4. **The next replay faults.** When a graph that captured that workspace replays, its cuBLAS split-K and GEMV kernels
   write partial sums into unmapped memory. The result is a Warp MMU Fault and a sticky `cudaErrorIllegalAddress`.
   If the driver has already handed the address to a new allocation, the replay instead silently overwrites another
   tensor.

**Why `_dsa_copy_check` is where this shows up.** Unlike Engine A's `_ead_copy_one`, which is one function call per
T, the copy check runs every T inside one loop body. The previous T's state graph stays referenced by the loop
variable `graph` until just after the next T's state capture. The sequence is:

- T=1 state graph (T1A) is captured. The cached workspace entry is empty, so the workspace is allocated inside T1A's
  private pool.
- T=8 state graph (T8A) is captured and reuses that workspace.
- Rebinding `graph` destroys T1A. Its reset clears the entry, and the workspace's pool becomes freeable.
- T=8 copy graph (T8B) is captured. Its `empty_cache()` releases the segment.
- T8A replays and writes into the unmapped segment.

`_dl3_alternation_check` has the same inlined-loop pattern (that is where the deterministic H100 fault below
occurs), and so do `_hc_copy_check` and the other copy checks that were copied from it. All of them capture through
`_rt_graph` or `_sk_graph_of`, so the fix covers them.

T8A touches the workspace only during prepare. The FP32 router at T=1 and T=8 still runs through cuBLAS `F.linear`
until `_dc0f_gate`, at the very end of prepare, moves it to a Triton kernel. Whether the replay faults or silently
writes into reused memory depends on what the allocator and driver did in between. That is why the fault is
intermittent, and why loops run after prepare could never reproduce it.

**Fix.** A helper `_rt_cap(g)` now wraps all four capture sites in the submission. Before each capture it destroys a
one-kernel dummy graph on the default capture stream. That makes torch clear the stream's workspace entry, so the real
capture allocates its own workspace inside its own private pool. No graph can then reference a workspace it does not
own, and destroying other graphs can no longer unmap memory a live graph writes.

## Evidence

### 1. A deterministic fault on master's bytes, in the same class (dl3 two-state check)

- **Seed.** The fault reproduces in every run with judge-like E2b inputs and seed `2693396613634879665`, found by the
  sr-ml agent: 9 of 9 processes.
  - sr-ml runs: `20261030-184411` (5/5) and `20261030-190336` p10 (1/1).
  - My runs: `20261030-215242`, `20261030-220017` and `20261030-221336` (1/1 each).
  - The fault is in `_dl3_alternation_check`, in the bisect pass with piece A only, case `full kv_scale=1`, replay 0
    on state 0.
- **GPU coredump** (`20261030-220017`, read with cuda-gdb 13.3; cuda-gdb 13.0 aborts on device exception 30):
  - Exception: `CUDA Exception: Warp MMU Fault` in `dot_kernel<float, 128, 0,
    cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>>>`, grid
    (8,1,256).
  - Faulting block (4,0,16), thread 0. It stores to `0x155006000210`. That address is workspace base `0x155006000000`
    (R8:R9) plus 4 x 132, the partial-sum slot of block x=4 in batch z=16.
- **Allocator trace** (`20261030-221336`, `torch.cuda.memory._record_memory_history`, Python stacks):
  - The 32 MiB segment `0x155006000000` is allocated on the capture stream by the eager warm-up:
    `_dc0_lin:7087 < _dl_forward < kernel_fn < fn < _sk_graph_of:3143`.
  - It is freed at `_dl3_alternation_check:4172`, the `graph = _sk_graph_of(...)` rebinding that destroyed the
    previous case's leftover graph.
  - It gets `segment_free` (cudaFree) from `empty_cache < torch.cuda.graph.__enter__ < _sk_graph_of:3147`, the
    state-1 capture.
  - Then the state-0 replay faults, at the same address as the coredump.
- **Capture and lifetime trace** (`20261030-221336`, hook `alias`, which does not change behaviour).
  - The faulting graph `g32` was captured right after a warm-up whose pool draw #31 was the default capture stream
    (`ALIASED`). The previous leftover graph `g31` was destroyed immediately after `g32`'s capture, `g33` was
    captured, and `g32`'s first replay faulted.
  - In the same process's first pass (pieces A+B+C), the matching capture `g28` was not aliased. It did not fault.
- **Fixed file on the same seed** (`20261030-222423`): 2 of 2 processes ran clean. The same capture is still
  `ALIASED` (the fix does not change pool draws). The replay now completes and gets the benign numeric FAIL described
  in Remaining risk 4.

### 2. Stale-workspace detector: no luck involved

`sr_fx.py` mode `wsuse` and hook `dsatime` + `SRFX_DSATIME_WS=1` do the following per T:

1. Put the capture stream's cuBLAS workspaces in the default pool, as an aliased warm-up does.
2. Capture `kernel_fn`, either with a raw `torch.cuda.graph` or with the submission's `_rt_graph`.
3. Drop the workspace map, which is what any graph's reset does.
4. Re-occupy the freed blocks with 0x5A sentinel tensors of the same size on the same stream.
5. Replay, and count the sentinel bytes the replay overwrote.

| where | T | raw capture | master `_rt_graph` | fixed `_rt_graph` |
|---|---|---|---|---|
| prepare time, inside `_dsa_setup` (`20261030-222958`) | 1 | 8,162 B | 8,162 B | 0 |
| | 2 | 130,697 B | 130,697 B | 0 |
| | 8 | 1,080,589 B | 1,080,589 B | 0 |
| | 16, 32 | 0 | 0 | 0 |
| after prepare, DSA on (`20261030-222424`) | 2 / 128 | 130,691 / 2,352,522 B | same | 0 |
| after prepare, DSA off | 1, 2, 8, 9, 16, 32, 128 | 18 KB to 2.35 MB each | same | 0 |

So at prepare time, the graphs of the DSA copy check at T=1 and T=8 write into the shared workspace. With swap-AB
off, the cuBLAS qkv_gate and o GEMMs write into it at every T. In other words, the hazard is not specific to swap-AB;
swap-AB actually removes most of it.

### 3. The DSA copy check itself faults at prepare time with the judge's exact signature

Hook `dsatime` runs the submission's own `_dsa_copy_check` 64 extra times, from inside `_dsa_setup`. That is the
real prepare-time state, before dc0f, b12p and the other later stages. Each run is preceded by one extra pool draw.

- **Master** (`20261030-222958` p1, seed 777002): 6 copy checks passed, and the 7th raised. The log reads
  `[dense-swapab] AcceleratorError: CUDA error: an illegal memory access was encountered`, then `ticket reset raised`
  and `final sync raised`. After that, hotcold, b12p, skfull, c320d, w6b and w6p each catch the sticky error, and
  prepare dies. This is line for line the tail of judge log `submission.cb182920110e4cd0.log`.
  - The faulting check followed exactly the T1A -> T8A -> destroy T1A -> T8B -> replay T8A sequence above, with no
    aliasing at all: captures `g484`, `g486`, destroy `g484`, `g487`, fault.
- **Fixed** (`20261030-222958` p2, same seed): 64 extra plus 1 real copy check, all PASS, rc 0.
- **Deterministic.** The configuration is seed 777002, `--fixed-weights`, hooks `alias,dsatime`,
  `SRFX_DSATIME_WS=1`. In it, master faulted at the 7th copy check in 4 of 4 processes: `20261030-222958` p1,
  `20261030-224121` p1 and p3, and `20261030-224835` p1 (the last one with the allocator trace below). The fixed file
  passed all 65 checks in 3 of 3 processes: `20261030-222958` p2 and `20261030-224121` p2 and p4.
- **Allocator trace of the DSA fault** (`20261030-224835`, master, `memhist`), for the faulting check:
  1. The 32 MiB segment `0x154e12000000` is allocated during T1A's capture, on the capture stream, by
     `_dc0_lin:7087 < _dl_forward < kernel_fn < <lambda>:14067 < _rt_graph:8316`. That is the capture line: the T=1
     FP32 router `F.linear`.
  2. It is freed at `_dsa_copy_check:14067`, the `graph, out = _rt_graph(...)` rebinding of the T=8 state capture,
     which destroys T1A.
  3. It gets `segment_free` (cudaFree) from `empty_cache < torch.cuda.graph.__enter__ < _rt_graph:8315`, the T8B
     capture.
  4. T8B allocates its own workspace at a different address, `0x154e07600000`, so `0x154e12000000` stays unmapped.
  5. The T=8 replays fault.
- **A/B over other seeds** (`20261030-223407`, plain hook `dsatime`, 64 extra checks plus the real one):

  | seed | master | fixed |
  |---|---|---|
  | 777003 | 65 of 65 PASS | 65 of 65 PASS |
  | 777004 | fault at the 14th check | 65 of 65 PASS |
  | 777005 | fault at the 23rd check | 65 of 65 PASS |
  | 777006 | 65 of 65 PASS | 65 of 65 PASS |

- **With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (`20261030-223837`, seeds 777011 and 777012): master
  and the fix both passed 65 of 65. Expandable segments change whether a freed address is unmapped or remapped by the
  time of the replay, which fits the explanation that the fault is the unmapped case of a use-after-free that happens
  in every check.
- **Totals for the prepare-time copy check:**
  - master faulted in 6 of 10 processes (6 faults in 325 checks);
  - the fix faulted in 0 of 9 processes (0 faults in 585 checks).

### 4. Why the earlier H100 attempts saw 0 faults

- **Cold prepares (32).** Each runs this copy check once. Whether the freed segment's address is still unmapped at
  T8A's replay depends on allocation history, and in a single check it usually is not. Here, master passed 65 of 65
  checks for seed 777003 and faulted on the 7th for seed 777002.
- **Post-prepare loops (about 2,650 iterations).** By then `_dc0f_gate` has moved the T=1 and T=8 FP32 router to the
  Triton `_dc0_r32_kernel`, so those graphs no longer touch the workspace at all. The detector confirms this: T=1 and
  T=8 are clean after prepare and dirty at prepare time.
- **Phase sweep after prepare** (`20261030-221338`, 64 copy checks covering all 32 pool phases): 0 faults, for the
  same reason.
- **Engine A's copy check** has no exposure, because each T runs in its own `_ead_copy_one` call, so both graphs die
  before the next T's captures.

## Fix

The diff against `b5f84e27`:

- **New `_rt_cap(g)`**, next to `_rt_graph`, about 15 lines. It:
  - creates a cached 1-element tensor;
  - captures `t.add_(0.0)` into a dummy `CUDAGraph`, in `capture_error_mode="thread_local"` so the tiny window cannot
    interfere with the background build threads;
  - deletes the dummy, whose reset clears the default capture stream's cuBLAS workspace entry;
  - returns `torch.cuda.graph(g)`.
- **Every capture site now uses `with _rt_cap(...)`** instead of `with torch.cuda.graph(...)`:
  - `_rt_graph`, which is used by every copy, layer and gate check, including `_dsa_copy_check`;
  - `_sk_graph_of`, used by the sk and dl3 alternation checks and the selectors;
  - `_dl_self_check`;
  - `_dl_pdl_probe`.

**Why a per-stream clear and not the global `torch._C._cuda_clearCublasWorkspaces()`.** The baseline, and any graph
the harness keeps alive, runs cuBLAS on a pooled side stream (`baseline.py` `state["side_stream"]`). A global clear
during our prepare could free that stream's workspace under a live foreign graph. The per-stream clear is exactly what
torch already does on every graph destruction, so it adds no new risk class.

**Okc rules.** These are prepare-time helpers only:

- no data-derived value is cached across calls;
- no host sync is added to `kernel_fn`;
- `kernel_fn` and the timed path are byte-identical.

## Validation

- **Deterministic dl3 reproducer:** master faulted in 9 of 9 processes; fixed in 0 of 2.
- **Prepare-time DSA copy check:** master faulted in 6 of 10 processes (6 faults in 325 checks; 4 of 4 in the
  deterministic configuration); fixed in 0 of 9 processes (0 faults in 585 checks).
- **Stale-workspace detector:** fixed `_rt_graph` overwrote 0 bytes at every T, both at prepare time and after it,
  with DSA on and off.
- **Full nodejudge decode run of the fixed file** (`20261030-223052`, config `sr-fx-val`, GPU 6, cold builds,
  judge-like E2b inputs, router_seed 2306202727525448386):
  - 13 of 13 tests pass. Prepare took 122.3 s against the 300 s budget (enforced); import with cold builds took
    199.7 s. The estimated judge run is 393 s of 900.
  - `[dense-swapab] copy check: 32 replays on two states, T [1, 8, 16, 32]: PASS`.
  - The PATH CHOSEN lines still show swap-AB:
    - T=1: qkv_gate SWAP-AB, o SWAP-AB.
    - T=2..8: SWAP-AB.
    - T=9..16: SWAP-AB, unpooler master (pinned).
    - T=17..32: SWAP-AB (pinned).
  - Engine A copy check: 64 replays, PASS. Headline x1.4453 [1.4236, 1.4670] against SGLang on this H100. That number
    is not judge-comparable; `kernel_fn` is unchanged.

## How to reproduce

```bash
cd ~/git/takemyjob/prototypes/hopper-probe
HP_BUNDLE=sr-fx HP_ONLY=sr-fx bin/hopper-probe build gpunode06        # every plan in probes/sr-fx/plans.txt
HP_BUNDLE=sr-fx bin/hopper-probe run gpunode06 --gpus 5 sr-fx-dsarep  # DSA copy check: master faults at check 7, fix passes (x2)
HP_BUNDLE=sr-fx bin/hopper-probe run gpunode06 --gpus 5 sr-fx-why     # dl3 fault with alias + allocator trace (master)
HP_BUNDLE=sr-fx bin/hopper-probe run gpunode06 --gpus 5 sr-fx-ws      # stale-workspace detector, master then fix
HP_BUNDLE=sr-fx bin/hopper-probe run gpunode06 --gpus 5 sr-fx-clean   # remove /tmp/USER-hopper/sr-fx-cache afterwards
```

Use one GPU per run. Plan `core` adds a GPU coredump, which is about 0.3 GB and is deleted after cuda-gdb 13.3 has read
it.

## Remaining risk

1. **Not identical to the judge.** I could not replay the judge's own H200 state: its allocator history, its harness,
   and the real SGLang kernels instead of the node's shims. Two parts of the case are shown: the H100 prepare-time
   reproduction of the identical log signature in the identical stage, and the stale writes of T=1/8 graphs at
   prepare time. That the cb182920 fault was this event is an inference, but the strongest available. cuBLAS
   heuristics on the H200 can differ in which GEMMs use the workspace; the fix covers any GEMM, because every captured
   graph owns its workspace.
2. **Version dependence.** The dummy-graph clear relies on `CUDAGraph::reset()` calling
   `clearCublasWorkspacesForStream`, which is true in torch 2.14.1, the judge's version. In a torch without that call,
   graph destruction does not free the workspace either, so the hazard does not arise.
3. **Outside our control.** The judge harness's own captures could hit the same class: a warm-up on a stream that
   aliases the capture stream, plus destroying graphs between capture and replay. That would affect any submission
   and the baseline alike.
4. **Separate finding, not a crash.** On some weights, the dl3 alternation check reports a false FAIL: hidden 0.14
   with `ids_ok True`. A near-tie routing flip sends piece A to local expert 124 where the parent picks expert 208
   (`[sr-ids]`, run `20261030-215242`), and the check's 2e-2 hidden tolerance cannot absorb a flipped local expert.
   Piece A then falls back to the parent chain, which costs a little speed but is safe. This is left as is.
5. **Silent corruption is gone too.** The same stale writes could corrupt a tensor that reused the freed address, for
   example causing a copy check to FAIL and fall back. The fix removes that as well.

## Review notes (workflow wf_ed98bfd8-716 verify stage; main thread 10-31 ~07:15Z)
- **Confidence:** the mechanism is SHOWN on the H100, by the hooked prepare-time DSA copy checks and the natural-flow dl3
  seed 2693396613634879665. cb182920 is attributed to it by stage and log signature only: the judge log names neither the
  faulting kernel nor the T. Natural single-check DSA reproduction on the H100 is 0 of 32.
- **Independent verify (GPU 7, sr-v1):**
  - master faulted in 8 of 10 stressed prepares (dsarep 2/2, fresh weights 3/4, allocator churn 3/4); the fix faulted in
    0 of 10, with 586 checks and 0 sentinel bytes corrupted;
  - plain-torch mechanism test (mech.py): the default-pool aliased path faults raw 6/6 and 0/6 with `_rt_cap`;
  - nodejudge A/B: both 13/13 and swap-AB chosen, headline level (master x1.4585, fix level).
- **Overclaim withdrawn:** "fixed _rt_graph overwrote 0 B" proves nothing. In the fixed rows the sentinel never sat at
  the old address, because `_rt_cap`'s empty_cache released it first.
- **Invariant scope:** per-graph ownership holds only for the default capture stream's cuBLAS workspace. The t128 fork's
  side-stream cuBLAS (`_t128_side` F.linear, `_t128_router` torch.mm) still shares one workspace across graphs. That is
  safe while no capture runs on those streams and nothing calls _cuda_clearCublasWorkspaces. The _rt_cap comment now
  says so.
- **Open risk, outside our control:** the judge harness's own timed-path captures (several graphs per shape on the
  default capture stream) could hit the same class if it destroys one graph between another's capture and replay. The
  SGLang baseline shares that exposure.
- **Cost:** about +3 s warm prepare (33.5 → 36.4 s); one 32 MiB workspace per live graph.
- **Attribution diagnostics added (+237 B):** a per-T progress line in _dsa_copy_check, and the DSA except handler now
  prints the head and tail of the error (the driver-logged messages were cut at 400 chars before). If the fault recurs
  on the H200, the log names the T and the driver message.
