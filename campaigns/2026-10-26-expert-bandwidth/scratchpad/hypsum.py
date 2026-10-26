"""Compact summary of queued hypotheses: python3 -I hypsum.py [problem ...]"""
import glob
import os
import re
import sys

ROOT = "/home/user/git/musespark"
PICK = ("combo-verify-cells", "sliding-decode-sk-s3", "mc4-cluster")
probs = sys.argv[1:] or ["muse_bf16_proj", "muse_attention_decode", "muse_prefill_layer"]
for prob in probs:
    for f in sorted(glob.glob(f"{ROOT}/{prob}/hypotheses/queue/*.md")):
        base = os.path.basename(f)
        if not (base.startswith("2026-10-26T22") or any(s in base for s in PICK)):
            continue
        t = open(f).read()
        parts = t.split("\n---\n", 1)
        fm, body = parts[0], (parts[1] if len(parts) > 1 else "")

        def key(k):
            m = re.search(rf"^{k}: *(.*)$", fm, re.M)
            return m.group(1).strip() if m else ""

        def sec(name):
            m = re.search(rf"^## {name}\n+(.*?)(?=^## |\Z)", body, re.M | re.S)
            return m.group(1).strip() if m else ""

        ch = " ".join(sec("Change").split())
        rk = " ".join(sec("Risk").split())
        ex = " | ".join(" ".join(l.split()) for l in sec("Expected").splitlines() if l.strip())
        tags = " ".join(f"{k}={key(k)}" for k in ("kind", "stage", "priority", "risk", "review", "status", "depends_on"))
        print(f"### {prob} :: {key('slug')}  [{tags}]")
        print("AREA:", key("area"))
        print("CHANGE:", ch[:650])
        print("EXPECTED:", ex[:420])
        print("RISK/SIZING:", rk[-420:])
        print()
