"""Rewrite LOG.md's "## Now" block and append an entry: python3 -I log_now.py NOW_FILE [ENTRY_FILE]."""
import sys

P = "/home/user/git/musespark/campaigns/2026-10-26-expert-bandwidth/LOG.md"
s = open(P).read()
a = s.index("\n## Now\n") + 1
b = s.index("\n## 20", a) + 1
now = open(sys.argv[1]).read().strip() + "\n\n"
s = s[:a] + now + s[b:]
if len(sys.argv) > 2:
    s = s.rstrip() + "\n\n" + open(sys.argv[2]).read().strip() + "\n"
open(P, "w").write(s)
print("ok")
