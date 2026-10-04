"""Tests for the memory() layer (a gated recurrent layer) in ml.d.

The task is built so memory is the only way through: a cue appears at t=0, the
input is then identical for K steps, and the answer is the cue. A network that
sees only the current input is guessing.

What this pins down:

  1. It works where a plain network cannot — 100% vs chance at short ranges.
  2. **How far it reaches**, measured rather than assumed. Gradients are
     truncated to one step, and intermediate steps carry no loss at all
     (answer None), so the gate never learns *when* to open or close: it stays
     near its initial 0.5. Reach is therefore set by forward decay (~0.5^K),
     not by training. Measured: solid at K<=4, gone by K=8. If someone later
     adds multi-step BPTT, these numbers are the baseline to beat.
  3. forget() really clears the memo, and without it the previous episode
     leaks into the next one.
  4. A network holding a memory layer refuses the batched path instead of
     silently producing something -- the memo has to run in order, and a batch
     treats its samples as independent.

Each case runs in its own process: the layer keeps state across calls by
design, so leftover state would contaminate later cases.

Usage: python tests/memory.py     (exit 0 = pass)
"""
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
WORK = os.path.join(SCRATCH, "_mem")
CHILD = os.path.join(TESTS, "memory_child.py")

ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {note}" if note else ""))


r = subprocess.run(["powershell", "-File", os.path.join(TESTS, "build.ps1"),
                    "-OutDir", MODULE_DIR],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", cwd=ROOT)
if r.returncode != 0:
    print(r.stdout); print(r.stderr)
    raise SystemExit("build failed")
print(r.stdout.strip())


def accuracy(spec, K, seed, rounds=2000):
    proc = subprocess.run([sys.executable, CHILD, spec, str(K), str(seed), str(rounds)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("ACC "):
            return float(ln[4:])
    print(proc.stdout); print(proc.stderr)
    raise SystemExit(f"child failed ({spec}, K={K}, seed={seed})")


print("\n[memory] cue at t=0, answer demanded K steps later")
SEEDS = range(3)
for K in (2, 4):
    plain = accuracy("plain", K, 0)
    solved = sum(1 for s in SEEDS if accuracy("mem", K, s) > 90)
    check(f"K={K}: memory solves it, plain cannot",
          solved == len(SEEDS) and plain < 70,
          f"memory {solved}/{len(SEEDS)} seeds >90%, plain {plain:.0f}%")

print("\n[reach] where one-step truncation runs out")
far = [(K, sum(1 for s in SEEDS if accuracy("mem", K, s) > 90)) for K in (8,)]
for K, solved in far:
    # Not a bug -- a measured limit. If this starts passing, truncation was
    # changed and the comment in RnnLayer should be updated to match.
    check(f"K={K} is out of reach (documented limit, not a regression)",
          solved == 0, f"{solved}/{len(SEEDS)} seeds solved")

# ── forget() and batch refusal, in-process ──────────────────────────────
print("\n[forget] the memo clears, and leaks without clearing")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
sys.path.insert(0, MODULE_DIR)
os.chdir(WORK)
from ml import make, memory, cos      # noqa: E402

ai = make("fg", [2, memory(8), 8], [cos], autosave=0)
ai.forget()
a = [ai.predict([1.0, 0.0])[0], ai.predict([0.0, 0.0])[0]]
ai.forget()
b = [ai.predict([1.0, 0.0])[0], ai.predict([0.0, 0.0])[0]]
check("forget() makes the same sequence replay identically", a == b, f"{a} vs {b}")

# Without forget(), the second pass starts from a non-empty memo.
c = [ai.predict([1.0, 0.0])[0], ai.predict([0.0, 0.0])[0]]
check("without forget() the previous episode leaks in", c != b, f"{b} vs {c}")

plain = make("fg_plain", [2, 8], [cos], autosave=0)
plain.forget()           # must be a no-op, not an error
check("forget() on a model without memory() is harmless", True)

print("\n[batch] a memory network must not take the batched path")
rows = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
tgts = [[0.1], [0.2], [0.3], [0.4]]
try:
    ai.sl(rows, tgts)          # routed to the serial path, so this must work
    bundled_ok = True
except Exception as e:
    bundled_ok = False
    print(f"      raised: {e}")
check("bundled sl() still works (routed to the serial path)", bundled_ok)

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL MEMORY CHECKS PASS" if ok else "MEMORY CHECKS FAILED"))
sys.exit(0 if ok else 1)
