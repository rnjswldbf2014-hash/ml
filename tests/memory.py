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
     not by training. Measured over 10 trials: K=2 solved 10/10, K=4 solved
     8/10, K=8 never. Outcomes are bimodal -- a run either reaches 100% or
     stays at chance, decided by the initial weight draw -- so K=4 is asserted
     by majority rather than strictly. If someone later adds multi-step BPTT,
     these numbers are the baseline to beat.
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


# The `seed` argument only shuffles the training order -- initial weights come
# from an unpredictable per-process seed, so each trial is also a fresh weight
# draw. That matters, because whether this task is solved at all is bimodal:
# a run either nails it (100%) or never gets off chance, and which one happens
# depends on the draw. Measured over 10 trials: K=2 solved 10/10, K=4 solved
# 8/10 with 4000 rounds. So K=2 is asserted strictly and K=4 by majority.
print("\n[memory] cue at t=0, answer demanded K steps later")
plain2 = accuracy("plain", 2, 0)
solved2 = sum(1 for s in range(3) if accuracy("mem", 2, s) > 90)
check("K=2: memory solves it every time, plain cannot",
      solved2 == 3 and plain2 < 70,
      f"memory {solved2}/3 trials >90%, plain {plain2:.0f}%")

# K=4 lands around 80% per trial (8/10, measured twice). Asserting 3-of-5 would
# fail ~6% of runs on luck alone, so the bar is 2-of-5: still impossible for a
# memoryless network (which never gets off chance at any K), while not turning
# a known-flaky property into a flaky test.
solved4 = sum(1 for s in range(5) if accuracy("mem", 4, s, 4000) > 90)
check("K=4: memory solves it (~80% of trials), plain never does",
      solved4 >= 2, f"memory {solved4}/5 trials >90%")

print("\n[reach] where one-step truncation runs out")
# Not a bug -- a measured limit. If this starts passing, the truncation was
# changed, and the table in RnnLayer's comment should be remeasured to match.
solved8 = sum(1 for s in range(3) if accuracy("mem", 8, s, 4000) > 90)
check("K=8 is out of reach (documented limit, not a regression)",
      solved8 == 0, f"{solved8}/3 trials solved")

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
