"""Tests for the memory() layer (a gated recurrent layer) in ml.d.

The task is built so memory is the only way through: a cue appears at t=0, the
input is then identical for K steps, and the answer is the cue. A network that
sees only the current input is guessing.

What this pins down:

  1. It works where a plain network cannot — 100% vs chance at short ranges.
  2. **How far it reaches, and that this depends on how you call it.** Calling
     sl() one step at a time leaves no chain inside a single call (B=1), so the
     gradient is truncated to one step; intermediate steps carry no loss at all
     (answer None), so the gate never learns *when* to open or close and stays
     near its initial 0.5. Reach is then set by forward decay (~0.5^K) rather
     than by training. Handing the whole episode over as one bundle instead
     lets the backward pass walk the chain (BPTT), and the gate does learn.
     Measured over 10 trials, [2, memory(16), 16], lr=0.02, 4000 rounds:

                      K=4     K=8     K=16    K=32
         one at a time  9/10    1/10    0/10    0/10
         bundled       10/10   10/10    5/10    1/10

     Outcomes are bimodal either way -- a run either reaches 100% or stays at
     chance, decided by the initial weight draw -- so the K=4 and K=8 bars are
     majorities, not absolutes.
  3. forget() really clears the memo, and without it the previous episode
     leaks into the next one.
  4. BPTT is on by default and only the bundled path can do it: B=1 must be
     bit-identical with MYML_BPTT either way, and a real bundle must differ.
     That second one is what catches BPTT silently not running.

The batched path is NOT compared against the serial path here -- the serial
path interleaves forward and backward per sample, so the rest of the chain does
not exist yet when a sample's backward runs, and BPTT is structurally
impossible there. tests/regression.py runs the memory topologies with
MYML_BPTT=0 so that comparison still has something to say.

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

# K=4 lands around 80% per trial (8/10, measured twice). With 5 trials a 2-of-5
# bar still failed a run on luck alone (1/5 came up once), so this uses 8 trials
# and asks for 3 -- at p=0.8 that is a ~0.1% flake rate, and a memoryless
# network never gets off chance at any K, so the bar still separates them.
solved4 = sum(1 for s in range(8) if accuracy("mem", 4, s, 4000) > 90)
check("K=4: memory solves it (~80% of trials), plain never does",
      solved4 >= 3, f"memory {solved4}/8 trials >90%")

print("\n[reach] where one-step truncation runs out")
# Not a bug -- a measured limit of calling sl() one step at a time. There is no
# chain inside a single call (B=1), so the gradient cannot go anywhere.
solved8 = sum(1 for s in range(3) if accuracy("mem", 8, s, 4000) > 90)
check("K=8 is out of reach one step at a time (documented, not a regression)",
      solved8 == 0, f"{solved8}/3 trials solved")

# ── BPTT: hand the episode over as one bundle ───────────────────────────
# The bundle IS the sequence (the memo runs in order inside the layer), so the
# backward pass can walk back down it instead of stopping after one step.
# Measured over 10 seeds at K=8: 1/10 one-at-a-time, 10/10 bundled. The bar is
# 4-of-5 rather than 5-of-5 because the whole task stays bimodal on the weight
# draw -- but at K=8 the gap between the two call shapes is not subtle.
print("\n[bptt] the same K=8, handed over as a bundle instead")
solved8b = sum(1 for s in range(5) if accuracy("bundle", 8, s, 4000) > 90)
check("K=8 IS in reach when the episode is bundled", solved8b >= 4,
      f"bundled {solved8b}/5 trials >90%  (one-at-a-time: {solved8}/3)")

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

# ── BPTT is on by default, and only the bundled path can use it ─────────
# Two properties, both bit-exact, both from the SAME weight snapshot:
#   singles: one sl() per step has no chain in it (B=1), so MYML_BPTT must make
#            no difference at all. If it does, the reverse walk is leaking
#            state across calls.
#   bundle:  one sl() for the whole sequence does have a chain, so MYML_BPTT
#            must make a difference. If it does not, BPTT silently is not
#            running and the K=8 result above was luck.
print("\n[bptt] on by default; B=1 unaffected, a real bundle affected")
BCHILD = os.path.join(TESTS, "memory_bptt_child.py")
BWORK = os.path.join(SCRATCH, "_bptt")


def bptt_run(mode, on, cwd):
    env = dict(os.environ)
    env["MYML_BPTT"] = "1" if on else "0"
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, BCHILD, MODULE_DIR, mode], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"bptt child failed ({mode}, bptt={on})")
    if mode == "init":
        return ""                       # init only persists the snapshot
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln and all(c in "0123456789abcdef," for c in ln):
            return ln
    raise SystemExit(f"bptt child printed nothing usable ({mode})")


shutil.rmtree(BWORK, ignore_errors=True)
os.makedirs(BWORK)
bptt_run("init", True, BWORK)                      # one shared snapshot
SNAP = os.path.join(BWORK, "bp_ml_memory.pth")
results = {}
for mode in ("singles", "bundle"):
    for on in (True, False):
        d = os.path.join(BWORK, f"{mode}_{int(on)}")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(SNAP, os.path.join(d, "bp_ml_memory.pth"))
        results[(mode, on)] = bptt_run(mode, on, d)

check("B=1 (one sl() per step): BPTT changes nothing, bit for bit",
      results[("singles", True)] == results[("singles", False)])
check("a real bundle: BPTT changes the answer, so it is actually running",
      results[("bundle", True)] != results[("bundle", False)])
shutil.rmtree(BWORK, ignore_errors=True)

print("\n" + ("ALL MEMORY CHECKS PASS" if ok else "MEMORY CHECKS FAILED"))
sys.exit(0 if ok else 1)
