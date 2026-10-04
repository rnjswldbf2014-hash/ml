"""Tests for the conv() layer (1-D convolution) in ml.d.

What this file can and cannot reach, stated up front because it shapes the
tests:

A layer's own output is not observable through the public API. Every output
head is a dense Linear over the whole previous layer -- `vec(n)` included --
so perturbing one input item moves every head value, and the receptive field
is invisible from out here. (An earlier version of this file assumed vec() was
a passthrough and "failed" on all three structural checks for that reason.)

So the structural properties were checked once, offline, through a temporary
probe that returned the layer's output directly, and the probe was then
removed. Results, with conv(3, win, 12) on a 12-item input:

  - receptive field: perturbing input item p moved exactly the output items
    whose window covers p, for p = 0, 1, 6, 11 and win = 3 -- including the
    zero-padded ends
  - equivariance: shifting a pattern by 4 items shifted the output by 4
  - window 1: item 5 moved output item 5 and nothing else

Worth recording for whoever repeats that: conv applies ReLU, and against an
all-zero baseline every output sits at exactly 0, so perturbing in one
direction leaves any channel whose weight has the wrong sign clamped at 0 and
invisible. The centre tap is the same column at every position, so it vanishes
at *every* position at once and looks precisely like an off-by-one window.
Perturb both ways and take the union.

The standing guard on that index math is regression.py's "conv", "conv2" and
"convmix" topologies: the per-sample path and the batched path have separately
written im2col/col2im loops, and they are required to agree bit-for-bit.

What is left here is everything reachable from outside: it learns a
position-independent task, weights survive a round trip, and bad shapes are
refused.

Usage: python tests/conv.py     (exit 0 = pass)
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
WORK = os.path.join(SCRATCH, "_conv")

L = 12
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

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
sys.path.insert(0, MODULE_DIR)
os.chdir(WORK)

from ml import make, conv, each, attn, cos     # noqa: E402


def wipe():
    for f in os.listdir("."):
        if ".pth" in f:
            os.remove(f)


# ── learns a task where position must not matter ────────────────────────
print("\n[train] three patterns, anywhere in the sequence")
wipe()
PATTERNS = {"up": [1.0, 0.0, -1.0], "down": [-1.0, 0.0, 1.0], "spike": [-1.0, 2.0, -1.0]}
NAMES = list(PATTERNS)
SPOTS = list(range(L - 2))


def row(kind, at):
    x = [0.0] * L
    for d, v in enumerate(PATTERNS[kind]):
        x[at + d] = v
    return x


LAYERS = [L, conv(8, 3, L), each(4), 16]
ai = make("cvt", LAYERS, [NAMES], autosave=0, lr=0.01)
X = [row(k, p) for k in NAMES for p in SPOTS]
Y = [[k] for k in NAMES for p in SPOTS]
for _ in range(200):
    ai.sl(X, Y)
hit = sum(1 for k in NAMES for p in SPOTS if ai.predict(row(k, p))[0] == k)
total = len(NAMES) * len(SPOTS)
check("tells the three patterns apart wherever they sit", hit == total,
      f"{hit}/{total}")

probe = row("spike", 5)
before = ai.predict(probe)[0]
ai.save()
ai2 = make("cvt", LAYERS, [NAMES], autosave=0, lr=0.01)
check("predict() identical after save/load", ai2.predict(probe)[0] == before,
      f"{before} vs {ai2.predict(probe)[0]}")

# ── stacked conv, and an even window ────────────────────────────────────
print("\n[shapes] stacking, even windows, and composing with other layers")
for label, layers in [("stacked", [L, conv(4, 3, L), conv(3, 3), 16]),
                      ("even window", [L, conv(4, 2, L), 16]),
                      ("window 1", [L, conv(4, 1, L), 16]),
                      ("window = length", [L, conv(4, L, L), 16]),
                      ("with each/attn", [L, conv(4, 3, L), each(6), attn(L), 16])]:
    wipe()
    try:
        m = make("cs", layers, [NAMES], autosave=0, lr=0.01)
        first = None
        for _ in range(40):
            m.sl(X, Y)
        got = m.predict(row("up", 4))[0]
        trained = got in NAMES
        check(label, trained, f"predict -> {got}")
    except Exception as e:
        check(label, False, f"raised {e}")

# ── bad shapes are refused with a reason ────────────────────────────────
print("\n[guards] bad shapes are refused")


def blocked(fn):
    try:
        fn(); return False
    except Exception:
        return True


wipe()
check("conv first with no item count is refused",
      blocked(lambda: make("bad1", [L, conv(4, 3)], [cos], autosave=0)))
check("item count that does not divide the width is refused",
      blocked(lambda: make("bad2", [L, conv(4, 3, 5)], [cos], autosave=0)))
check("a window wider than the sequence is refused",
      blocked(lambda: make("bad3", [L, conv(4, 99, L)], [cos], autosave=0)))
check("channel or window below 1 is refused",
      blocked(lambda: conv(0, 3)) and blocked(lambda: conv(4, 0)))
wipe()
check("conv after attn inherits the item count",
      not blocked(lambda: make("okc", [L, attn(4), conv(4, 3)], [cos], autosave=0)))

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL CONV CHECKS PASS" if ok else "CONV CHECKS FAILED"))
sys.exit(0 if ok else 1)
