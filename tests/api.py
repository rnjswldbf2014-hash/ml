"""Tests for the calling conventions -- the rules a user has to hold in their head.

The library had collected a pile of implicit rules, several of which failed
silently. This file pins down the ones that were removed, so they do not creep
back:

  - make() takes three things positionally. Everything else is keyword-only,
    so there is no argument order to memorise, and all of it is also settable
    afterwards as a property.
  - A single output does not have to be wrapped in a list, and neither do the
    reward / answer / legal that go with it. Lists keep working.
  - each() and conv() no longer require a particular layer in front of them;
    they take an explicit item count instead.
  - A model with memory() warns once if forget() was never called. That was
    the worst of the old rules: nothing failed, training just quietly got
    worse, because every episode ran into the next one.
  - memory() works with bundled training now (it used to refuse), and the
    bundled result matches the one-at-a-time result -- regression.py's
    "memory" and "memorymix" topologies hold that line.

Usage: python tests/api.py     (exit 0 = pass)
"""
import inspect
import os
import shutil
import subprocess
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
WORK = os.path.join(SCRATCH, "_api")

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

import ml                                                   # noqa: E402
from ml import make, cos, vec, attn, each, conv, memory      # noqa: E402

ACTS = ["A", "B"]


def wipe():
    for f in os.listdir("."):
        if ".pth" in f:
            os.remove(f)


def blocked(fn):
    try:
        fn(); return False
    except Exception:
        return True


# ── make(): three positional, the rest keyword-only ─────────────────────
print("\n[make] three things to remember, the rest by name")
params = list(inspect.signature(make).parameters.values())
pos = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
kw = [p for p in params if p.kind is p.KEYWORD_ONLY]
check("three positional parameters (+ optimizer)", len(pos) == 4,
      f"{[p.name for p in pos]}")
check("the tuning knobs are keyword-only", len(kw) == 6, f"{[p.name for p in kw]}")
wipe()
check("passing a knob positionally is refused",
      blocked(lambda: make("m", [4, 8], cos, 'adam', 0.5)))
wipe()
ai = make("m", [4, 8], cos, lr=0.02, decay=0.001, temp=0.5, autosave=0)
check("knobs given by name land where expected",
      abs(ai.lr - 0.02) < 1e-9 and abs(ai.decay - 0.001) < 1e-9
      and abs(ai.temp - 0.5) < 1e-9)

# ── a single output needs no wrapping ───────────────────────────────────
print("\n[one output] no list needed, and lists still work")
for label, outs in [("cos bare", cos), ("cos wrapped", [cos]),
                    ("actions bare", ACTS), ("actions wrapped", [ACTS]),
                    ("vec bare", vec(4)), ("vec wrapped", [vec(4)])]:
    wipe()
    try:
        m = make("o", [4, 8], outs, autosave=0)
        check(label, m.heads == 1, f"heads={m.heads}")
    except Exception as e:
        check(label, False, f"raised {e}")

wipe()
two = make("t", [4, 8], [ACTS, cos], autosave=0)
check("two outputs still parse as two", two.heads == 2, f"heads={two.heads}")

print("\n[one output] reward / answer / legal take a bare value too")
wipe()
m = make("b", [4, 8], ACTS, autosave=0)
x = [0.1, 0.2, 0.3, 0.4]
step = m.rl(x)
check("reward(step, 1.0) without a list", m.save(m.reward(step, 1.0)) == 1)
check("reward(step, [1.0]) still works", m.save(m.reward(step, [1.0])) == 1)
check("sl(x, \"A\") without a list", m.sl(x, "A") is not None)
check("legal=[\"A\"] without nesting", m.rl(x, legal=["A"]).output[0] == "A")
check("legal=[[\"A\"]] still works", m.rl(x, legal=[["A"]]).output[0] == "A")
# With two outputs the list is still required, and the message says so.
check("two outputs still require a list for the reward",
      blocked(lambda: two.reward(two.rl(x), 1.0)))

# ── each / conv no longer need a particular layer in front ──────────────
print("\n[items] each() and conv() take their own item count")
wipe()
check("each(width, items) works with nothing in front",
      not blocked(lambda: make("e1", [12, each(5, 4)], cos, autosave=0)))
wipe()
check("each(width) after attn still inherits",
      not blocked(lambda: make("e2", [12, attn(4), each(5)], cos, autosave=0)))
wipe()
check("each(width) with no source is refused, with a reason",
      blocked(lambda: make("e3", [12, each(5)], cos, autosave=0)))
wipe()
check("conv(ch, win, items) works with nothing in front",
      not blocked(lambda: make("c1", [12, conv(4, 3, 12)], cos, autosave=0)))
wipe()
check("each after conv inherits the item count",
      not blocked(lambda: make("c2", [12, conv(4, 3, 12), each(2)], cos, autosave=0)))

# ── memory: warns once, and bundles ─────────────────────────────────────
print("\n[memory] the silent trap now speaks up")
wipe()
mem = make("mm", [4, memory(6), 8], ACTS, autosave=0)
with warnings.catch_warnings(record=True) as got:
    warnings.simplefilter("always")
    mem.sl(x, "A")
    mem.sl(x, "A")
check("training without forget() warns", len(got) == 1,
      f"{len(got)} warnings: {[str(w.message)[:40] for w in got]}")

wipe()
mem2 = make("mm2", [4, memory(6), 8], ACTS, autosave=0)
with warnings.catch_warnings(record=True) as got2:
    warnings.simplefilter("always")
    mem2.forget()
    mem2.sl(x, "A")
check("no warning once forget() has been called", len(got2) == 0,
      f"{[str(w.message)[:40] for w in got2]}")

wipe()
nomem = make("nm", [4, 8], ACTS, autosave=0)
with warnings.catch_warnings(record=True) as got3:
    warnings.simplefilter("always")
    nomem.sl(x, "A")
check("a model without memory() never warns", len(got3) == 0)

print("\n[memory] bundled training is allowed now")
wipe()
mb = make("mb", [4, memory(6), 8], cos, autosave=0)
mb.forget()
rows = [[0.1 * k, 0.2, -0.1, 0.3] for k in range(6)]
tgts = [[0.1 * k] for k in range(6)]
try:
    for _ in range(5):
        mb.sl(rows, tgts)
    bundled = True
except Exception as e:
    bundled = False
    print(f"      raised: {e}")
check("bundled sl() on a memory network runs", bundled)

wipe()
enc = make("je", [4, memory(6), 8], vec(6), autosave=0)
prd = make("jp", [6 + 2, 8], vec(6), autosave=0)
try:
    w = ml.jepa(enc, prd)
    w.train(rows, rows, [[1.0, 0.0]] * 6)
    jepa_ok = True
except Exception as e:
    jepa_ok = False
    print(f"      raised: {e}")
check("memory() can be a jepa encoder now", jepa_ok)

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL API CHECKS PASS" if ok else "API CHECKS FAILED"))
sys.exit(0 if ok else 1)
