"""Tests for the logic() layer (neuro-symbolic gate learning) in ml.d.

Three things matter here, and the first two are what make the layer worth
having over a plain Linear:

  1. LEARNS — give it a problem that *is* a boolean function and it solves it.
     With logic(1) there is exactly one gate between the inputs and the head,
     so the only way to fit XOR is for that gate to actually become XOR.

  2. READS BACK — rules() must name the gate it settled on. This is the whole
     point of the layer; a Linear can fit the same function and tell you
     nothing. A learned gate may come out as its complement (XNOR for XOR,
     NAND for AND): the head is a Linear and can flip the sign, so the circuit
     is right either way. The check accepts a gate or its complement.

  3. SURVIVES A ROUND TRIP — each unit's two input wires are drawn at random
     when the layer is built, so they have to be written to the weight file.
     Forget that and a reloaded model is a different circuit entirely, while
     still looking fine until you compare outputs. This is the cheapest
     mistake to make and the hardest to notice, so it gets a direct test.

Determinism and batched-vs-serial equivalence for logic layers live in
regression.py, as the "logic", "logic2" and "logicmix" topologies.

Usage: python tests/logic.py     (exit 0 = pass)
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
WORK = os.path.join(SCRATCH, "_logic")

# Index == truth table; the complement of gate g is gate 15-g. Mirrors
# GATE_NAME in ml.d -- if that list changes, this one has to change with it.
GATES = ["거짓", "a∧b", "a∧¬b", "a", "¬a∧b", "b", "a⊕b", "a∨b",
         "¬(a∨b)", "a↔b", "¬b", "a∨¬b", "¬a", "¬a∨b", "¬(a∧b)", "참"]

TRUE, FALSE = 6.0, -6.0      # sigmoid() takes these to 0.9975 / 0.0025

ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {note}" if note else ""))


def build():
    r = subprocess.run(["powershell", "-File", os.path.join(TESTS, "build.ps1"),
                        "-OutDir", MODULE_DIR],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit("build failed")
    print(r.stdout.strip())


def names_for(target):
    """The expressions that count as correct for gate `target`, as written over
    x0 and x1 -- the gate or its complement, with either input order."""
    out = set()
    for g in (target, 15 - target):
        for a, b in (("x0", "x1"), ("x1", "x0")):
            out.add(GATES[g].replace("a", a).replace("b", b))
    return out


build()
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
sys.path.insert(0, MODULE_DIR)
os.chdir(WORK)

from ml import make, logic, cos   # noqa: E402


def wipe():
    for f in os.listdir("."):
        if f.endswith(".pth"):
            os.remove(f)


# ── 1 & 2: learns a boolean function, and names the gate it learned ──────
print("\n[gates] one gate between input and head, so it has to BE the function")
TRUTH = {"XOR": (6, lambda a, b: a ^ b),
         "AND": (1, lambda a, b: a & b),
         "OR":  (7, lambda a, b: a | b),
         "NAND": (14, lambda a, b: 1 - (a & b))}

for label, (gate, fn) in TRUTH.items():
    wipe()
    ai = make(f"lg_{label}", [2, logic(1)], [["0", "1"]], autosave=0)
    X = [[TRUE if a else FALSE, TRUE if b else FALSE]
         for a in (0, 1) for b in (0, 1)]
    Y = [[str(fn(a, b))] for a in (0, 1) for b in (0, 1)]
    for _ in range(400):
        ai.sl(X, Y)
    hit = sum(1 for x, y in zip(X, Y) if ai.predict(x)[0] == y[0])
    expr = ai.rules()[0][0].split()[0]      # drop the "  (NN%)" suffix
    check(f"{label}: fits the truth table", hit == len(X), f"{hit}/{len(X)}")
    check(f"{label}: rules() names it", expr in names_for(gate),
          f"got {expr}, accepted {sorted(names_for(gate))}")

# ── 3: the random input wiring has to survive save/load ─────────────────
print("\n[round trip] random input wiring and gate weights persist")
wipe()
LAYERS = [6, 12, logic(10), logic(10), 8]
ai = make("lg_rt", LAYERS, [cos], autosave=0)
rows = [[0.3 * ((i * 7 + k * 3) % 5) - 0.6 for k in range(6)] for i in range(16)]
targets = [[0.1 * (i % 4)] for i in range(16)]
for _ in range(200):
    ai.sl(rows, targets)

# Several probes, not one. A single probe can land somewhere the wiring does
# not matter: if every ReLU feeding the logic layer is dead for that input,
# all its inputs are sigmoid(0) = 0.5 and any wiring gives the same answer.
# That is exactly what made an earlier version of this check pass against a
# build that deliberately did not persist the wiring.
probes = [[0.2, -0.4, 0.1, 0.5, -0.3, 0.0],
          [-0.7, 0.9, -0.2, 0.3, 0.8, -0.5],
          [1.2, 0.4, -1.1, -0.6, 0.2, 0.7],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [-1.5, -1.2, 0.6, 1.4, -0.9, 1.1]]
before_out = [ai.predict(p)[0] for p in probes]
before_rules = ai.rules()
# A probe set that produces the same answer everywhere proves nothing about
# the wiring, so make sure it does not.
check("probe set actually distinguishes outputs", len(set(before_out)) > 1,
      f"{before_out}")
ai.save()

ai2 = make("lg_rt", LAYERS, [cos], autosave=0)
after_out = [ai2.predict(p)[0] for p in probes]
check("predict() identical after reload", after_out == before_out,
      f"{before_out} vs {after_out}")
check("rules() identical after reload", ai2.rules() == before_rules)
check("rules() returns one list per logic layer", len(before_rules) == 2,
      f"got {len(before_rules)}")
check("each list has one entry per gate",
      all(len(r) == 10 for r in before_rules),
      f"got {[len(r) for r in before_rules]}")

# A model with no logic layer must simply report nothing, not fail.
wipe()
plain = make("lg_none", [4, 8], [cos], autosave=0)
check("no logic layer => rules() is empty", plain.rules() == [])

# ── stacked logic layers on a harder function ───────────────────────────
print("\n[stacked] 3-input majority through two logic layers")
wipe()
maj = make("lg_maj", [3, logic(8), logic(8)], [["0", "1"]], autosave=0)
X, Y = [], []
for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            X.append([TRUE if v else FALSE for v in (a, b, c)])
            Y.append([str(1 if a + b + c >= 2 else 0)])
for _ in range(1500):
    maj.sl(X, Y)
hit = sum(1 for x, y in zip(X, Y) if maj.predict(x)[0] == y[0])
check("majority fits", hit == len(X), f"{hit}/{len(X)}")
check("both stacked layers report rules", len(maj.rules()) == 2)

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL LOGIC CHECKS PASS" if ok else "LOGIC CHECKS FAILED"))
sys.exit(0 if ok else 1)
