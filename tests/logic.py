"""Tests for the logic() layer, plus the weight-file version handling it bumped.

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

The last section covers the file format. Every new layer kind bumps it, because
an older reader would mistake the new kind for an each layer (logic took it to
ver 10, memory to ver 11). The rules under test: old versions still load, saving
upgrades them in place, and a file from some *newer* version is moved aside
rather than silently overwritten on the next save -- "cannot read it" is not the
same as "safe to throw away".

Usage: python tests/logic.py     (exit 0 = pass)
"""
import os
import random
import shutil
import struct
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

import ml                        # noqa: E402
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
rnd = random.Random(5)
rows = [[rnd.gauss(0, 1) for _ in range(6)] for _ in range(24)]
targets = [[sum(r[:3]) * 0.2] for r in rows]
for _ in range(40):
    ai.sl(rows, targets)

# Comparing predict() across inputs is the obvious way to check that the wiring
# survived -- and it does not work here, for a reason worth writing down:
#
# The 16 gates pair up into complements (gate k and gate 15-k always sum to 1),
# so a uniform softmax over them averages to exactly 0.5 for *any* a and b. Gate
# weights start at zero, so a freshly built logic layer outputs a constant 0.5
# no matter what it is fed, and everything downstream of it is constant too.
# Training breaks that symmetry, but train much and the gates settle on
# constants again and the network goes flat a second time. Either way the output
# can stop depending on the input, which would make an output comparison pass
# without testing anything. (An earlier version of this check did exactly that
# against a build that deliberately dropped the wiring.)
#
# So the round trip is checked against state instead of behaviour: rules()
# covers the logic layers (wiring + gate weights), export_weights() covers the
# ordinary Linear layers, and predict() equality is kept as a cheap necessary
# condition rather than the main evidence.
probes = [[0.2, -0.4, 0.1, 0.5, -0.3, 0.0],
          [-0.7, 0.9, -0.2, 0.3, 0.8, -0.5],
          [1.2, 0.4, -1.1, -0.6, 0.2, 0.7],
          [-1.5, -1.2, 0.6, 1.4, -0.9, 1.1]]
before_out = [ai.predict(p)[0] for p in probes]
before_rules = ai.rules()
before_w = ml._ml_export_weights(ai._h)
ai.save()

ai2 = make("lg_rt", LAYERS, [cos], autosave=0)
after_out = [ai2.predict(p)[0] for p in probes]
after_w = ml._ml_export_weights(ai2._h)
check("rules() identical after reload (wiring + gate weights)",
      ai2.rules() == before_rules)
check("Linear weights identical after reload",
      all(before_w[k] == after_w[k] for k in before_w if k.endswith(".weight")),
      f"{sorted(k for k in before_w if k.endswith('.weight'))}")
check("predict() identical after reload", after_out == before_out,
      f"{before_out} vs {after_out}")
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

# ── weight-file version handling ────────────────────────────────────────
print("\n[version] ver 9 loads, save upgrades it, newer files are kept")
VPTH = "vr_ml_memory.pth"
VLAY = [6, 12, 10]


def file_ver(p):
    with open(p, "rb") as f:
        magic, v = struct.unpack("<II", f.read(8))
    assert magic == 0xBEEFCAFE, hex(magic)
    return v


def set_file_ver(p, v):
    with open(p, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<I", v))


def downgrade_to_v11(p, nlay):
    """Rewrite a ver-12 file as a genuine ver-11 one.

    Stamping the version number alone is not enough any more: ver 12 widened
    each layer spec from three words to four (conv needs channels, window and
    item count), so a ver-11 reader expects a shorter header. Leaving the extra
    words in place desynchronises the whole stream and every weight after it
    comes out as garbage -- which is exactly how this check first failed.
    """
    with open(p, "rb") as f:
        blob = f.read()
    head = 4 * 5                       # magic, ver, opt, inputSz, nLay
    specs = blob[head:head + nlay * 16]
    kept = b"".join(specs[i * 16:i * 16 + 12] for i in range(nlay))
    out = blob[:4] + struct.pack("<I", 11) + blob[8:head] + kept \
        + blob[head + nlay * 16:]
    with open(p, "wb") as f:
        f.write(out)


wipe()
vr = make("vr", VLAY, [cos], autosave=0)
rows = [[0.1 * k for k in range(6)]] * 4
for _ in range(10):
    vr.sl(rows, [[0.3]] * 4)
vprobe = [0.2, -0.4, 0.1, 0.5, -0.3, 0.0]
vbefore = vr.predict(vprobe)[0]
vr.save()
# One place to change on the next format bump (12 -> 13 was lattn).
CURRENT_VER = 13
check(f"save() writes the current version ({CURRENT_VER})", file_ver(VPTH) == CURRENT_VER,
      f"got {file_ver(VPTH)}")

# Turn it into a real ver-11 file (narrower layer specs), not just a restamped one.
downgrade_to_v11(VPTH, len(VLAY) - 1)
check("the downgraded file says ver 11", file_ver(VPTH) == 11, f"got {file_ver(VPTH)}")
vr2 = make("vr", VLAY, [cos], autosave=0)
check("an older-version file still loads, weights intact",
      vr2.predict(vprobe)[0] == vbefore,
      f"{vbefore} vs {vr2.predict(vprobe)[0]}")
vr2.save()
check("saving an older file upgrades it to the current version", file_ver(VPTH) == CURRENT_VER,
      f"got {file_ver(VPTH)}")

# A file from a newer version cannot be parsed, but it must not be destroyed.
set_file_ver(VPTH, 99)
with open(VPTH, "rb") as f:
    newer_bytes = f.read()
vr3 = make("vr", VLAY, [cos], autosave=0)
bak = VPTH + ".bak"
moved = os.path.exists(bak) and open(bak, "rb").read() == newer_bytes
check("newer-version file is moved aside intact", moved)
vr3.save()
still = os.path.exists(bak) and open(bak, "rb").read() == newer_bytes
check("the moved file survives the next save()", still)

# A second failure must not clobber the first backup.
set_file_ver(VPTH, 99)
make("vr", VLAY, [cos], autosave=0)
check("a second rescue numbers the backup", os.path.exists(VPTH + ".bak2"))

# Changing the layer spec is a deliberate reset, so it should NOT leave backups.
wipe()
for f in os.listdir("."):
    if ".pth" in f:
        os.remove(f)
make("vr", [6, 12, 10], [cos], autosave=0).save()
make("vr", [6, 99, 10], [cos], autosave=0)
leftovers = sorted(f for f in os.listdir(".") if f.startswith(VPTH))
check("a structure change leaves no backup", leftovers == [VPTH], f"{leftovers}")

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL LOGIC CHECKS PASS" if ok else "LOGIC CHECKS FAILED"))
sys.exit(0 if ok else 1)
