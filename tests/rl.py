"""Regression harness for the RL learning path (save(scored)) in ml.d.

learnBatch() gained a batched path (forwardBatch/backwardBatch, the same core
sl() uses). This checks it did not change what the model learns:

  - STRICT (bit-exact): threads 1/4/8/default all run the batched path with
    different actual thread counts and must agree bit-for-bit. A mismatch
    means a real race or a summation-order bug in _parChunk.
  - TOLERANCE (1e-5): MYML_NOBATCH=1 (the per-sample serial path) vs the
    batched baseline. Same two-tier scheme, and same caveat, as
    regression.py -- see its docstring for why attn/each topologies can
    differ by ~1 ULP under ldc2 -O3.

Also checks the autosave policy: save(scored) must only touch the weight file
on the schedule make(..., autosave=N) asks for, and save() with no argument
must always write.

Usage: python tests/rl.py     (exit 0 = pass)
"""
import os
import shutil
import struct
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TOLERANCE = 1e-5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
CHILD = os.path.join(TESTS, "rl_child.py")

TOPOLOGIES = ["linear", "attn", "each", "mixed"]
CONFIGS = [("threads1", {"MYML_THREADS": "1"}),
           ("threads4", {"MYML_THREADS": "4"}),
           ("threads8", {"MYML_THREADS": "8"}),
           ("default",  {}),
           ("nobatch",  {"MYML_NOBATCH": "1"})]

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


def floats(line):
    return [struct.unpack("<f", bytes.fromhex(h))[0] for h in line.split()[1:]]


def close_enough(a, b):
    return len(a) == len(b) and all(
        abs(x - y) <= TOLERANCE * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


build()
snap = os.path.join(SCRATCH, "_rl_snap")
work = os.path.join(SCRATCH, "_rl_work")

print("\n[equivalence] batched RL path vs the per-sample serial path")
for topo in TOPOLOGIES:
    for d in (snap, work):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    subprocess.run([sys.executable, CHILD, snap, topo, "snapshot"],
                   capture_output=True, cwd=ROOT,
                   env={**os.environ, "MYML_THREADS": "1"})
    weights = [f for f in os.listdir(snap) if f.endswith(".pth")]
    if not weights:
        check(topo, False, "snapshot not created")
        continue

    results = {}
    for name, env in CONFIGS:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        for f in weights:
            shutil.copy(os.path.join(snap, f), work)
        proc = subprocess.run([sys.executable, CHILD, work, topo, "run"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=ROOT, env={**os.environ, **env})
        line = next((l for l in (proc.stdout or "").splitlines()
                     if l.startswith("HEX ")), None)
        if line is None:
            print(proc.stdout); print(proc.stderr)
            check(f"{topo}/{name}", False, "no output")
            continue
        results[name] = line

    base = results.get("threads1")
    differing = [k for k, v in results.items() if k != "nobatch" and v != base]
    serial_ok = ("nobatch" in results
                 and close_enough(floats(base), floats(results["nobatch"])))
    exact = results.get("nobatch") == base
    check(topo, len(results) == len(CONFIGS) and not differing and serial_ok,
          ("threads 1/4/8/default bit-identical; serial "
           + ("bit-identical" if exact else f"within {TOLERANCE}"))
          if not differing and serial_ok
          else f"thread mismatch={differing}, serial_ok={serial_ok}")

shutil.rmtree(snap, ignore_errors=True)

# ── autosave policy ──────────────────────────────────────────────────────
print("\n[autosave] save(scored) only writes the weight file when asked to")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
sys.path.insert(0, MODULE_DIR)
os.chdir(work)
from ml import make, cos, Scored

ACTIONS = ["A", "B"]
PATH = "as_ml_memory.pth"


def one(ai):
    return Scored([0.5] * 8, [ACTIONS[0], 0.0], [1.0, None], [0, 0], [0.0, 0.0])


def mtime():
    return os.path.getmtime(PATH) if os.path.exists(PATH) else None


ai = make("as", [8, 16], [ACTIONS, cos], autosave=0)
if os.path.exists(PATH):
    os.remove(PATH)
for _ in range(5):
    ai.save(one(ai))
check("autosave=0 never writes on its own", not os.path.exists(PATH))
ai.save()
check("save() with no argument always writes", os.path.exists(PATH))

ai.autosave = 3
os.remove(PATH)
ai.save(one(ai)); ai.save(one(ai))
wrote_early = os.path.exists(PATH)
ai.save(one(ai))
check("autosave=3 writes on the 3rd call, not before",
      not wrote_early and os.path.exists(PATH))

ai2 = make("as2", [8, 16], [ACTIONS, cos])
check("default is autosave=1 (writes every time, unchanged behaviour)",
      ai2.autosave == 1)
os.chdir(ROOT)
shutil.rmtree(work, ignore_errors=True)

print("\n" + ("ALL RL CHECKS PASS" if ok else "RL CHECKS FAILED"))
sys.exit(0 if ok else 1)
