"""Test harness for the jepa (JEPA / world-model) path in rnjswldbf_2014/ml.d.

Three groups of checks:

  1. WORLD  — does it actually learn? A hidden 6-cell ring world is observed
     through a fixed per-cell vector plus fresh noise on every look. The noise
     is unpredictable by construction, so matching raw observations is a losing
     game; matching only the *summary* is not. We assert that after training
     the summary (a) identifies the cell through the noise and (b) lets
     imagine(obs, action) land on the real next cell.

  2. COLLAPSE — turning the guard off (var=0, cov=0) must actually collapse
     (every input summarised to the same value, loss -> 0). This is the
     control: it proves the variance/covariance term is what prevents it,
     rather than the model happening not to collapse on its own.

  3. MISC — save/load round-trip, attn/each inside either network, the
     action-free (plain self-supervised) form, misuse guards, and the shared
     trunk case: one model holding both a vec summary (trained by jepa) and a
     pick head (trained by sl/rl) off the same weights.

  4. DETERMINISM — training from one shared weight snapshot under
     MYML_THREADS=1/4/8/default must be bit-identical, for pure-Linear,
     attn- and each-containing encoders. Same invariant the sl() path is held
     to in regression.py; jepa always uses the batched path, so unlike
     regression.py there is no nobatch tier and no tolerance tier.

Usage: python tests/jepa.py     (exit 0 = pass)
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
CHILD = os.path.join(TESTS, "jepa_child.py")

DET_TOPOLOGIES = ["linear", "attn", "each"]
DET_CONFIGS = [("threads1", {"MYML_THREADS": "1"}),
               ("threads4", {"MYML_THREADS": "4"}),
               ("threads8", {"MYML_THREADS": "8"}),
               ("default",  {})]


def build():
    r = subprocess.run(["powershell", "-File", os.path.join(TESTS, "build.ps1"),
                        "-OutDir", MODULE_DIR],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit("build failed")
    print(r.stdout.strip())


def run(workdir, topo, mode, env=None):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    return workdir, subprocess.run(
        [sys.executable, CHILD, workdir, topo, mode], capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=ROOT, env={**os.environ, **(env or {})})


def line_starting(proc, prefix):
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith(prefix):
            return ln
    return None


ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {note}" if note else ""))


build()
work = os.path.join(SCRATCH, "_jepa")

# ── 1 & 2: does it learn, and is the collapse guard what stops collapse? ──
print("\n[world] ring-world scenario")
_, proc = run(work, "world", "world")
ln = line_starting(proc, "WORLD ")
if ln is None:
    print(proc.stdout); print(proc.stderr)
    check("world scenario ran", False)
    m = {}
else:
    m = dict(kv.split("=") for kv in ln.split()[1:])
    m = {k: float(v) for k, v in m.items()}
    check("summary identifies the cell through noise", m["ident"] > 0.90,
          f"{m['ident']*100:.1f}% (chance 16.7%)")
    check("imagine() lands on the real next cell", m["next"] > 0.80,
          f"{m['next']*100:.1f}% (chance 16.7%)")
    check("summaries stay spread out", m["spread"] > 0.3, f"spread={m['spread']:.4f}")

print("\n[collapse] same scenario with the guard off (var=0, cov=0)")
_, proc = run(work, "nocollapseguard", "world")
ln2 = line_starting(proc, "WORLD ")
if ln2 is None:
    print(proc.stdout); print(proc.stderr)
    check("collapse-control scenario ran", False)
else:
    m2 = dict(kv.split("=") for kv in ln2.split()[1:])
    m2 = {k: float(v) for k, v in m2.items()}
    check("guard off => summaries collapse", m2["spread"] < 0.05,
          f"spread={m2['spread']:.6f} vs {m.get('spread', float('nan')):.4f} with guard")
    check("guard off => loss falls to ~0 (the trivial solution)", m2["loss"] < 1e-3,
          f"loss={m2['loss']:.6f}")

# ── 3: bit-exact across thread counts ────────────────────────────────────
print("\n[determinism] identical results across thread counts")
snap = os.path.join(SCRATCH, "_jepa_snap")
for topo in DET_TOPOLOGIES:
    run(snap, topo, "snapshot", {"MYML_THREADS": "1"})
    weights = [f for f in os.listdir(snap) if f.endswith(".pth")]
    if not weights:
        check(topo, False, "snapshot not created")
        continue

    results = {}
    for name, env in DET_CONFIGS:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        for f in weights:
            shutil.copy(os.path.join(snap, f), work)
        proc = subprocess.run([sys.executable, CHILD, work, topo, "det"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=ROOT, env={**os.environ, **env})
        hx = line_starting(proc, "HEX ")
        if hx is None:
            print(proc.stdout); print(proc.stderr)
            check(f"{topo}/{name}", False, "no output")
            continue
        results[name] = hx

    base = results.get("threads1")
    diff = [k for k, v in results.items() if v != base]
    check(topo, len(results) == len(DET_CONFIGS) and not diff,
          "bit-identical for threads 1/4/8/default" if not diff else f"differs: {diff}")

shutil.rmtree(snap, ignore_errors=True)

# ── 4: save/load round-trip, mixed topologies, misuse guards ─────────────
print("\n[misc] save/load, mixed layer kinds, misuse guards")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work, exist_ok=True)
sys.path.insert(0, MODULE_DIR)
os.chdir(work)
import struct
import random
import ml as my_ml
from ml import make, vec, cos, attn, each, jepa


def wipe():
    for f in os.listdir("."):
        if f.endswith(".pth"):
            os.remove(f)


def hexs(vals):
    return " ".join(struct.pack("<f", float(v)).hex() for v in vals)


def data(rnd, B, inw, actw):
    x = [[rnd.gauss(0, 1) for _ in range(inw)] for _ in range(B)]
    y = [[rnd.gauss(0, 1) for _ in range(inw)] for _ in range(B)]
    a = None if actw == 0 else [[rnd.gauss(0, 1) for _ in range(actw)] for _ in range(B)]
    return x, y, a


def blocked(fn):
    try:
        fn(); return False
    except Exception:
        return True


wipe()
e = make("sv_enc", [12, 32], [vec(6)])
p = make("sv_prd", [6 + 3, 32], [vec(6)])
w = jepa(e, p)
rnd = random.Random(3)
for _ in range(20):
    w.train(*data(rnd, 32, 12, 3))
x = [0.11 * i - 0.5 for i in range(12)]
before, before_im = w.encode(x), w.imagine(x, [0.2, -0.4, 0.9])
w.save()
w2 = jepa(make("sv_enc", [12, 32], [vec(6)]), make("sv_prd", [6 + 3, 32], [vec(6)]))
check("encode() survives save/load bit-exactly", hexs(before) == hexs(w2.encode(x)))
check("imagine() survives save/load bit-exactly",
      hexs(before_im) == hexs(w2.imagine(x, [0.2, -0.4, 0.9])))
check("embed() matches encode()", hexs(w2.encoder.embed(x)) == hexs(before))

for label, enc_l, prd_l in [
    ("attn in encoder",  [12, attn(4), 32],           [6 + 2, 32]),
    ("each in encoder",  [12, attn(4), each(8), 32],  [6 + 2, 32]),
    ("attn in predictor", [12, 32],                   [6 + 2, attn(2), 32]),
    ("both mixed",       [12, attn(4), each(8), 24],  [6 + 2, attn(4), each(4), 32]),
]:
    wipe()
    try:
        ww = jepa(make("m_enc", enc_l, [vec(6)]), make("m_prd", prd_l, [vec(6)]))
        r = random.Random(9)
        first = ww.train(*data(r, 32, 12, 2))
        for _ in range(60):
            last = ww.train(*data(r, 32, 12, 2))
        check(label, first == first and last == last and last < first,
              f"loss {first:.3f} -> {last:.3f}")
    except Exception as ex:
        check(label, False, f"raised {ex}")

wipe()
w3 = jepa(make("na_enc", [12, 32], [vec(6)]), make("na_prd", [6, 32], [vec(6)]))
r = random.Random(4)
first = w3.train(*data(r, 32, 12, 0))
for _ in range(100):
    last = w3.train(*data(r, 32, 12, 0))
check("action-free form (plain self-supervised) trains", w3.actions == 0 and last < first,
      f"loss {first:.3f} -> {last:.3f}")

wipe()
e4 = make("g_enc", [12, 32], [vec(6)])
w4 = jepa(e4, make("g_prd", [6 + 2, 32], [vec(6)]))
check("mismatched x/y counts rejected",
      blocked(lambda: w4.train([[0.0] * 12] * 4, [[0.0] * 12] * 3, [[0.0, 0.0]] * 4)))
check("missing actions rejected",
      blocked(lambda: w4.train([[0.0] * 12] * 4, [[0.0] * 12] * 4)))
wipe()
bad = make("b_prd", [6 + 2, 32], [vec(5)])
check("summary-size mismatch rejected", blocked(lambda: jepa(e4, bad)))
wipe()
check("non-vec first output rejected as encoder",
      blocked(lambda: jepa(make("nv", [12, 32], [cos]),
                           make("nv2", [1 + 2, 32], [vec(6)]))))

# ── shared trunk: one model carrying both a vec summary and a pick head ──
# The summary is trained by jepa, the pick head by sl() — same weights
# underneath, so the self-supervised signal also shapes the acting head.
wipe()
acts = ["a", "b", "c"]
mix = make("mix_enc", [12, 32], [vec(6), acts])
mw = jepa(mix, make("mix_prd", [6 + 2, 32], [vec(6)]))
r = random.Random(13)
mfirst = mw.train(*data(r, 32, 12, 2))
for _ in range(60):
    mlast = mw.train(*data(r, 32, 12, 2))
check("jepa trains a model that also has a pick head", mlast == mlast and mlast < mfirst,
      f"loss {mfirst:.3f} -> {mlast:.3f}")
X = [[r.gauss(0, 1) for _ in range(12)] for _ in range(24)]
Y = [[None, acts[i % 3]] for i in range(24)]
mix.sl(X, Y)
out = mix.predict(X[0])
check("predict(): vec slot is None, pick slot is a real answer",
      out[0] is None and out[1] in acts, f"{out}")
check("embed(x, 0) returns the whole summary", len(mix.embed(X[0], 0)) == 6)
check("supervising a vec slot is rejected",
      blocked(lambda: mix.sl(X, [[0.0, acts[0]]] * 24)))
step = mix.rl(X[0])
check("rl() works, leaving the vec slot None", step.output[0] is None and step.output[1] in acts)
check("scoring a vec slot is rejected", blocked(lambda: mix.reward(step, [1.0, 1.0])))
check("scoring only the pick slot is fine", mix.save(mix.reward(step, [None, 1.0])) == 1)

os.chdir(ROOT)
shutil.rmtree(work, ignore_errors=True)
print("\n" + ("ALL JEPA CHECKS PASS" if ok else "JEPA CHECKS FAILED"))
sys.exit(0 if ok else 1)
