"""Guards on the OpenCL GPU path in gpu_cl.d / ml.d.

The important one is the first: **the default configuration must not use the
GPU.** MYML_GPU used to default to "auto", and back when the kernels were an
untiled matmul with uncoalesced reads that silently made training 17-25x
slower once a batch crossed the FLOP threshold -- with correct results, so
nobody could tell why it crawled.

The kernels are tiled now and land around parity with the 12-thread CPU path
(slightly ahead only on the largest shapes measured). That is not enough of a
win to justify switching users over by default on hardware nobody benchmarked,
so it stays opt-in: MYML_GPU=1 (force) or MYML_GPU=auto (threshold). Flip the
default only with a benchmark to point at.

The rest checks that the GPU path still computes the right thing when it is
asked for, so it does not rot while off by default -- including shapes that do
not line up with the tile size, which is where tiled kernels usually break.

Skips itself cleanly when no OpenCL GPU is present.

Usage: python tests/gpu.py     (exit 0 = pass or skip)
"""
import os
import shutil
import struct
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TOLERANCE = 1e-3        # GPU sums in a different order; this is not a bit-exact tier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
CHILD = os.path.join(TESTS, "gpu_child.py")
WORK = os.path.join(SCRATCH, "_gpu_work")
SNAP = os.path.join(SCRATCH, "_gpu_snap")

# Big enough to clear the default auto-mode thresholds (5e7 FLOPs, batch 64),
# which is exactly the regime where the old default silently switched over.
LAYERS = [128, 256, 256]
BATCH = 256

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


def run(mode, gpu, batch=BATCH, layers=None):
    env = {**os.environ}
    if gpu is None:
        env.pop("MYML_GPU", None)          # unset == whatever the library defaults to
    else:
        env["MYML_GPU"] = gpu
    proc = subprocess.run(
        [sys.executable, CHILD, ",".join(map(str, layers or LAYERS)), str(batch),
         mode, WORK],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT, env=env)
    out = {}
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("HEX "):
            out["hex"] = ln
        elif ln.startswith("RUNS "):
            out["runs"] = int(ln[5:])
        elif ln.startswith("INFO "):
            for kv in ln.split()[1:]:
                k, _, v = kv.partition("=")
                out[k] = v
    if not out and mode != "snapshot":
        print(proc.stdout); print(proc.stderr)
        raise SystemExit(f"child failed (mode={mode}, MYML_GPU={gpu})")
    return out


def floats(line):
    return [struct.unpack("<f", bytes.fromhex(h))[0] for h in line.split()[1:]]


def snapshot(layers=None):
    for d in (SNAP, WORK):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    run("snapshot", "0", layers=layers)
    files = [f for f in os.listdir(WORK) if f.endswith(".pth")]
    for f in files:
        shutil.copy(os.path.join(WORK, f), SNAP)
    return files


def restore(files):
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    for f in files:
        shutil.copy(os.path.join(SNAP, f), WORK)


build()
os.makedirs(WORK, exist_ok=True)

info = run("info", "1")
has_gpu = info.get("available") == "1"
print(f"\n[gpu] OpenCL device: {info.get('device') or '(none)'}")
if not has_gpu:
    print("  no OpenCL GPU on this machine -- skipping (not a failure)")
    shutil.rmtree(WORK, ignore_errors=True)
    sys.exit(0)

files = snapshot()

print("\n[default] the default configuration must not route training onto the GPU")
restore(files)
d = run("probe", None)
check("default config leaves the GPU alone", d.get("runs", -1) == 0,
      f"GPU path ran {d.get('runs')} times on a {LAYERS} x{BATCH} batch")

print("\n[opt-in] the GPU path still works when explicitly asked for")
restore(files)
c = run("probe", "0")
restore(files)
g = run("probe", "1")
ran = g.get("runs", 0) > 0
check("MYML_GPU=1 actually engages the GPU path", ran, f"runs={g.get('runs')}")
if ran:
    a, b = floats(c["hex"]), floats(g["hex"])
    err = max(abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b))
    check("GPU result matches the CPU result", err < TOLERANCE,
          f"max relative error {err:.2e} (tolerance {TOLERANCE})")

# ── tile-boundary correctness ────────────────────────────────────────────
# The matmul kernels tile the work 16x16 and round the launch up, relying on
# in-kernel bounds checks plus zero-filled local tiles to stay correct when a
# dimension is not a multiple of 16. Getting that wrong yields plausible-looking
# but wrong numbers, and a suite that only ever uses round sizes never sees it.
print("\n[tiles] sizes that do not divide evenly by the 16x16 tile")
for layers, batch in [([17, 33, 19], 7),        # every dimension below one tile
                      ([16, 16, 16], 16),       # exactly one tile
                      ([100, 150, 70], 100),    # nothing aligned
                      ([1, 32, 1], 65),         # single-element input and output
                      ([255, 257, 33], 255),    # one off the tile boundary
                      ([129, 16, 300], 17)]:
    files = snapshot(layers)
    restore(files)
    c = run("probe", "0", batch, layers)
    restore(files)
    g = run("probe", "1", batch, layers)
    if g.get("runs", 0) == 0:
        check(f"{layers} x{batch}", False, "GPU path did not engage")
        continue
    a, b = floats(c["hex"]), floats(g["hex"])
    err = max(abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b))
    check(f"{layers} x{batch}", err < TOLERANCE, f"relative error {err:.2e}")

shutil.rmtree(SNAP, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL GPU CHECKS PASS" if ok else "GPU CHECKS FAILED"))
sys.exit(0 if ok else 1)
