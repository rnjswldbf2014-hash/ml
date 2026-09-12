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

# ── optimizer state left on the device ───────────────────────────────────
# The GPU path keeps Adam's m/v on the device between calls and only pulls them
# back when the host needs them (a CPU step(), or save()). That is worth ~30% of
# the runtime, but a missed sync point strands stale m/v on the host with no
# visible symptom until the weights are written out -- training keeps running
# and keeps looking plausible. So compare a GPU run against a CPU-only run of
# the same fixed sequence, including sequences that alternate between the two.
print("\n[sync] Adam state survives GPU/CPU interleaving")
SYNC_CHILD = os.path.join(TESTS, "gpu_sync_child.py")
SYNC_LAYERS = [64, 128, 128]


SYNC_PTH = "sy_ml_memory.pth"


def sync_run(seq, gpu, start=None):
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    # every run has to start from the same weights -- make() randomises a fresh
    # model, so without this the two runs are simply different models
    if start is not None:
        with open(os.path.join(WORK, SYNC_PTH), "wb") as f:
            f.write(start)
    env = {**os.environ, "MYML_GPU": gpu,
           # low enough that the big batch clears it, so batch size alone picks
           # the route and one process can alternate
           "MYML_GPU_MIN_FLOPS": "1000000", "MYML_GPU_MIN_B": "64"}
    proc = subprocess.run([sys.executable, SYNC_CHILD, WORK, seq],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT, env=env)
    path = os.path.join(WORK, SYNC_PTH)
    if not os.path.exists(path):
        print(proc.stdout); print(proc.stderr)
        raise SystemExit("sync child produced no weight file")
    runs, probe = 0, None
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("RUNS "):
            runs = int(ln[5:])
        elif ln.startswith("PROBE "):
            probe = floats(ln)
    with open(path, "rb") as f:
        blob = f.read()
    # header: magic, ver, opt, inputSz, nLay, nLay*(kind,a,b), nHeads,
    #         then per head (outSz, cos, nActions) -- a cos head carries no names
    nlay = len(SYNC_LAYERS) - 1
    head = 4 * 5 + nlay * 12 + 4 + 12
    vals = struct.unpack(f"<{(len(blob) - head) // 4}f", blob[head:head + ((len(blob) - head) // 4) * 4])
    return vals, runs, probe


# Start from a model that already has a few CPU steps on it, not a fresh one.
# On Adam's very first step the second moment is still zero, so the update
# degenerates to +/-lr per weight -- the magnitude of the gradient drops out and
# only its sign matters. Any gradient that lands near zero then flips sign
# between two float summation orders and the weight moves the opposite way, so a
# from-scratch comparison diverges for reasons that have nothing to do with
# syncing. A few steps in, the second moment is non-zero and the two paths track
# each other closely.
sync_run("ccc", "0")
with open(os.path.join(WORK, SYNC_PTH), "rb") as f:
    SYNC_START = f.read()

def deviation(ref, got):
    """Largest disagreement, measured against the scale of the array.

    Per-element relative error is the wrong tool here: weights and Adam's first
    moment cross zero, so a value that lands near 0.0 produces a huge relative
    error from pure rounding while meaning nothing. (Second moments never cross
    zero and do stay within 1e-5 element-wise.) Scaling by the array's RMS keeps
    the check sensitive to a whole block being stale -- which is what a missed
    sync looks like -- without tripping on sign noise.
    """
    rms = (sum(x * x for x in ref) / max(1, len(ref))) ** 0.5
    return max(abs(a - b) for a, b in zip(ref, got)) / max(rms, 1e-6)


for seq in ["gggg", "ggcc", "gcgcgc", "cggc"]:
    ref, _, refp = sync_run(seq, "0", SYNC_START)
    got, runs, gotp = sync_run(seq, "auto", SYNC_START)
    if runs == 0:
        check(f"sequence {seq}", False, "no GPU steps actually ran")
        continue
    dev = deviation(ref, got)
    check(f"saved weights after {seq} ({runs} GPU steps)", dev < 0.05,
          f"largest disagreement vs CPU-only, over array RMS: {dev:.2e}")
    # predict() is probed before save(). Weights now stay on the device between
    # GPU calls, so a missing sync on the forward path answers from the stale
    # host copy -- and save() would paper over it by syncing on its own.
    perr = max(abs(a - b) / max(1e-2, abs(a), abs(b)) for a, b in zip(refp, gotp))
    check(f"predict() after {seq}", perr < 0.02, f"relative difference {perr:.2e}")

shutil.rmtree(SNAP, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL GPU CHECKS PASS" if ok else "GPU CHECKS FAILED"))
sys.exit(0 if ok else 1)
