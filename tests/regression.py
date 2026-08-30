"""
Determinism regression harness for rnjswldbf_2014/ml.d.

For each network topology (pure-Linear / each / attn / mixed), this:
  1. builds ml.pyd once (tests/build.ps1),
  2. creates one canonical initial weight snapshot per topology
     (tests/probe.py mode=init),
  3. runs a fixed, RNG-free sl() training sequence + predict() probe
     from that SAME snapshot under several MYML_* configs, each in its
     own subprocess (env vars are read once at module load, so a fresh
     process is required per config).

Two comparison tiers:
  - STRICT (bit-exact): threads1 / threadsN / default all run through the
    exact same batched code path (forwardBatch/backwardBatch), just with
    different actual thread counts. These MUST be bit-identical — any
    mismatch here means _parChunk introduced a real race/order bug.
  - TOLERANCE (relative error < 1e-5): nobatch (the old per-sample serial
    path) vs the batched baseline. For topologies with NO attn layer this
    is still bit-exact in practice. For attn-containing topologies, a rare
    (~1-in-10 to 1-in-40 runs), ~1-ULP difference has been observed that
    is reproducible ONLY under ldc2 -O3 (absent at -O2 and -O0), survives
    bounds-checked and vectorization-disabled builds, and occurs even
    though weights/gradients are provably bit-identical (verified via
    exact hex dumps) between the two paths — i.e. it's very likely an
    LDC/LLVM -O3 codegen quirk in how @target("avx2,fma") multiversioned
    functions get compiled, not a data race or logic bug. Investigated
    extensively (see project notes); tracked here as a known, harmless,
    accepted limitation rather than blocking on it further.

This does not (and cannot, without unpicking the RNG) verify anything
about rl()/reward() sampling.

Usage: python tests/regression.py
Exit code 0 = all checks pass. Non-zero = FAIL.
"""
import os
import shutil
import struct
import subprocess
import sys

TOLERANCE = 1e-5

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")

TOPOLOGIES = ["linear", "each", "attn", "mixed"]

CONFIGS = [
    ("threads1", {"MYML_THREADS": "1"}),
    ("threadsN", {"MYML_THREADS": "4"}),
    ("nobatch", {"MYML_NOBATCH": "1"}),
    ("default", {}),
]


def build_module():
    if os.path.isdir(MODULE_DIR):
        shutil.rmtree(MODULE_DIR)
    os.makedirs(MODULE_DIR, exist_ok=True)
    cmd = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", os.path.join(TESTS, "build.ps1"),
        "-OutDir", MODULE_DIR,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"build failed (exit {r.returncode})")


def run_probe(mode, topo, name, cwd, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    cmd = [sys.executable, os.path.join(TESTS, "probe.py"), MODULE_DIR, mode, topo, name]
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(
            f"probe failed (topo={topo} mode={mode} env={extra_env})\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
    # D side (writefln, e.g. "새로 생성되었습니다") can print an extra line to
    # stdout around ml.make() — its buffering isn't reliably ordered before
    # or after probe.py's own print(), so pick out the line that actually
    # looks like the comma-separated hex payload rather than assuming
    # first/last.
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln and all(c in "0123456789abcdefABCDEF," for c in ln):
            return ln
    return ""


def decode_hexes(hexline):
    return [struct.unpack("<d", bytes.fromhex(h))[0] for h in hexline.split(",")]


def close_enough(a_hex, b_hex, tol):
    a, b = decode_hexes(a_hex), decode_hexes(b_hex)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
            return False
    return True


def main():
    build_module()
    if os.path.isdir(SCRATCH):
        for d in os.listdir(SCRATCH):
            if d != "_module":
                shutil.rmtree(os.path.join(SCRATCH, d), ignore_errors=True)

    all_ok = True
    for topo in TOPOLOGIES:
        name = f"probe_{topo}"
        init_dir = os.path.join(SCRATCH, topo, "_init")
        os.makedirs(init_dir, exist_ok=True)
        run_probe("init", topo, name, init_dir, {})
        pth_name = f"{name}_ml_memory.pth"
        snapshot = os.path.join(init_dir, pth_name)
        if not os.path.isfile(snapshot):
            all_ok = False
            print(f"[{topo}] FAIL: init did not produce {pth_name}")
            continue

        outputs = {}
        for cfg_name, extra_env in CONFIGS:
            cfg_dir = os.path.join(SCRATCH, topo, cfg_name)
            os.makedirs(cfg_dir, exist_ok=True)
            shutil.copyfile(snapshot, os.path.join(cfg_dir, pth_name))
            outputs[cfg_name] = run_probe("run", topo, name, cfg_dir, extra_env)

        # tier 1: threads1/threadsN/default must be bit-exact (same code path)
        strict_baseline = outputs["threads1"]
        strict_mismatches = [
            (cfg, outputs[cfg]) for cfg in ("threadsN", "default")
            if outputs[cfg] != strict_baseline
        ]
        if strict_mismatches:
            all_ok = False
            print(f"[{topo}] FAIL (strict): batched path not bit-identical across thread counts")
            for cfg, val in strict_mismatches:
                print(f"    {cfg}: {val}")
            print(f"    threads1 (baseline): {strict_baseline}")
            continue

        # tier 2: nobatch (old serial path) vs batched — tolerance-based
        # (see module docstring for why this isn't required to be bit-exact)
        if close_enough(outputs["nobatch"], strict_baseline, TOLERANCE):
            print(f"[{topo}] OK: threads1/threadsN/default bit-identical, "
                  f"nobatch within tolerance ({TOLERANCE})")
        else:
            all_ok = False
            print(f"[{topo}] FAIL (tolerance): nobatch vs batched exceeds {TOLERANCE}")
            print(f"    nobatch:  {outputs['nobatch']}")
            print(f"    threads1: {strict_baseline}")

    if not all_ok:
        raise SystemExit(1)
    print("ALL TOPOLOGIES PASS (batched path bit-identical across thread counts; "
          "nobatch within tolerance)")


if __name__ == "__main__":
    main()
