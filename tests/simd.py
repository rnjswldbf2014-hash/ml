"""Tests the SIMD dispatch -- and in particular the AVX-512 path, which the
author could not run.

The development machine is a Ryzen 5 7535HS: znver3 (Zen 3+), which does NOT
support AVX-512 (LLVM reports -avx512f for -mcpu=native). The AVX-512 kernels
are therefore written but have never executed on real hardware. What protects
them:

  1. They are generated from the SAME mixin string as the AVX2 and scalar
     kernels, so the arithmetic cannot drift between them. Only the @target
     attribute differs.
  2. The generated code was checked to actually use 512-bit registers
     (adamRow_avx512: 33 zmm instructions, 0 ymm; adamRow_avx2: 0 zmm, 30 ymm).
  3. This file. On a machine that has AVX-512 it runs the same fixed training
     sequence at every available SIMD level and compares the results. If you
     have such a CPU, running this is the verification that was missing.

Why tolerance and not bit-exact: the optimizer kernels are element-wise, so
vector width cannot change their results at all. But dot() is a reduction, and
a wider vector sums in a different order -- so the last bits legitimately move
between SIMD levels. (This is why two different CPUs were never expected to
agree bit-for-bit; MYML_THREADS is the one that must, and regression.py covers
it.) A broken kernel would miss by far more than 1e-5.

Usage: python tests/simd.py     (exit 0 = pass)
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
WORK = os.path.join(SCRATCH, "_simd")
PROBE = os.path.join(TESTS, "probe.py")

TOPOLOGIES = ["linear", "mixed", "conv", "memory"]

ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {note}" if note else ""))


r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", os.path.join(TESTS, "build.ps1"), "-OutDir", MODULE_DIR],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", cwd=ROOT)
if r.returncode != 0:
    print(r.stdout); print(r.stderr)
    raise SystemExit("build failed")
print(r.stdout.strip())

sys.path.insert(0, MODULE_DIR)
import ml                                                      # noqa: E402

info = ml.cpu_info()
print(f"\n이 CPU: simd={info['simd']}  avx2={info['avx2']}  avx512={info['avx512']}"
      f"  cores={info['cores']}")

# Only levels this CPU can actually run. Asking for a missing one must fall
# back rather than crash, which is checked separately below.
LEVELS = ["base"]
if info["avx2"]:
    LEVELS += ["sse", "avx2"]
if info["avx512"]:
    LEVELS.append("avx512")
else:
    print("  (이 CPU 는 AVX-512 가 없어서 그 경로는 여기서 검증되지 않습니다)")


def run_probe(mode, topo, name, cwd, simd=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if simd:
        env["MYML_SIMD"] = simd
    if topo.startswith("memory"):
        env["MYML_BPTT"] = "0"
    r = subprocess.run([sys.executable, PROBE, MODULE_DIR, mode, topo, name],
                       cwd=cwd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"probe failed (topo={topo} simd={simd})")
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln and all(c in "0123456789abcdefABCDEF," for c in ln):
            return ln
    return ""


def close_enough(a_hex, b_hex, tol):
    a = [struct.unpack("<d", bytes.fromhex(h))[0] for h in a_hex.split(",")]
    b = [struct.unpack("<d", bytes.fromhex(h))[0] for h in b_hex.split(",")]
    if len(a) != len(b):
        return False, "길이가 다름"
    worst = 0.0
    for x, y in zip(a, b):
        d = abs(x - y) / max(1.0, abs(x), abs(y))
        worst = max(worst, d)
    return worst <= tol, f"최대 상대오차 {worst:.2e}"


# ── 각 SIMD 수준이 같은 답을 내는지 ─────────────────────────────────────
print(f"\n[일치] 같은 가중치에서 {LEVELS} 를 비교 (허용오차 {TOLERANCE})")
shutil.rmtree(WORK, ignore_errors=True)
for topo in TOPOLOGIES:
    name = f"simd_{topo}"
    pth = f"{name}_ml_memory.pth"
    init = os.path.join(WORK, topo, "init")
    os.makedirs(init, exist_ok=True)
    run_probe("init", topo, name, init)
    snap = os.path.join(init, pth)

    outs = {}
    for lv in LEVELS:
        d = os.path.join(WORK, topo, lv)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(snap, os.path.join(d, pth))
        outs[lv] = run_probe("run", topo, name, d, simd=lv)

    worst_lv, worst_note, good = None, "", True
    for lv in LEVELS:
        if lv == "avx2":
            continue
        same, note = close_enough(outs[lv], outs.get("avx2", outs["base"]), TOLERANCE)
        if not same:
            good = False
            worst_lv, worst_note = lv, note
    check(f"{topo}", good,
          worst_note if not good else f"{len(LEVELS)}개 수준 일치")
    if not good:
        print(f"     어긋난 수준: {worst_lv}")

# ── MYML_SIMD 가 실제로 먹는지 / 없는 걸 요구하면 어떻게 되는지 ─────────
print("\n[MYML_SIMD] 내려 쓰기는 되고, 없는 걸 올려 쓰려 하면 되돌아간다")


def simd_of(value):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if value is not None:
        env["MYML_SIMD"] = value
    code = ("import sys; sys.path.insert(0, r'%s'); import ml; "
            "print('SIMD', ml.cpu_info()['simd'])" % MODULE_DIR)
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"cpu_info failed (MYML_SIMD={value})")
    for ln in r.stdout.splitlines():
        if ln.startswith("SIMD "):
            return ln[5:].strip()
    raise SystemExit("cpu_info printed nothing")


for lv in LEVELS:
    check(f"MYML_SIMD={lv} 이 그대로 선택된다", simd_of(lv) == lv, f"-> {simd_of(lv)}")

if not info["avx512"]:
    got = simd_of("avx512")
    check("없는 avx512 를 요구하면 쓸 수 있는 것으로 되돌아간다 (죽지 않는다)",
          got == info["simd"], f"-> {got}")
got = simd_of("헛소리")
check("모르는 값은 무시하고 기본 선택으로 간다", got == info["simd"], f"-> {got}")

shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL SIMD CHECKS PASS" if ok else "SIMD CHECKS FAILED"))
sys.exit(0 if ok else 1)
