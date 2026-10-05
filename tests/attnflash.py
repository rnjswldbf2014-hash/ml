"""Does fattn() compute the same thing as attn()?

That is the only question that matters about the flash path, and it is askable
because the two layers have **identical parameters** -- same LayerNorm, same
Wq/Wk/Wv/Wo. Only the route differs: fattn walks s in blocks keeping a running
max and sum (online softmax), never builds the S*S score matrix, and recomputes
the scores in the backward pass instead of caching them.

So the test is: train one model with attn, save it, flip the one word in the
weight file that says "this attention layer is flash", and load the same bytes
the other way. Any disagreement is a bug in the flash kernels, not a difference
in what was learned.

That word is layC for the attention layer. The header is uint32 little-endian:
    [0] magic  [1] version  [2] optimizer  [3] inputSz  [4] nLayers
    then per layer: kind, A, B, C
so layer i's C sits at word 5 + 4*i + 3. Nothing else in the file changes --
this is the same byte-level trick tests/logic.py uses to make an old-version
file. (The format version is NOT bumped for flash: an older binary reading such
a file ignores C and runs standard attention, which gives the same answer, just
slower. Unlike logic/conv, there is nothing to misread.)

Tolerance, not bit-exact: a running rescale and a single final divide do not
produce the same last bits as "subtract the row max, then normalise by the sum".
1e-4 on the values (and the control below shows what a real disagreement looks
like -- it is not subtle).

What regression.py covers instead: that fattn's own per-sample and batched
paths agree bit-for-bit, and across thread counts (topologies fattn / fattn2 /
fattnmix). fattn2 has 36 items so it spans more than one FBS=32 block, which is
where the rescale bookkeeping actually gets exercised.

Usage: python tests/attnflash.py     (exit 0 = pass)
"""
import os
import random
import shutil
import struct
import subprocess
import sys

TOL = 1e-4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
WORK = os.path.join(SCRATCH, "_flash")

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

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
sys.path.insert(0, MODULE_DIR)
os.chdir(WORK)

from ml import make, cos, attn, fattn, each     # noqa: E402

DIM, S, HEADS, B = 72, 36, 2, 24            # S=36 > FBS=32, so blocks are real
NAME = "fl"
PTH = f"{NAME}_ml_memory.pth"

rnd = random.Random(7)
ROWS = [[rnd.uniform(-1, 1) for _ in range(DIM)] for _ in range(B)]
TGTS = [[rnd.uniform(-1, 1)] for _ in range(B)]
PROBES = [[rnd.uniform(-1, 1) for _ in range(DIM)] for _ in range(6)]


def spec(flash):
    f = fattn if flash else attn
    return [DIM, f(S, HEADS), each(4), 16]


def set_flash_flag(path, layer, value):
    """Rewrite layC for one layer in place."""
    with open(path, "rb") as fh:
        raw = bytearray(fh.read())
    head = struct.unpack_from("<5I", raw, 0)
    assert head[0] == 0xBEEFCAFE, f"magic {head[0]:#x}"
    assert head[1] == 12, f"unexpected version {head[1]}"
    off = 4 * (5 + 4 * layer + 3)
    was = struct.unpack_from("<I", raw, off)[0]
    kind = struct.unpack_from("<I", raw, 4 * (5 + 4 * layer))[0]
    assert kind == 1, f"layer {layer} is kind {kind}, not attention"
    struct.pack_into("<I", raw, off, value)
    with open(path, "wb") as fh:
        fh.write(raw)
    return was


# ── 같은 가중치를 만들어 저장한다 ───────────────────────────────────────
print("\n[준비] attn 으로 학습해서 저장한 뒤, 파일의 플래시 플래그만 뒤집는다")
std_dir = os.path.join(WORK, "std")
fl_dir = os.path.join(WORK, "flash")
ctl_dir = os.path.join(WORK, "control")
for d in (std_dir, fl_dir, ctl_dir):
    os.makedirs(d, exist_ok=True)

os.chdir(std_dir)
seed = make(NAME, spec(False), [cos], lr=0.02)
for _ in range(8):
    seed.sl(ROWS, TGTS)
seed.save(seed.reward(seed.rl(ROWS[0]), 1.0))       # persist
del seed
shutil.copyfile(PTH, os.path.join(fl_dir, PTH))
was = set_flash_flag(os.path.join(fl_dir, PTH), 0, 1)
check("저장된 파일의 어텐션 층 플래그가 0(스탠다드)이었다", was == 0, f"layC={was}")


def load(d, flash):
    os.chdir(d)
    return make(NAME, spec(flash), [cos], lr=0.02)


std = load(std_dir, False)
fl = load(fl_dir, True)
os.chdir(ctl_dir)
ctl = make(NAME, spec(True), [cos], lr=0.02)        # 새 무작위 가중치 (대조군)


def worst(a, b):
    return max(abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def outs(m):
    return [m.predict(x)[0] for x in PROBES]


# ── 순전파 ──────────────────────────────────────────────────────────────
print("\n[순전파] 같은 가중치, 다른 경로")
o_std, o_fl, o_ctl = outs(std), outs(fl), outs(ctl)
check("fattn 과 attn 의 출력이 일치한다", worst(o_std, o_fl) < TOL,
      f"최대 상대오차 {worst(o_std, o_fl):.2e}")
# 검사가 비어있지 않다는 증거: 가중치가 다르면 전혀 안 맞는다.
check("무작위 가중치와는 전혀 안 맞는다 (검사가 비어있지 않다)",
      worst(o_std, o_ctl) > 100 * TOL, f"최대 상대오차 {worst(o_std, o_ctl):.2e}")

# ── 역전파 ──────────────────────────────────────────────────────────────
# 플래시는 점수를 저장하지 않고 역전파에서 다시 계산한다. 한 번 학습시킨 뒤에도
# 같이 움직이는지가 그 재계산(과 D_t = dO·O 항등식)이 맞는지를 본다.
print("\n[역전파] 같은 학습을 한 번 시킨 뒤에도 같이 움직이는가")
for m in (std, fl):
    os.chdir(std_dir if m is std else fl_dir)
    m.sl(ROWS, TGTS)
o_std, o_fl = outs(std), outs(fl)
check("한 스텝 학습 뒤에도 일치한다", worst(o_std, o_fl) < TOL,
      f"최대 상대오차 {worst(o_std, o_fl):.2e}")

print("\n[누적] 20스텝 더 — 오차가 쌓여서 갈라지지 않는가")
for _ in range(20):
    for m in (std, fl):
        m.sl(ROWS, TGTS)
o_std, o_fl = outs(std), outs(fl)
check("20스텝 더 학습한 뒤에도 일치한다", worst(o_std, o_fl) < TOL,
      f"최대 상대오차 {worst(o_std, o_fl):.2e}")

# ── 한 블록 안에 들어가는 작은 S 도 ─────────────────────────────────────
print("\n[작은 S] 블록이 하나뿐일 때 (되스케일이 한 번도 안 일어나는 경로)")
os.chdir(WORK)
small_std, small_fl = os.path.join(WORK, "s_std"), os.path.join(WORK, "s_fl")
os.makedirs(small_std, exist_ok=True); os.makedirs(small_fl, exist_ok=True)
SMALL = [DIM, attn(6, 1), each(4), 16]
SMALLF = [DIM, fattn(6, 1), each(4), 16]
os.chdir(small_std)
m = make("sm", SMALL, [cos], lr=0.02)
for _ in range(8):
    m.sl(ROWS, TGTS)
m.save(m.reward(m.rl(ROWS[0]), 1.0))
del m
shutil.copyfile("sm_ml_memory.pth", os.path.join(small_fl, "sm_ml_memory.pth"))
set_flash_flag(os.path.join(small_fl, "sm_ml_memory.pth"), 0, 1)
os.chdir(small_std); a = make("sm", SMALL, [cos], lr=0.02)
os.chdir(small_fl);  b = make("sm", SMALLF, [cos], lr=0.02)
check("S=6 (한 블록) 에서도 일치한다",
      worst(outs(a), outs(b)) < TOL, f"최대 상대오차 {worst(outs(a), outs(b)):.2e}")

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL FLASH-ATTENTION CHECKS PASS" if ok else "FLASH-ATTENTION CHECKS FAILED"))
sys.exit(0 if ok else 1)
