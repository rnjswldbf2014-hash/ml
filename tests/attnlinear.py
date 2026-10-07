"""Tests for lattn() -- linear attention (Katharopoulos et al. 2020).

Unlike fattn, lattn is a **different function** from attn: softmax is replaced
by a positive feature map phi(x) = elu(x) + 1, so

    o_t = phi(q_t)^T KV / (phi(q_t) . Z),   KV = sum_s phi(k_s) v_s^T,  Z = sum_s phi(k_s)

KV and Z do not depend on t, so they are built once and shared by every query:
cost is O(S * d^2) instead of O(S^2 * d). It wins when the item count S exceeds
the head width d.

Because it is a different function there is no oracle to compare against the
way tests/attnflash.py compares fattn with attn. What was done instead, once,
with a temporary probe that has since been removed (same practice as conv and
memory BPTT): the layer's weights and gradients were read out and recomputed
from scratch in numpy -- forward against predict(), backward against numpy
finite differences -- on four shapes (S<d, S>d, 1/2/3 heads). Both agreed to
~1e-7 (float32 noise) with gradients of order 1, and half the q/k values took
the exp branch of phi, so both branches were exercised.

What stands here permanently:

  1. It actually switches the function. A weight file trained as attn, with
     the one layC word flipped to 2, must give DIFFERENT answers. Otherwise a
     bug that quietly ran softmax under the lattn name would pass every
     learning check below.
  2. It mixes items. A task that cannot be written as sum_items f(item) --
     the product of two global means -- is unlearnable without attention, and
     learnable with lattn.
  3. Save/load round-trips (layC = 2 survives), and the file says ver 13.
  4. At many items it is faster than attn (loose bound -- measured 4.4x).

tests/regression.py covers per-sample vs batched bit-identity (lattn, lattn2,
lattnmix). Both paths call the same per-head kernel, by construction.

Usage: python tests/attnlinear.py     (exit 0 = pass)
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
SCRATCH = os.path.join(TESTS, "_scratch")
MODULE_DIR = os.path.join(SCRATCH, "_module")
WORK = os.path.join(SCRATCH, "_lattn")

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

import numpy as np                                          # noqa: E402
from ml import make, cos, attn, lattn, each                 # noqa: E402


def wipe(d="."):
    for f in os.listdir(d):
        if ".pth" in f:
            os.remove(os.path.join(d, f))


def file_head(path):
    with open(path, "rb") as fh:
        return struct.unpack("<5I", fh.read(20))


def set_layc(path, layer, value):
    with open(path, "rb") as fh:
        raw = bytearray(fh.read())
    off = 4 * (5 + 4 * layer + 3)
    assert struct.unpack_from("<I", raw, 4 * (5 + 4 * layer))[0] == 1, "not attention"
    struct.pack_into("<I", raw, off, value)
    with open(path, "wb") as fh:
        fh.write(raw)


# ── 1. 정말로 다른 함수로 바뀌는가 ────────────────────────────────────
print("\n[전환] layC=2 가 실제로 리니어로 바꾸는가")
D, S = 48, 12
rnd = random.Random(5)
ROWS = [[rnd.uniform(-1, 1) for _ in range(D)] for _ in range(16)]
TGT = [[rnd.uniform(-1, 1)] for _ in range(16)]
a_dir, l_dir = os.path.join(WORK, "a"), os.path.join(WORK, "l")
os.makedirs(a_dir); os.makedirs(l_dir)
os.chdir(a_dir)
m = make("sw", [D, attn(S, 2), each(4)], [cos], lr=0.02)
for _ in range(20):
    m.sl(ROWS, TGT)
m.save()
del m
pth = "sw_ml_memory.pth"
shutil.copyfile(pth, os.path.join(l_dir, pth))
set_layc(os.path.join(l_dir, pth), 0, 2)
a = make("sw", [D, attn(S, 2), each(4)], [cos], lr=0.02)
os.chdir(l_dir)
l = make("sw", [D, lattn(S, 2), each(4)], [cos], lr=0.02)
pa = [a.predict(x)[0] for x in ROWS[:6]]
pl = [l.predict(x)[0] for x in ROWS[:6]]
diff = max(abs(x - y) for x, y in zip(pa, pl))
check("같은 가중치로 attn 과 다른 답을 낸다 (softmax 를 조용히 돌리지 않는다)",
      diff > 1e-3, f"최대 차이 {diff:.4f}")
os.chdir(WORK)

# ── 2. 항목을 섞는가 ──────────────────────────────────────────────────
# 뒤는 each(항목마다 따로) + 마지막 선형 합이라, 어텐션이 없으면 모델은
# sum_items f(item) 꼴만 표현할 수 있다. '전체 평균 두 개의 곱' 은 그 꼴이 아니다.
print("\n[섞기] 어텐션 없이는 못 배우는 과제")
IT, W = 8, 4
nrng = np.random.default_rng(0)


def task(n):
    X = nrng.uniform(-1, 1, (n, IT, W)).astype(np.float32)
    y = X[:, :, 0].mean(1) * X[:, :, 1].mean(1) * 4
    return X.reshape(n, -1), y.reshape(n, 1)


Xtr, ytr = task(512)
Xte, yte = task(256)
base = float(((yte - yte.mean()) ** 2).mean())


def rel_err(spec, tag):
    wipe()
    ai = make(tag, spec, [cos], autosave=0, lr=0.005)
    for _ in range(600):
        i = nrng.permutation(512)[:64]
        ai.sl(Xtr[i].tolist(), ytr[i].tolist())
    p = np.array([ai.predict(x)[0] for x in Xte.tolist()])
    return float(((p - yte[:, 0]) ** 2).mean()) / base


none = rel_err([IT * W, each(16, IT), each(8)], "mixn")
lin = rel_err([IT * W, lattn(IT, 1), each(16), lattn(IT, 1), each(8)], "mixl")
check("어텐션 없으면 못 배운다 (평균만 찍는 수준 이상)", none > 0.7, f"{none:.3f}")
check("lattn 은 배운다 (실측 0.08 근처)", lin < 0.3, f"{lin:.3f}")

# ── 3. 저장/불러오기 ──────────────────────────────────────────────────
print("\n[저장] layC=2 가 살아남는가")
wipe()
m = make("rt", [D, lattn(S, 2), each(4)], [cos], lr=0.02)
for _ in range(10):
    m.sl(ROWS, TGT)
before = [m.predict(x)[0] for x in ROWS[:5]]
m.save()
head = file_head("rt_ml_memory.pth")
check("파일 버전이 13 이다", head[1] == 13, f"ver {head[1]}")
with open("rt_ml_memory.pth", "rb") as fh:
    raw = fh.read()
check("파일의 어텐션 층 표시가 2 다",
      struct.unpack_from("<I", raw, 4 * (5 + 3))[0] == 2)
del m
m2 = make("rt", [D, lattn(S, 2), each(4)], [cos], lr=0.02)
after = [m2.predict(x)[0] for x in ROWS[:5]]
check("불러온 뒤 같은 답", before == after, f"{before[0]:.6f} vs {after[0]:.6f}")

# ── 4. 많은 항목에서 빠른가 ──────────────────────────────────────────
# 시간 측정이라 여유를 크게 둔다 (실측 4.4배인데 1.4배만 요구한다).
print("\n[속도] 조각이 많을 때")
BD, BS, BB = 1024, 256, 32
brng = random.Random(1)
BX = [[brng.uniform(-1, 1) for _ in range(BD)] for _ in range(BB)]
BY = [[brng.uniform(-1, 1)] for _ in range(BB)]


def per_sl(spec, tag):
    wipe()
    ai = make(tag, spec, [cos], autosave=0)
    for _ in range(2):
        ai.sl(BX, BY)
    t0 = time.perf_counter()
    for _ in range(6):
        ai.sl(BX, BY)
    return (time.perf_counter() - t0) / 6 * 1000


ta = per_sl([BD, attn(BS), each(8)], "spa")
tl = per_sl([BD, lattn(BS), each(8)], "spl")
check(f"S={BS} (헤드폭 {BD // BS}) 에서 attn 보다 빠르다", tl * 1.4 < ta,
      f"attn {ta:.1f} ms vs lattn {tl:.1f} ms ({ta / tl:.1f}배)")

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL LINEAR-ATTENTION CHECKS PASS" if ok else "LINEAR-ATTENTION CHECKS FAILED"))
sys.exit(0 if ok else 1)
