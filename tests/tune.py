"""Tests that lr / decay / temp actually do something, not just plumb through.

These three used to be unreachable: lr was welded to 0.01 inside ml.d with no
Python accessor at all, there was no weight decay, and there was no sampling
temperature. Wiring a knob up is easy; wiring it up so turning it has the
effect its name promises is the part worth testing. So each check compares two
runs that differ only in that one setting, and asserts the direction of the
difference.

  lr     — bigger steps move the weights further per update
  decay  — pulls weights toward zero, so trained weights end up smaller
  temp   — low temperature concentrates sampling on the best action,
           temp=0 removes randomness entirely, high temperature spreads out
  live   — all of them are settable after make(), for schedules

Usage: python tests/tune.py     (exit 0 = pass)
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
WORK = os.path.join(SCRATCH, "_tune")

ok = True


def check(name, cond, note=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {note}" if note else ""))


r = subprocess.run(["powershell", "-File", os.path.join(TESTS, "build.ps1"),
                    "-OutDir", MODULE_DIR],
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

import ml                                      # noqa: E402
from ml import make, cos                       # noqa: E402

ACTS = ["a", "b", "c", "d"]
ROWS = [[0.3, -0.2, 0.5, 0.1], [-0.4, 0.6, -0.1, 0.2],
        [0.1, 0.1, -0.5, -0.3], [0.7, -0.6, 0.2, 0.4]]
TGT = [[0.8], [-0.5], [0.2], [-0.9]]


def wipe():
    for f in os.listdir("."):
        if ".pth" in f:
            os.remove(f)


def mean_abs_w(ai):
    flat = ml._ml_export_weights(ai._h)["hidden.0.weight"]
    return sum(abs(v) for v in flat) / len(flat)


SNAP_NAME = "snap"
SNAP_PTH = SNAP_NAME + "_ml_memory.pth"


def snapshot(layers, outs):
    """One set of starting weights, shared by every run that follows.

    make() randomises a fresh model, so two runs that differ only in a setting
    are otherwise two *different* models -- any difference you measure is
    mostly the random draw. (An earlier version of this file compared two
    independently initialised models and the decay check came out backwards
    for exactly that reason.)
    """
    wipe()
    ai = make(SNAP_NAME, layers, outs, autosave=0)
    ai.save()
    with open(SNAP_PTH, "rb") as f:
        return f.read()


def from_snapshot(blob, layers, outs, **kw):
    for f in os.listdir("."):
        if ".pth" in f:
            os.remove(f)
    with open(SNAP_PTH, "wb") as f:
        f.write(blob)
    return make(SNAP_NAME, layers, outs, autosave=0, **kw)


# ── lr: bigger steps move further ───────────────────────────────────────
print("\n[lr] 학습률이 실제로 걸음 크기를 바꾸는가")
base = make("lr_probe", [4, 12], [cos], autosave=0)
before = ml._ml_export_weights(base._h)["hidden.0.weight"][:]


def moved(lr):
    wipe()
    ai = make("lr_m", [4, 12], [cos], autosave=0, lr=lr)
    start = ml._ml_export_weights(ai._h)["hidden.0.weight"][:]
    ai.sl(ROWS, TGT)            # exactly one update
    end = ml._ml_export_weights(ai._h)["hidden.0.weight"]
    return sum(abs(a - b) for a, b in zip(start, end))


small, big = moved(0.001), moved(0.1)
check("lr=0.1 이 lr=0.001 보다 가중치를 더 많이 움직인다", big > small * 5,
      f"이동량 {small:.3e} -> {big:.3e}")
check("기본값이 0.01 이다", abs(make("lr_d", [4, 4], [cos], autosave=0).lr - 0.01) < 1e-9)

# ── decay: weights end up smaller ───────────────────────────────────────
# Two bars, because how MUCH the weights shrink depends a lot on the starting
# draw. Measured over 80 snapshots (40 on this build, 40 on the build from
# before the optimizer rewrite -- the spread is the same, so it is a property of
# the task, not a regression):
#
#     decay=0.5, 300 steps   ratio on/off  0.67 .. 0.98   (median 0.86)
#     decay=8.0, 300 steps   ratio on/off  0.11 .. 0.27   (median 0.21)
#
# The old single bar was "decay=0.5 must shrink by at least 5%", and the tail of
# that first row crosses it: it failed ~8% of runs on the draw alone. A test
# that fails one run in twelve for no reason is worse than no test. So the
# realistic strength only has to shrink the weights at all, and a strong setting
# carries the tight bar.
print("\n[decay] 가중치 감쇠가 실제로 가중치를 줄이는가")


def after_decay(snap, decay, steps=300):
    ai = from_snapshot(snap, [4, 12], [cos], decay=decay)
    for _ in range(steps):
        ai.sl(ROWS, TGT)
    return mean_abs_w(ai)


snap = snapshot([4, 12], [cos])
off = after_decay(snap, 0.0)
mild = after_decay(snap, 0.5)
hard = after_decay(snap, 8.0)
check("같은 출발점에서, 감쇠를 켜면 가중치가 작아진다", mild < off,
      f"평균 |가중치| {off:.4f} -> {mild:.4f}  (비율 {mild/off:.3f})")
check("세게 걸면 확실히 작아진다 (실측 0.11~0.27)", hard < off * 0.5,
      f"평균 |가중치| {off:.4f} -> {hard:.4f}  (비율 {hard/off:.3f})")
check("세게 걸면 약하게 걸 때보다 더 작아진다", hard < mild,
      f"{mild:.4f} vs {hard:.4f}")
check("기본값은 꺼짐(0)", make("wd_d", [4, 4], [cos], autosave=0).decay == 0.0)

# ── temp: concentrates or spreads the sampling ──────────────────────────
print("\n[temp] 온도가 고르기의 과감함을 바꾸는가")
wipe()
# Train only lightly. Train it hard and the logits saturate: the favourite wins
# ~100% of the time at temp 1 already, leaving nothing for a lower temperature
# to concentrate -- which is how an earlier version of this check failed.
pick = make("tp", [4, 12], [ACTS], autosave=0, lr=0.02)
for _ in range(25):
    pick.sl(ROWS, [["a"], ["a"], ["a"], ["a"]])
probe = ROWS[0]
best = pick.predict(probe)[0]


def share(t, n=4000):
    pick.temp = t
    hits = sum(1 for _ in range(n) if pick.rl(probe).output[0] == best)
    return hits / n


hot, warm, cold, zero = share(5.0), share(1.0), share(0.2), share(0.0)
check("온도를 낮출수록 최선을 더 자주 고른다", cold >= warm >= hot and cold - hot > 0.1,
      f"temp 5.0={hot:.2f}  1.0={warm:.2f}  0.2={cold:.2f}")
check("temp=0 이면 무작위성이 사라진다", zero == 1.0, f"{zero:.3f}")
check("predict() 는 온도에 영향받지 않는다", pick.predict(probe)[0] == best)

# ── settable after make() ───────────────────────────────────────────────
print("\n[live] 만든 뒤에도 바꿀 수 있는가 (스케줄용)")
wipe()
s = make("sch", [4, 8], [cos], autosave=0, lr=0.02, decay=0.001, temp=0.5)
check("make() 로 준 값이 그대로 읽힌다",
      abs(s.lr - 0.02) < 1e-9 and abs(s.decay - 0.001) < 1e-9
      and abs(s.temp - 0.5) < 1e-9,
      f"lr={s.lr} decay={s.decay} temp={s.temp}")
s.lr, s.decay, s.temp, s.sigma, s.entropy = 0.003, 0.02, 2.0, 0.5, 0.05
check("쓰고 다시 읽으면 바뀐 값이 나온다",
      abs(s.lr - 0.003) < 1e-9 and abs(s.decay - 0.02) < 1e-9
      and abs(s.temp - 2.0) < 1e-9 and abs(s.sigma - 0.5) < 1e-9
      and abs(s.entropy - 0.05) < 1e-9)
try:
    ml._ml_tune(s._h, "없는값", 1.0)
    check("모르는 설정값은 거부한다", False, "통과해버림")
except Exception:
    check("모르는 설정값은 거부한다", True)

os.chdir(ROOT)
shutil.rmtree(WORK, ignore_errors=True)
print("\n" + ("ALL TUNE CHECKS PASS" if ok else "TUNE CHECKS FAILED"))
sys.exit(0 if ok else 1)
