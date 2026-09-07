"""Subprocess worker for tests/jepa.py.

Runs one jepa training scenario in a fresh process (MYML_* env vars are read
once at module load, so each config needs its own process) and prints a
result line the parent parses.

argv: <workdir> <topology> <mode>
  mode=snapshot : create the models, save untrained weights, exit
  mode=det      : train from the snapshot, print HEX of encode()+imagine()
  mode=world    : run the ring-world scenario, print WORLD <metrics>
"""
import math
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK, TOPO, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(HERE, "_scratch", "_module"))
os.chdir(WORK)

import ml as my_ml
from ml import make, vec, attn, each, jepa

ENC_LAYERS = {
    "linear": [12, 32],
    "attn":   [12, attn(4), 32],
    "each":   [12, attn(4), each(8), 32],
}


def hexs(vals):
    return " ".join(struct.pack("<f", float(v)).hex() for v in vals)


# ── determinism modes ────────────────────────────────────────────────────
if MODE in ("snapshot", "det"):
    e = make("d_enc", ENC_LAYERS[TOPO], [vec(6)])
    p = make("d_prd", [6 + 2, 32], [vec(6)])
    w = jepa(e, p)

    if MODE == "snapshot":
        w.save()          # untrained — the common starting point for every config
        sys.exit(0)

    rnd = random.Random(1234)
    for _ in range(40):
        B = 48
        xs = [[rnd.gauss(0, 1) for _ in range(12)] for _ in range(B)]
        ys = [[rnd.gauss(0, 1) for _ in range(12)] for _ in range(B)]
        ac = [[rnd.gauss(0, 1) for _ in range(2)] for _ in range(B)]
        w.train(xs, ys, ac)

    x = [0.07 * i - 0.4 for i in range(12)]
    print("HEX " + hexs(w.encode(x) + w.imagine(x, [0.3, -0.7])))
    sys.exit(0)

# ── world-model mode ─────────────────────────────────────────────────────
# Hidden world: 6 cells on a ring, moved left/right.
# Observation = that cell's fixed 24-dim vector + fresh noise every time.
# The noise is unpredictable by construction, so a pixel-style model would
# waste capacity on it; JEPA only has to match the *summary*, so dropping the
# noise and keeping the cell identity is the winning strategy. We check it does.
N, OBS, SUM, NOISE = 6, 24, 8, 0.6
random.seed(7)
BASE = [[random.gauss(0, 1) for _ in range(OBS)] for _ in range(N)]

VAR, COV = (0.0, 0.0) if TOPO == "nocollapseguard" else (25.0, 1.0)


def obs(p, rnd):    return [BASE[p][i] + rnd.gauss(0, NOISE) for i in range(OBS)]
def onehot(a):      return [1.0, 0.0] if a == 0 else [0.0, 1.0]
def move(p, a):     return (p + (1 if a == 0 else -1)) % N
def dist(u, v):     return math.sqrt(sum((x - y) ** 2 for x, y in zip(u, v)))


def centroids(w, rnd, M=32):
    out = []
    for q in range(N):
        vs = [w.encode(obs(q, rnd)) for _ in range(M)]
        out.append([sum(v[i] for v in vs) / M for i in range(SUM)])
    return out


e = make("w_enc", [OBS, 64], [vec(SUM)])
p = make("w_prd", [SUM + 2, 64], [vec(SUM)])
w = jepa(e, p, var=VAR, cov=COV)

rnd = random.Random(11)
for i in range(400):
    xs, ys, acts = [], [], []
    for _ in range(64):
        q = rnd.randrange(N); a = rnd.randrange(2)
        xs.append(obs(q, rnd)); ys.append(obs(move(q, a), rnd)); acts.append(onehot(a))
    loss = w.train(xs, ys, acts)

rnd = random.Random(31)
cen = centroids(w, rnd)

# spread of the summaries: near zero == collapsed
mean = [sum(c[i] for c in cen) / N for i in range(SUM)]
spread = math.sqrt(sum((c[i] - mean[i]) ** 2 for c in cen for i in range(SUM)) / (N * SUM))

# can we tell which cell an observation came from, through the noise?
hit = 0
for _ in range(300):
    q = rnd.randrange(N)
    s = w.encode(obs(q, rnd))
    if min(range(N), key=lambda t: dist(s, cen[t])) == q:
        hit += 1
ident = hit / 300

# does imagine(obs, action) land on the actual next cell?
hit = 0
for _ in range(200):
    q = rnd.randrange(N); a = rnd.randrange(2)
    pr = w.imagine(obs(q, rnd), onehot(a))
    if min(range(N), key=lambda t: dist(pr, cen[t])) == move(q, a):
        hit += 1
nxt = hit / 200

print(f"WORLD loss={loss:.6f} spread={spread:.6f} ident={ident:.4f} next={nxt:.4f}")
