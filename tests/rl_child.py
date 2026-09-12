"""Subprocess worker for tests/rl.py.

Runs the RL learning path (save(scored)) from a shared weight snapshot with a
fixed, RNG-free batch, then prints a probe of predict(). rl() itself samples
from an RNG, so the (input, chosen, value, score) tuples are built by hand
instead of being collected from rl() -- otherwise nothing would be comparable
across processes.

argv: <workdir> <topology> <mode:snapshot|run>
"""
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK, TOPO, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(HERE, "_scratch", "_module"))
os.chdir(WORK)

from ml import make, attn, each, cos, Scored

ACTIONS = ["A", "B", "C", "D"]
LAYERS = {
    "linear": [16, 32],
    "attn":   [16, attn(4), 32],
    "each":   [16, attn(4), each(8), 32],
    "mixed":  [16, attn(4), each(8), 24],
}[TOPO]

# autosave=0: this harness controls when weights hit disk, so a run never
# overwrites the snapshot it is supposed to be starting from.
ai = make("rl_m", LAYERS, [ACTIONS, cos], autosave=0)

if MODE == "snapshot":
    ai.save()
    sys.exit(0)

rnd = random.Random(4242)
B = 40
inputs = [[rnd.gauss(0, 1) for _ in range(16)] for _ in range(B)]
chosen = [[rnd.randrange(len(ACTIONS)), 0] for _ in range(B)]
raw    = [[0.0, rnd.gauss(0, 1)] for _ in range(B)]
# every 7th sample leaves the cos head unscored, to exercise the NaN skip path
points = [[rnd.gauss(0, 1), None if i % 7 == 0 else rnd.gauss(0, 1)]
          for i in range(B)]

batch = [Scored(inputs[i], [ACTIONS[chosen[i][0]], raw[i][1]],
                points[i], chosen[i], raw[i]) for i in range(B)]

for _ in range(15):
    ai.save(batch)

out = []
for x in ([0.03 * i - 0.2 for i in range(16)], [0.11 - 0.02 * i for i in range(16)]):
    o = ai.predict(x)
    out.append(float(ACTIONS.index(o[0])))
    out.append(float(o[1]))
print("HEX " + " ".join(struct.pack("<f", float(v)).hex() for v in out))
