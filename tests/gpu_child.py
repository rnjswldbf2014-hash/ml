"""Subprocess worker for tests/gpu.py.

argv: <layers,comma-separated> <batch> <mode>
  mode=snapshot : create the model, save untrained weights, exit
  mode=probe    : one sl() step from the snapshot, print predict() as HEX
  mode=info     : print the gpu_info() dict fields this harness cares about

MYML_GPU / MYML_THREADS come from the environment (read once at module load,
so every config needs its own process).
"""
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_scratch", "_module"))
WORK = sys.argv[4] if len(sys.argv) > 4 else os.getcwd()
os.chdir(WORK)

import ml
from ml import make, cos

LAYERS = [int(x) for x in sys.argv[1].split(",")]
B, MODE = int(sys.argv[2]), sys.argv[3]

# The GPU path only covers pure-Linear + one cos head; anything else falls back
# to the CPU batched path, so this is the shape that actually exercises it.
ai = make("gp", LAYERS, [cos], autosave=0)

if MODE == "snapshot":
    ai.save()
    sys.exit(0)

if MODE == "info":
    i = ml.gpu_info()
    print(f"INFO available={i['available']} mode={i['mode']} device={i['device']}")
    sys.exit(0)

rnd = random.Random(7)
X = [[rnd.gauss(0, 1) for _ in range(LAYERS[0])] for _ in range(B)]
Y = [[rnd.gauss(0, 1)] for _ in range(B)]
ai.sl(X, Y)

probe = [ai.predict([0.01 * i - 0.3 for i in range(LAYERS[0])])[0],
         ai.predict([0.4 - 0.02 * i for i in range(LAYERS[0])])[0]]
print("HEX " + " ".join(struct.pack("<f", float(v)).hex() for v in probe))
print("RUNS %d" % ml.gpu_info()["runs"])
