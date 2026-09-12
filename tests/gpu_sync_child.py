"""Subprocess worker for the GPU/CPU optimizer-state sync check in tests/gpu.py.

The GPU path leaves Adam's m/v on the device and only pulls them back when the
host actually needs them (a CPU step(), or save()). If a sync point is missed,
the host keeps stale m/v -- training still *looks* fine, and the only visible
symptom is in the saved weight file. So: run a fixed training sequence, save,
and let the parent diff the resulting .pth against a CPU-only run.

Batch size decides the route when MYML_GPU=auto, so alternating batch sizes in
one process alternates GPU and CPU steps -- which is exactly the interleaving
that can strand stale state.

argv: <workdir> <sequence>
  sequence is a string of 'g' (big batch -> GPU) and 'c' (small batch -> CPU)
"""
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK, SEQ = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(HERE, "_scratch", "_module"))
os.chdir(WORK)

import ml
from ml import make, cos

LAYERS = [64, 128, 128]
BIG, SMALL = 128, 8

ai = make("sy", LAYERS, [cos], autosave=0)

rnd = random.Random(99)
for step in SEQ:
    b = BIG if step == "g" else SMALL
    X = [[rnd.gauss(0, 1) for _ in range(LAYERS[0])] for _ in range(b)]
    Y = [[rnd.gauss(0, 1)] for _ in range(b)]
    ai.sl(X, Y)

# Probe BEFORE save(). With weights left on the device, a missing sync hook on
# the forward path shows up here -- predict() would answer from the stale host
# copy while the real weights sit on the GPU. save() would mask it, since it
# syncs on its own.
probe = [ai.predict([0.02 * i - 0.5 for i in range(LAYERS[0])])[0],
         ai.predict([0.6 - 0.03 * i for i in range(LAYERS[0])])[0]]
print("PROBE " + " ".join(struct.pack("<f", float(v)).hex() for v in probe))

ai.save()
print("RUNS %d" % ml.gpu_info()["runs"])
