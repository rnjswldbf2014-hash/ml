"""Subprocess worker for the BPTT checks in tests/memory.py.

Runs a fixed, RNG-free training sequence from an existing weight snapshot and
prints the resulting predictions as exact hex, so two runs under different
MYML_BPTT settings can be compared bit-for-bit.

argv: <module-dir> <mode>
  mode=init     create the model and persist a snapshot to compare from
  mode=singles  train with one sl() call per step  (B=1: no chain to walk)
  mode=bundle   train with the whole sequence in one sl() call (B=6: a chain)
"""
import os
import struct
import sys

MOD_DIR, MODE = sys.argv[1], sys.argv[2]
sys.path.insert(0, MOD_DIR)

import ml                                    # noqa: E402
from ml import make, memory, cos             # noqa: E402

NAME = "bp"
XS = [[0.3, -0.2, 0.5], [-0.1, 0.4, 0.2], [0.2, 0.2, -0.3],
      [-0.4, 0.1, 0.1], [0.5, -0.3, 0.4], [0.1, 0.0, -0.2]]
YS = [[0.4], [-0.3], [0.2], [0.1], [-0.5], [0.3]]

ai = make(NAME, [3, memory(4), 5], [cos], lr=0.05)

if MODE == "init":
    ai.save(ai.reward(ai.rl(XS[0]), 1.0))    # persists NAME_ml_memory.pth
    raise SystemExit(0)

for _ in range(20):
    with ai.round():
        if MODE == "bundle":
            ai.sl(XS, YS)
        else:
            for x, y in zip(XS, YS):
                ai.sl(x, y)

with ai.round():
    outs = [ai.predict(x)[0] for x in XS]
print(",".join(struct.pack("<d", float(v)).hex() for v in outs))
