"""Subprocess worker for tests/memory.py.

The task cannot be solved without memory: a cue appears at t=0, then the input
is identical ([0,0]) for K steps, and at the end the model must report the cue.
Intermediate steps are trained with answer None, so they contribute no loss --
they only advance the memo.

argv: <layers-spec> <K> <seed> <rounds>
  layers-spec is "plain"  (no memory layer)
                "mem"    (memory layer, sl() called one step at a time)
                "bundle" (memory layer, the whole episode handed over as one
                          bundle -- which is what lets BPTT run through it)
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_scratch", "_module"))
WORK = os.path.join(HERE, "_scratch", "_mem")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
for f in os.listdir("."):
    if ".pth" in f:
        os.remove(f)

from ml import make, memory          # noqa: E402

SPEC, K, SEED, ROUNDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ANS = ["A", "B"]
LAYERS = [2, 16] if SPEC == "plain" else [2, memory(16), 16]
CUE = lambda c: [1.0, 0.0] if c == 0 else [0.0, 1.0]

ai = make("m", LAYERS, [ANS], autosave=0, lr=0.02)


def steps(cue):
    """The episode as plain data: K+1 inputs, an answer only on the last one."""
    xs = [CUE(cue)] + [[0.0, 0.0]] * K
    ys = [None] * K + [[ANS[cue]]]
    return xs, ys


def episode(cue, learn):
    xs, ys = steps(cue)
    with ai.round():
        if learn:
            if SPEC == "bundle":
                # One update for the whole chain. The memo runs in order inside
                # the layer, so the bundle IS the sequence -- and the backward
                # pass can walk back down it.
                ai.sl(xs, ys)
            else:
                # One update per step. There is no chain in a single call, so
                # the gradient is truncated to one step no matter what.
                for x, y in zip(xs, ys):
                    ai.sl(x, y)
            return None
        for x in xs[:-1]:
            ai.predict(x)
        return ai.predict(xs[-1])[0]


rnd = random.Random(SEED)
for _ in range(ROUNDS):
    episode(rnd.randrange(2), True)
hit = 0
for _ in range(200):
    c = rnd.randrange(2)
    if episode(c, False) == ANS[c]:
        hit += 1
print(f"ACC {hit / 2.0}")
