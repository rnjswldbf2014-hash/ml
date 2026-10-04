"""Subprocess worker for tests/memory.py.

The task cannot be solved without memory: a cue appears at t=0, then the input
is identical ([0,0]) for K steps, and at the end the model must report the cue.
Intermediate steps are trained with answer None, so they contribute no loss --
they only advance the memo.

argv: <layers-spec> <K> <seed> <rounds>
  layers-spec is "plain" (no memory layer) or "mem" (with one)
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

ai = make("m", LAYERS, [ANS], autosave=0, lr=0.02)


def episode(cue, learn):
    ai.forget()
    for t in range(K + 1):
        x = ([1.0, 0.0] if cue == 0 else [0.0, 1.0]) if t == 0 else [0.0, 0.0]
        if t < K:
            # No answer: this step only moves the memo along.
            ai.sl(x, [None]) if learn else ai.predict(x)
        else:
            if learn:
                ai.sl(x, [ANS[cue]])
                return None
            return ai.predict(x)[0]


rnd = random.Random(SEED)
for _ in range(ROUNDS):
    episode(rnd.randrange(2), True)
hit = 0
for _ in range(200):
    c = rnd.randrange(2)
    if episode(c, False) == ANS[c]:
        hit += 1
print(f"ACC {hit / 2.0}")
