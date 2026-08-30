"""
Single-process probe used by tests/regression.py.

mode=init : create a fresh model, run one rl()/reward()/save() cycle
            (forces a persisted <name>_ml_memory.pth so later 'run'
            invocations across separate processes/env-var configs can
            all start from byte-identical weights), then exit.
mode=run  : load the existing <name>_ml_memory.pth (created by 'init'),
            run a fixed, RNG-free sl() training sequence, then print
            bit-exact (struct-packed hex) predict() outputs on a fixed
            probe set — one line, comma-separated.
"""
import sys
import struct

mod_dir, mode, topo, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, mod_dir)
import ml  # noqa: E402


def topo_layers(topo):
    # each() always needs a preceding attn()/token layer to know the item
    # count ("1번째 층 each: 앞에 tok 이나 attn 이 있어야 항목 수를 압니다"),
    # so an "each"-only topology still includes one attn() to establish it.
    if topo == "linear":
        return [4, 16, 8]
    if topo == "attn":
        return [4, ml.attn(4), 8]
    if topo == "each":
        return [4, ml.attn(4), ml.each(8), 8]
    if topo == "mixed":
        return [4, ml.attn(4), ml.each(8), ml.attn(4), ml.each(4), 8]
    raise ValueError(f"unknown topology {topo!r}")


layers = topo_layers(topo)
ai = ml.make(name, layers, [ml.cos])

if mode == "init":
    step = ai.rl([0.11, 0.22, 0.33, 0.44])
    scored = ai.reward(step, [0.5])
    ai.save(scored)
    sys.exit(0)

if mode == "run":
    B = 6
    xs = [[0.02 * i, 0.03 * i, -0.02 * i, 0.01 * i] for i in range(1, B + 1)]
    ys = [[0.04 * i] for i in range(1, B + 1)]

    # bundle sl() rounds (exercises the batched/threaded path)
    for _ in range(2):
        ai.sl(xs, ys)
    # single-sample sl() rounds (exercises the non-batched path)
    for i in range(2):
        ai.sl([xs[i]], [ys[i]])

    probe_inputs = [[0.01 * i, -0.005 * i, 0.02 * i, 0.008 * i] for i in range(1, 9)]
    hexes = []
    for p in probe_inputs:
        out = ai.predict(p)
        for v in out:
            hexes.append(struct.pack("<d", float(v)).hex())
    print(",".join(hexes))
    sys.exit(0)

raise ValueError(f"unknown mode {mode!r}")
