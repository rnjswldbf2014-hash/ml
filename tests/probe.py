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
    if topo == "logic":
        return [4, 16, ml.logic(8), 8]
    # two logic layers back to back: the second must NOT squash its input
    # again (it is already 0..1), so this covers that branch too
    if topo == "logic2":
        return [4, ml.logic(8), ml.logic(8), 8]
    if topo == "logicmix":
        return [4, ml.attn(4), ml.each(8), ml.logic(12), 8]
    # conv has two code paths written separately (per-sample fwd/bwd and
    # fwdBatch/bwdBatch, each with its own im2col/col2im loop). Them agreeing
    # bit-for-bit is the main correctness check on that index math.
    if topo == "conv":
        return [4, ml.conv(5, 3, 4), 8]
    if topo == "conv2":                       # stacked, and an even window
        return [4, ml.conv(3, 3, 4), ml.conv(2, 2), 8]
    if topo == "convmix":                     # conv keeps the item structure
        return [4, ml.conv(4, 3, 4), ml.each(6), ml.attn(4), 8]
    # memory carries state across calls, so its batched path walks samples in
    # order while the layers around it stay batched. Batched and serial must
    # still land in the same place -- that equivalence is the whole reason the
    # batched path is allowed to exist for a recurrent layer.
    if topo == "memory":
        return [4, ml.memory(6), 8]
    if topo == "memorymix":
        return [4, 8, ml.memory(6), ml.logic(5), 8]
    # fattn computes the same function as attn by a different route (online
    # softmax over blocks, no S*S matrix, scores recomputed in the backward
    # pass). It has its own per-sample and batched implementations, so them
    # agreeing bit-for-bit is the check on the block/rescale bookkeeping.
    # Whether it agrees with attn is a separate question -- tests/attnflash.py.
    if topo == "fattn":
        return [4, ml.fattn(4), 8]
    if topo == "fattn2":                      # more items than one block (FBS=32)
        return [72, ml.fattn(36), 8]
    if topo == "fattnmix":
        return [4, ml.fattn(4), ml.each(8), ml.fattn(4), ml.each(4), 8]
    # lattn is a different function (no softmax), so the only oracle here is
    # itself: per-sample and batched paths call the same per-head kernel, and
    # must agree bit-for-bit. lattn2 has 2 heads and more items than head width,
    # which is the regime it is meant for.
    if topo == "lattn":
        return [4, ml.lattn(4), 8]
    if topo == "lattn2":
        return [48, ml.lattn(12, 2), 8]
    if topo == "lattnmix":
        return [8, ml.lattn(4, 2), ml.each(6), ml.attn(4), ml.each(4), 8]
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
