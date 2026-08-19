// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 rnjswldbf2014-hash
module my_ml;

import std.string : fromStringz, toStringz;
import core.memory : GC;

version(Windows) {
    import core.sys.windows.windows;
    import core.runtime : rt_init, rt_term;
    import core.thread : thread_attachThis, thread_detachThis;

    extern(Windows) BOOL DllMain(HINSTANCE h, ULONG reason, LPVOID reserved) nothrow {
        try {
            switch (reason) {
                case DLL_PROCESS_ATTACH: rt_init(); break;
                case DLL_PROCESS_DETACH:
                    // reserved != null 이면 프로세스 종료 중. 이때 DllMain 은 로더 락을
                    // 쥔 채 호출되므로, rt_term() 이 GC 를 정리하며 스레드를 정지/join
                    // 하려다 로더 락과 교착(deadlock)한다. 프로세스가 끝나는 중이면
                    // 메모리는 OS 가 회수하므로 정리를 건너뛴다.
                    if (reserved is null) rt_term();
                    break;
                case DLL_THREAD_ATTACH:  thread_attachThis(); break;
                case DLL_THREAD_DETACH:  thread_detachThis(); break;
                default:
            }
        } catch (Throwable) {}
        return TRUE;
    }
}

import std.stdio     : writefln, File;
import std.math      : exp, sqrt, pow, log, cos, PI;
import std.random    : Random, uniform, uniform01, unpredictableSeed;
import std.file      : exists, remove, rename;
import std.algorithm : countUntil, min;
import std.conv      : to;

private Random rng;
static this() { rng = Random(unpredictableSeed); }

// ── CPU Dispatch ─────────────────────────────────────────────────────────
private import ldc.attributes : target;

private alias DotFn   = float function(const(float)[], const(float)[]) pure nothrow @nogc;
private alias SaxpyFn = void  function(float[], const(float)[], float)  pure nothrow @nogc;

private __gshared DotFn   _dot;
private __gshared SaxpyFn _saxpy;

@target("avx2,fma") private float dot_avx2 (const(float)[] a, const(float)[] b) pure nothrow @nogc
{ float s = 0f; foreach (i; 0..a.length) s += a[i]*b[i]; return s; }
@target("sse4.1")   private float dot_sse41(const(float)[] a, const(float)[] b) pure nothrow @nogc
{ float s = 0f; foreach (i; 0..a.length) s += a[i]*b[i]; return s; }
                    private float dot_base (const(float)[] a, const(float)[] b) pure nothrow @nogc
{ float s = 0f; foreach (i; 0..a.length) s += a[i]*b[i]; return s; }

@target("avx2,fma") private void saxpy_avx2 (float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }
@target("sse4.1")   private void saxpy_sse41(float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }
                    private void saxpy_base (float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }

shared static this() {
    import cpu = core.cpuid;
    if (cpu.avx2)       { _dot = &dot_avx2;  _saxpy = &saxpy_avx2;  }
    else if (cpu.sse42) { _dot = &dot_sse41; _saxpy = &saxpy_sse41; }
    else                { _dot = &dot_base;  _saxpy = &saxpy_base;  }
}

// ─────────────────────────────────────────────
// Optimizer
// ─────────────────────────────────────────────
private enum Opt { adam, sgd, rmsprop, adagrad }

private Opt parseOpt(string s) {
    switch (s) {
        case "sgd":     return Opt.sgd;
        case "rmsprop": return Opt.rmsprop;
        case "adagrad": return Opt.adagrad;
        default:        return Opt.adam;
    }
}

private string optToStr(Opt o) pure nothrow {
    final switch (o) {
        case Opt.adam:    return "adam";
        case Opt.sgd:     return "sgd";
        case Opt.rmsprop: return "rmsprop";
        case Opt.adagrad: return "adagrad";
    }
}

// ─────────────────────────────────────────────
// Linear layer
// ─────────────────────────────────────────────
private struct Linear {
    int inSz, outSz;
    float[][] w, mW, vW, gradW;
    float[]   b, mB, vB, gradB;
    int t;

    this(int i, int o) {
        inSz = i; outSz = o;
        w     = new float[][](o, i);
        mW    = new float[][](o, i);
        vW    = new float[][](o, i);
        gradW = new float[][](o, i);
        b     = new float[o]; mB = new float[o]; vB = new float[o]; gradB = new float[o];
        foreach (j; 0..o) {
            b[j] = mB[j] = vB[j] = gradB[j] = 0f;
            foreach (k; 0..i) mW[j][k] = vW[j][k] = gradW[j][k] = 0f;
        }
        float s = sqrt(2.0f / (i + o));
        foreach (j; 0..o) foreach (k; 0..i) w[j][k] = uniform(-s, s, rng);
    }

    void forward(const(float)[] x, float[] out_) nothrow @nogc {
        foreach (j; 0..outSz) out_[j] = b[j] + _dot(w[j], x);
    }

    void accum(const(float)[] x, const(float)[] dOut, float[] dInBuf) nothrow @nogc {
        foreach (j; 0..outSz) {
            _saxpy(dInBuf,   w[j],    dOut[j]);
            _saxpy(gradW[j], x,       dOut[j]);
            gradB[j] += dOut[j];
        }
    }

    void zeroGrad() nothrow @nogc {
        foreach (j; 0..outSz) gradW[j][] = 0f;
        gradB[] = 0f;
    }

    void step(Opt opt, float lr) nothrow {
        enum float B1=0.9f, B2=0.999f, EPS=1e-8f, RHO=0.99f;
        final switch (opt) {
            case Opt.adam:
                t++;
                float bc1 = 1f - B1^^t, bc2 = 1f - B2^^t;
                foreach (j; 0..outSz) {
                    foreach (k; 0..inSz) {
                        float g = gradW[j][k];
                        mW[j][k] = B1*mW[j][k] + (1-B1)*g;
                        vW[j][k] = B2*vW[j][k] + (1-B2)*g*g;
                        w[j][k] -= lr * (mW[j][k]/bc1) / (sqrt(vW[j][k]/bc2) + EPS);
                    }
                    float gb = gradB[j];
                    mB[j] = B1*mB[j] + (1-B1)*gb; vB[j] = B2*vB[j] + (1-B2)*gb*gb;
                    b[j] -= lr * (mB[j]/bc1) / (sqrt(vB[j]/bc2) + EPS);
                }
                break;
            case Opt.sgd:
                foreach (j; 0..outSz) {
                    foreach (k; 0..inSz) w[j][k] -= lr * gradW[j][k];
                    b[j] -= lr * gradB[j];
                }
                break;
            case Opt.rmsprop:
                foreach (j; 0..outSz) {
                    foreach (k; 0..inSz) {
                        float g = gradW[j][k];
                        vW[j][k] = RHO*vW[j][k] + (1-RHO)*g*g;
                        w[j][k] -= lr * g / (sqrt(vW[j][k]) + EPS);
                    }
                    float gb = gradB[j];
                    vB[j] = RHO*vB[j] + (1-RHO)*gb*gb;
                    b[j] -= lr * gb / (sqrt(vB[j]) + EPS);
                }
                break;
            case Opt.adagrad:
                foreach (j; 0..outSz) {
                    foreach (k; 0..inSz) {
                        float g = gradW[j][k];
                        vW[j][k] += g*g;
                        w[j][k] -= lr * g / (sqrt(vW[j][k]) + EPS);
                    }
                    float gb = gradB[j];
                    vB[j] += gb*gb;
                    b[j] -= lr * gb / (sqrt(vB[j]) + EPS);
                }
                break;
        }
    }
}

// ─────────────────────────────────────────────
// Network — hot path is @nogc
// ─────────────────────────────────────────────
private class Network {
    Linear[] hidden;
    Linear[] heads;
    int      inputSz;

    float[][] _inp, _pre;
    float[]   _hout;
    float[][] _hd, _dHead;
    float[]   _dA, _dB;
    bool      fwdCached;

    this(int inputSz_, int[] hiddenSizes, int[] headSizes) {
        inputSz = inputSz_;
        int prev = inputSz;
        foreach (sz; hiddenSizes) { hidden ~= Linear(prev, sz); prev = sz; }
        foreach (sz; headSizes)   { heads  ~= Linear(prev, sz); }
        _allocScratch();
    }

    private void _allocScratch() {
        _inp = new float[][hidden.length]; _pre = new float[][hidden.length];
        foreach (i, ref h; hidden) { _inp[i] = new float[h.inSz]; _pre[i] = new float[h.outSz]; }
        int houtSz = hidden.length > 0 ? hidden[$-1].outSz : inputSz;
        _hout = new float[houtSz];
        _hd    = new float[][heads.length]; _dHead = new float[][heads.length];
        foreach (i, ref h; heads) { _hd[i] = new float[h.outSz]; _dHead[i] = new float[h.outSz]; }
        int maxSz = inputSz;
        foreach (ref h; hidden) { if (h.inSz > maxSz) maxSz = h.inSz; if (h.outSz > maxSz) maxSz = h.outSz; }
        _dA = new float[maxSz]; _dB = new float[maxSz];
        fwdCached = false;
    }

    void forward(const(float)[] x) nothrow @nogc {
        if (hidden.length == 0) {
            foreach (k; 0..x.length) _hout[k] = x[k];
        } else {
            foreach (k; 0..x.length) _inp[0][k] = x[k];
            foreach (i, ref h; hidden) {
                h.forward(_inp[i][0..h.inSz], _pre[i]);
                if (i + 1 < hidden.length)
                    foreach (k; 0..h.outSz) _inp[i+1][k] = _pre[i][k] < 0f ? 0f : _pre[i][k];
                else
                    foreach (k; 0..h.outSz) _hout[k] = _pre[i][k] < 0f ? 0f : _pre[i][k];
            }
        }
        foreach (i, ref head; heads) head.forward(_hout, _hd[i]);
        fwdCached = true;
    }

    void backward() nothrow @nogc {
        int houtSz = cast(int)_hout.length;
        _dA[0..houtSz] = 0f;
        foreach (i, ref head; heads) {
            _dB[0..houtSz] = 0f;
            head.accum(_hout, _dHead[i], _dB[0..houtSz]);
            foreach (k; 0..houtSz) _dA[k] += _dB[k];
        }
        for (int i = cast(int)hidden.length-1; i >= 0; i--) {
            int outSz = hidden[i].outSz, inSz = hidden[i].inSz;
            foreach (k; 0..outSz) if (_pre[i][k] <= 0f) _dA[k] = 0f;
            _dB[0..inSz] = 0f;
            hidden[i].accum(_inp[i][0..inSz], _dA[0..outSz], _dB[0..inSz]);
            foreach (k; 0..inSz) _dA[k] = _dB[k];
        }
        fwdCached = false;
    }

    void zeroGrad() nothrow @nogc { foreach (ref h; hidden) h.zeroGrad(); foreach (ref h; heads) h.zeroGrad(); }
    void step(Opt opt, float lr) nothrow { foreach (ref h; hidden) h.step(opt, lr); foreach (ref h; heads) h.step(opt, lr); }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
private void softmaxInPlace(float[] x) nothrow @nogc {
    float mx = x[0];
    foreach (v; x) if (v > mx) mx = v;
    float s = 0f;
    foreach (ref v; x) { v = exp(v - mx); s += v; }
    s = 1f / s;
    foreach (ref v; x) v *= s;
}

// @nogc string comparison (avoids D runtime __equals)
private bool strEq(const(char)[] a, const(char)[] b) pure nothrow @nogc {
    if (a.length != b.length) return false;
    foreach (i; 0..a.length) if (a[i] != b[i]) return false;
    return true;
}

// Write mask indices into outBuf; returns count. No allocation.
private int buildMaskInto(string[] all, string[] legal, int[] outBuf) nothrow @nogc {
    int n = 0;
    foreach (a; legal)
        foreach (j, ac; all)
            if (strEq(ac, a)) { outBuf[n++] = cast(int)j; break; }
    return n;
}

private int[] buildMask(string[] all, string[] legal) {
    int[] idx;
    foreach (a; legal)
        foreach (j, ac; all)
            if (ac == a) { idx ~= cast(int)j; break; }
    return idx;
}

// ─────────────────────────────────────────────
// BlackBoxAI — 순수 파이프라인
//   rl()    : 순전파 + 샘플링. 모델 상태를 바꾸지 않는다.
//   learn() : (입력, 출력, 보상) 묶음을 받아 역전파 + 파일 저장.
// ─────────────────────────────────────────────
class BlackBoxAI {
    string     name;
    string[][] actionLists;   // 내부는 다중 헤드 구조 유지, 공개 API 는 헤드 1개만 사용
    int[]      hiddenSizes;
    float      lr = 0.01f;
    Opt        opt;
    string     file;
    bool       cosMode;       // true 면 숫자(연속값) 출력
    float      cosSigma = 0.1f;
    int        outSz;

    Network net;
    bool    ready;
    private float[] _envBuf;
    private float[] _probBuf;
    private int[]   _maskBuf;

    this(string name, int[] layers, string[] actions, bool cos, Opt opt = Opt.adam) {
        this.name = name; this.opt = opt; this.cosMode = cos;
        file = name ~ "_ml_memory.pth";

        int inputSz = layers[0];
        outSz       = layers[$-1];
        hiddenSizes = layers[1..$-1].dup;
        actionLists = [cos ? cast(string[])[] : actions.dup];

        if (exists(file)) {
            try {
                load();
                // 저장된 구조와 요청한 layers 가 다르면 파일을 따르지 않는다.
                // (조용히 다른 신경망이 만들어지는 것을 막는다)
                bool same = (net.inputSz == inputSz) && (outSz == layers[$-1])
                         && (hiddenSizes.length == layers.length - 2);
                if (same) foreach (i, sz; hiddenSizes) if (sz != layers[1+i]) { same = false; break; }
                if (!same) {
                    int[] savedLayers = net.inputSz ~ hiddenSizes ~ outSz;
                    writefln(" [%s] 저장된 구조 %s 가 요청한 %s 와 다릅니다. 새로 만듭니다.",
                             name, savedLayers, layers);
                    outSz       = layers[$-1];
                    hiddenSizes = layers[1..$-1].dup;
                    actionLists = [cos ? cast(string[])[] : actions.dup];
                    cosMode     = cos;
                    net = null;
                } else {
                    ready = true;
                    writefln(" [%s] 이전 학습 데이터를 불러왔습니다.", name);
                }
            } catch (Exception e) {
                writefln(" [%s] 불러오기 실패 (%s). 새로 시작합니다.", name, e.msg);
                ready = false;
            }
        }
        if (!ready) {
            net = new Network(inputSz, hiddenSizes, [outSz]);
            writefln(" [%s] 새로 생성되었습니다. %s", name, layers);
            ready = true;
        }
        _envBuf  = new float[net.inputSz];
        _probBuf = new float[outSz];
        _maskBuf = new int[outSz];
    }

    // ── 합법 액션 → 인덱스 마스크 ─────────────────────────
    private int maskOf(string[] legal) {
        if (cosMode || legal.length == 0) {
            foreach (k; 0..outSz) _maskBuf[k] = k;
            return outSz;
        }
        return buildMaskInto(actionLists[0], legal, _maskBuf);
    }

    // ── B1 rl: 순수. 선택된 인덱스(이산) 또는 샘플값(cos) ──
    // 모델 가중치를 바꾸지 않는다.
    int pick(string[] legal, float[] input) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        int mlen = maskOf(legal);
        foreach (k; 0..mlen) _probBuf[k] = net._hd[0][_maskBuf[k]];
        softmaxInPlace(_probBuf[0..mlen]);
        double r = uniform01!double(rng), cum = 0.0;
        int c = mlen - 1;
        foreach (k; 0..mlen) { cum += _probBuf[k]; if (r < cum) { c = k; break; } }
        return _maskBuf[c];          // 전체 액션 목록 기준 인덱스
    }

    // cos 모드: 가우시안 정책 — 평균 mu 에서 표본 추출
    float pickCos(float[] input, out int unit) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        unit = 0;
        if (outSz > 1) unit = uniform(0, outSz, rng);
        float mu = net._hd[0][unit];
        float u1 = cast(float)uniform01!double(rng);
        float u2 = cast(float)uniform01!double(rng);
        if (u1 < 1e-9f) u1 = 1e-9f;
        float z = sqrt(-2f * log(u1)) * cos(2f * PI * u2);   // Box-Muller
        return mu + cosSigma * z;
    }

    // ── 예측(학습 없음): argmax 인덱스 / cos 는 평균값 ─────
    int predict(string[] legal, float[] input) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        int mlen = maskOf(legal);
        int best = 0;
        foreach (k; 1..mlen)
            if (net._hd[0][_maskBuf[k]] > net._hd[0][_maskBuf[best]]) best = k;
        return _maskBuf[best];
    }

    float predictCos(float[] input, int unit) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        return net._hd[0][unit < outSz ? unit : 0];
    }

    // ── B7 learn: (입력, 출력, 보상) 묶음으로 역전파 ───────
    // 이산: REINFORCE  d = score * (p - onehot)
    // cos : 가우시안 정책  d = -score * (y - mu) / sigma^2
    void learnBatch(float[][] inputs, int[] chosen, float[] outVals, float[] scores) {
        if (inputs.length == 0) return;
        net.zeroGrad();
        foreach (i; 0..inputs.length) {
            foreach (k; 0..inputs[i].length) _envBuf[k] = inputs[i][k];
            net.forward(_envBuf);
            net._dHead[0][] = 0f;
            if (cosMode) {
                int u = chosen[i] < outSz ? chosen[i] : 0;
                float mu = net._hd[0][u];
                net._dHead[0][u] = -scores[i] * (outVals[i] - mu) / (cosSigma * cosSigma);
            } else {
                foreach (k; 0..outSz) _probBuf[k] = net._hd[0][k];
                softmaxInPlace(_probBuf[0..outSz]);
                foreach (k; 0..outSz)
                    net._dHead[0][k] = scores[i] * (_probBuf[k] - (k == chosen[i] ? 1f : 0f));
            }
            net.backward();
        }
        net.step(opt, lr);
    }

    // ── B4 sl: 정답이 있으면 1스텝 학습 후 예측 ───────────
    int slStep(string[] legal, float[] input, int answerIdx) {
        if (answerIdx >= 0) {
            foreach (k; 0..input.length) _envBuf[k] = input[k];
            net.zeroGrad();
            net.forward(_envBuf);
            int mlen = maskOf(legal);
            foreach (k; 0..mlen) _probBuf[k] = net._hd[0][_maskBuf[k]];
            softmaxInPlace(_probBuf[0..mlen]);
            int tgt = -1;
            foreach (k; 0..mlen) if (_maskBuf[k] == answerIdx) { tgt = k; break; }
            net._dHead[0][] = 0f;
            foreach (k; 0..mlen)
                net._dHead[0][_maskBuf[k]] = _probBuf[k] - (k == tgt ? 1f : 0f);
            net.backward();
            net.step(opt, lr);
        }
        return predict(legal, input);
    }

    // cos 모드 지도학습: 평균제곱오차
    float slCos(float[] input, int unit, float target, bool train) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        if (train) {
            net.zeroGrad();
            net.forward(_envBuf);
            net._dHead[0][] = 0f;
            net._dHead[0][unit] = net._hd[0][unit] - target;
            net.backward();
            net.step(opt, lr);
        }
        net.forward(_envBuf);
        return net._hd[0][unit];
    }

    void save() {
        if (!ready) return;
        auto f = File(file, "wb");
        void wu(uint v)  { f.rawWrite((&v)[0..1]); }
        void wf(float v) { f.rawWrite((&v)[0..1]); }
        wu(0xBEEFCAFE); wu(3); wu(cast(uint)opt);
        wu(cosMode ? 1u : 0u);
        wu(cast(uint)outSz);
        wu(cast(uint)net.inputSz);
        wu(cast(uint)hiddenSizes.length);
        foreach (sz; hiddenSizes) wu(cast(uint)sz);
        wu(cast(uint)actionLists.length);
        foreach (al; actionLists) {
            wu(cast(uint)al.length);
            foreach (a; al) { auto bytes = cast(ubyte[])a; wu(cast(uint)bytes.length); f.rawWrite(bytes); }
        }
        void wl(ref Linear l) {
            foreach (row; l.w)  foreach (v; row) wf(v);
            foreach (v; l.b)    wf(v);
            foreach (row; l.mW) foreach (v; row) wf(v);
            foreach (row; l.vW) foreach (v; row) wf(v);
            foreach (v; l.mB)   wf(v);
            foreach (v; l.vB)   wf(v);
            wu(cast(uint)l.t);
        }
        foreach (ref h; net.hidden) wl(h);
        foreach (ref h; net.heads)  wl(h);
    }

    private void load() {
        auto f = File(file, "rb");
        uint  ru()  { uint v;  f.rawRead((&v)[0..1]); return v; }
        float rf()  { float v; f.rawRead((&v)[0..1]); return v; }
        if (ru() != 0xBEEFCAFE) throw new Exception("magic mismatch");
        uint ver = ru();
        if (ver >= 2) opt = cast(Opt)ru();
        if (ver >= 3) { cosMode = ru() != 0; outSz = ru(); }
        int inputSz = ru();
        int nH = ru(); hiddenSizes = new int[nH];
        foreach (i; 0..nH) hiddenSizes[i] = ru();
        int nL = ru(); actionLists = new string[][nL];
        foreach (i; 0..nL) {
            int nA = ru(); actionLists[i] = new string[nA];
            foreach (j; 0..nA) { auto buf = new ubyte[ru()]; f.rawRead(buf); actionLists[i][j] = cast(string)buf.dup; }
        }
        if (ver < 3) outSz = cast(int)actionLists[0].length;
        net = new Network(inputSz, hiddenSizes, [outSz]);
        void rl_(ref Linear l) {
            foreach (ref row; l.w)  foreach (ref v; row) v = rf();
            foreach (ref v; l.b)    v = rf();
            foreach (ref row; l.mW) foreach (ref v; row) v = rf();
            foreach (ref row; l.vW) foreach (ref v; row) v = rf();
            foreach (ref v; l.mB)   v = rf();
            foreach (ref v; l.vB)   v = rf();
            l.t = ru();
        }
        foreach (ref h; net.hidden) rl_(h);
        foreach (ref h; net.heads)  rl_(h);
    }

}

void resset(string modelName) {
    bool deleted = false;
    foreach (suffix; ["_ml_memory.pth", "_sl_memory.pth", "_auto_memory.pth"]) {
        string path = modelName ~ suffix;
        if (exists(path)) {
            try { remove(path); writefln(" [%s] 초기화: %s", modelName, path); deleted = true; }
            catch (Exception e) { writefln("오류: %s 삭제 실패 (%s)", path, e.msg); }
        }
    }
    if (!deleted) writefln(" [%s] 모델 파일이 존재하지 않습니다.", modelName);
}

private bool isTorchFile(string path) nothrow {
    try {
        auto f = File(path, "rb"); ubyte[2] magic; f.rawRead(magic[]);
        return magic[0] == 0x50 && magic[1] == 0x4B;
    } catch (Exception) { return false; }
}


// ─────────────────────────────────────────────
// change — 예전 포맷 파일을 현재 포맷으로 변환
//   ver 1/2 (다중 헤드 가능) → ver 3 (단일 헤드)
//   원본은 .bak 으로 남긴다.
// ─────────────────────────────────────────────
private int linearBytes(int inSz, int outSz) pure nothrow @nogc {
    // w + mW + vW = 3*out*in,  b + mB + vB + gradB 중 저장분은 b,mB,vB = 3*out
    // 실제 저장 순서: w, b, mW, vW, mB, vB, t
    return cast(int)((3L*outSz*inSz + 3L*outSz) * 4 + 4);
}

string changeFile(string path) {
    if (!exists(path)) throw new Exception(path ~ " 파일이 없습니다");
    if (isTorchFile(path))
        throw new Exception("PyTorch(zip) 포맷입니다. 이 함수는 my_ml 자체 포맷만 변환합니다");

    Opt        o = Opt.adam;
    int        inputSz, nH, nL;
    int[]      hid;
    string[][] als;
    ubyte[]    hiddenBlob, headBlob;
    int        droppedHeads;

    {
        auto f = File(path, "rb");
        uint ru() { uint v; f.rawRead((&v)[0..1]); return v; }
        if (ru() != 0xBEEFCAFE) throw new Exception("my_ml 포맷이 아닙니다");
        uint ver = ru();
        if (ver >= 3) return "already";
        if (ver >= 2) o = cast(Opt) ru();
        inputSz = ru();
        nH  = ru(); hid = new int[nH];
        foreach (i; 0..nH) hid[i] = ru();
        nL  = ru(); als = new string[][nL];
        foreach (i; 0..nL) {
            int nA = ru(); als[i] = new string[nA];
            foreach (j; 0..nA) { auto b = new ubyte[ru()]; f.rawRead(b); als[i][j] = cast(string) b.dup; }
        }
        if (nL == 0 || als[0].length == 0) throw new Exception("액션 목록이 비어 있습니다");

        // 은닉층 가중치 통째로
        int prev = inputSz, hbytes = 0;
        foreach (sz; hid) { hbytes += linearBytes(prev, sz); prev = sz; }
        hiddenBlob = new ubyte[hbytes];
        if (hbytes > 0) f.rawRead(hiddenBlob);

        // 헤드 0 만 남기고 나머지는 버린다
        headBlob = new ubyte[linearBytes(prev, cast(int) als[0].length)];
        f.rawRead(headBlob);
        droppedHeads = nL - 1;
    }

    // 원본 백업
    string bak = path ~ ".bak";
    if (exists(bak)) remove(bak);
    rename(path, bak);

    auto w = File(path, "wb");
    void wu(uint v) { w.rawWrite((&v)[0..1]); }
    wu(0xBEEFCAFE); wu(3); wu(cast(uint) o);
    wu(0u);                                   // cosMode = false
    wu(cast(uint) als[0].length);             // outSz
    wu(cast(uint) inputSz);
    wu(cast(uint) nH);
    foreach (sz; hid) wu(cast(uint) sz);
    wu(1u);                                   // 헤드 1개
    wu(cast(uint) als[0].length);
    foreach (a; als[0]) { auto b = cast(ubyte[]) a; wu(cast(uint) b.length); w.rawWrite(b); }
    if (hiddenBlob.length) w.rawWrite(hiddenBlob);
    w.rawWrite(headBlob);
    w.close();

    int[] layers = inputSz ~ hid ~ cast(int) als[0].length;
    if (droppedHeads > 0)
        return to!string(layers) ~ " (헤드 " ~ to!string(droppedHeads) ~ "개 버림)";
    return to!string(layers);
}

// ─────────────────────────────────────────────
// Python C API declarations
// ─────────────────────────────────────────────
private:
alias Py_ssize_t = long;
struct PyObject { Py_ssize_t ob_refcnt; void* ob_type; }
struct PyModuleDef_Base { PyObject ob_base; void* m_init; Py_ssize_t m_index; void* m_copy; }
struct PyModuleDef {
    PyModuleDef_Base m_base; const(char)* m_name; const(char)* m_doc;
    Py_ssize_t m_size; PyMethodDef* m_methods;
    void* m_slots; void* m_traverse; void* m_clear; void* m_free;
}
alias PyCFunction = extern(C) PyObject* function(PyObject*, PyObject*) nothrow;
alias PyCapsuleDestructor = extern(C) void function(PyObject*) nothrow;
struct PyMethodDef { const(char)* ml_name; PyCFunction ml_meth; int ml_flags; const(char)* ml_doc; }
enum METH_VARARGS = 0x0001, PYTHON_API_VERSION = 1013, Py_file_input = 257, Py_eval_input = 258;

private __gshared PyObject* _pyNone;
private __gshared PyObject* _pyRuntimeError;

private extern(C) nothrow @nogc {
    void        Py_IncRef(PyObject*); void Py_DecRef(PyObject*);
    int         PyArg_ParseTuple(PyObject*, const(char)*, ...);
    void        PyErr_SetString(PyObject*, const(char)*);
    int         PyErr_Occurred();
    PyObject*   PyUnicode_FromString(const(char)*);
    const(char)* PyUnicode_AsUTF8(PyObject*);
    PyObject*   PyList_New(Py_ssize_t);
    int         PyList_SetItem(PyObject*, Py_ssize_t, PyObject*);
    PyObject*   PyList_GetItem(PyObject*, Py_ssize_t);
    Py_ssize_t  PyList_Size(PyObject*);
    double      PyFloat_AsDouble(PyObject*);
    long        PyLong_AsLong(PyObject*);
    int         PyObject_IsTrue(PyObject*);
    PyObject*   PyCapsule_New(void*, const(char)*, PyCapsuleDestructor);
    void*       PyCapsule_GetPointer(PyObject*, const(char)*);
    PyObject*   PyModule_Create2(PyModuleDef*, int);
    int         PyModule_AddObject(PyObject*, const(char)*, PyObject*);
    PyObject*   PyModule_GetDict(PyObject*);
    PyObject*   PyRun_String(const(char)*, int, PyObject*, PyObject*);
    PyObject*   PyDict_GetItemString(PyObject*, const(char)*);
    int         PyDict_SetItemString(PyObject*, const(char)*, PyObject*);
    PyObject*   PyEval_GetBuiltins();
    PyObject*   PyFloat_FromDouble(double);
    PyObject*   PyLong_FromLong(long);
    PyObject*   PyDict_New();
}

private @trusted:

string[] pyStrList(PyObject* lst) {
    string[] r;
    foreach (i; 0..PyList_Size(lst))
        r ~= fromStringz(PyUnicode_AsUTF8(PyList_GetItem(lst, i))).idup;
    return r;
}
string[][] pyLals(PyObject* lals) {
    string[][] r;
    foreach (i; 0..PyList_Size(lals)) r ~= pyStrList(PyList_GetItem(lals, i));
    return r;
}
float[] pyFloatList(PyObject* lst) {
    float[] r;
    foreach (i; 0..PyList_Size(lst)) r ~= cast(float) PyFloat_AsDouble(PyList_GetItem(lst, i));
    return r;
}
int[] pyIntList(PyObject* lst) {
    int[] r;
    foreach (i; 0..PyList_Size(lst)) r ~= cast(int) PyLong_AsLong(PyList_GetItem(lst, i));
    return r;
}
PyObject* toPyList(string[] strs) {
    auto lst = PyList_New(strs.length);
    foreach (i, s; strs) PyList_SetItem(lst, i, PyUnicode_FromString(toStringz(s)));
    return lst;
}

extern(C) void bbai_dtor(PyObject* cap) nothrow @trusted {
    auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
    if (ai) try { GC.removeRoot(cast(void*) ai); } catch (Throwable) {}
}

// ─────────────────────────────────────────────
// Python extension functions
// ─────────────────────────────────────────────
extern(C) nothrow @trusted:

PyObject* py_ml_make(PyObject* self, PyObject* args) {
    try {
        PyObject* nm; PyObject* lay; PyObject* acts; int cosFlag; PyObject* opt;
        if (!PyArg_ParseTuple(args, "OOOiO", &nm, &lay, &acts, &cosFlag, &opt)) return null;
        string name   = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        string optStr = fromStringz(PyUnicode_AsUTF8(opt)).idup;
        int[] layers  = pyIntList(lay);
        if (layers.length < 2) {
            PyErr_SetString(_pyRuntimeError, "my_ml: layers 는 [입력, ..., 출력] 최소 2개가 필요합니다");
            return null;
        }
        string[] actions = cosFlag ? null : pyStrList(acts);
        auto ai = new BlackBoxAI(name, layers, actions, cosFlag != 0, parseOpt(optStr));
        GC.addRoot(cast(void*) ai);
        return PyCapsule_New(cast(void*) ai, "BlackBoxAI", &bbai_dtor);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_make"); return null; }
}

// B1 rl (이산): 선택된 인덱스 반환. 모델 상태 불변.
PyObject* py_ml_pick(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &legal, &inp)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        return PyLong_FromLong(ai.pick(pyStrList(legal), pyFloatList(inp)));
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_pick"); return null; }
}

// B1 rl (cos): (샘플값, 유닛번호) 반환. 모델 상태 불변.
PyObject* py_ml_pick_cos(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OO", &cap, &inp)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        int unit;
        float v = ai.pickCos(pyFloatList(inp), unit);
        auto t = PyList_New(2);
        PyList_SetItem(t, 0, PyFloat_FromDouble(cast(double)v));
        PyList_SetItem(t, 1, PyLong_FromLong(unit));
        return t;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_pick_cos"); return null; }
}

// 예측(학습 없음)
PyObject* py_ml_predict(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &legal, &inp)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        return PyLong_FromLong(ai.predict(pyStrList(legal), pyFloatList(inp)));
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_predict"); return null; }
}

// B7 learn: (입력, 출력, 보상) 묶음으로 역전파 후 파일 저장
PyObject* py_ml_learn(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inps; PyObject* chos; PyObject* vals; PyObject* scrs;
        if (!PyArg_ParseTuple(args, "OOOOO", &cap, &inps, &chos, &vals, &scrs)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        Py_ssize_t n = PyList_Size(inps);
        auto inputs = new float[][n];
        foreach (i; 0..n) inputs[i] = pyFloatList(PyList_GetItem(inps, i));
        auto chosen = pyIntList(chos);
        auto outv   = pyFloatList(vals);
        auto score  = pyFloatList(scrs);
        ai.learnBatch(inputs, chosen, outv, score);
        ai.save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_learn"); return null; }
}

// B4 sl (이산): 정답 인덱스가 >=0 이면 1스텝 학습 후 예측
PyObject* py_ml_sl(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp; int ansIdx;
        if (!PyArg_ParseTuple(args, "OOOi", &cap, &legal, &inp, &ansIdx)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        return PyLong_FromLong(ai.slStep(pyStrList(legal), pyFloatList(inp), ansIdx));
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_sl"); return null; }
}

// B4 sl (cos): 목표값 회귀
PyObject* py_ml_sl_cos(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inp; int unit; double target; int train;
        if (!PyArg_ParseTuple(args, "OOidi", &cap, &inp, &unit, &target, &train)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        return PyFloat_FromDouble(cast(double) ai.slCos(pyFloatList(inp), unit, cast(float)target, train != 0));
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_sl_cos"); return null; }
}

PyObject* py_ml_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_save"); return null; }
}

PyObject* py_ml_change(PyObject* self, PyObject* args) {
    try {
        PyObject* po;
        if (!PyArg_ParseTuple(args, "O", &po)) return null;
        string path = fromStringz(PyUnicode_AsUTF8(po)).idup;
        string r = changeFile(path);
        return PyUnicode_FromString(toStringz(r));
    } catch (Exception e) {
        PyErr_SetString(_pyRuntimeError, toStringz("my_ml: " ~ e.msg)); return null;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_change"); return null; }
}

PyObject* py_ml_resset(PyObject* self, PyObject* args) {
    try {
        PyObject* nm;
        if (!PyArg_ParseTuple(args, "O", &nm)) return null;
        resset(fromStringz(PyUnicode_AsUTF8(nm)).idup);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_resset"); return null; }
}

PyObject* py_ml_gc_disable(PyObject* self, PyObject* args) {
    try { GC.disable(); Py_IncRef(_pyNone); return _pyNone; }
    catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_gc_disable"); return null; }
}

PyObject* py_ml_gc_collect(PyObject* self, PyObject* args) {
    try { GC.enable(); GC.collect(); Py_IncRef(_pyNone); return _pyNone; }
    catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_gc_collect"); return null; }
}

PyObject* py_ml_export_weights(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto d  = PyDict_New();
        if (!ai || !ai.ready) return d;
        auto net = ai.net;
        foreach (i, ref h; net.hidden) {
            auto wflat = PyList_New(h.outSz * h.inSz); Py_ssize_t idx = 0;
            foreach (row; h.w) foreach (v; row) PyList_SetItem(wflat, idx++, PyFloat_FromDouble(v));
            string wkey = "hidden." ~ to!string(2*i) ~ ".weight";
            PyDict_SetItemString(d, toStringz(wkey), wflat); Py_DecRef(wflat);
            auto wshape = PyList_New(2);
            PyList_SetItem(wshape, 0, PyLong_FromLong(h.outSz)); PyList_SetItem(wshape, 1, PyLong_FromLong(h.inSz));
            PyDict_SetItemString(d, toStringz(wkey ~ ".shape"), wshape); Py_DecRef(wshape);
            auto bflat = PyList_New(h.outSz);
            foreach (j; 0..h.outSz) PyList_SetItem(bflat, j, PyFloat_FromDouble(h.b[j]));
            PyDict_SetItemString(d, toStringz("hidden." ~ to!string(2*i) ~ ".bias"), bflat); Py_DecRef(bflat);
        }
        foreach (i, ref h; net.heads) {
            auto wflat = PyList_New(h.outSz * h.inSz); Py_ssize_t idx = 0;
            foreach (row; h.w) foreach (v; row) PyList_SetItem(wflat, idx++, PyFloat_FromDouble(v));
            string wkey = "output_layers." ~ to!string(i) ~ ".weight";
            PyDict_SetItemString(d, toStringz(wkey), wflat); Py_DecRef(wflat);
            auto wshape = PyList_New(2);
            PyList_SetItem(wshape, 0, PyLong_FromLong(h.outSz)); PyList_SetItem(wshape, 1, PyLong_FromLong(h.inSz));
            PyDict_SetItemString(d, toStringz(wkey ~ ".shape"), wshape); Py_DecRef(wshape);
            auto bflat = PyList_New(h.outSz);
            foreach (j; 0..h.outSz) PyList_SetItem(bflat, j, PyFloat_FromDouble(h.b[j]));
            PyDict_SetItemString(d, toStringz("output_layers." ~ to!string(i) ~ ".bias"), bflat); Py_DecRef(bflat);
        }
        auto isz = PyLong_FromLong(net.inputSz); PyDict_SetItemString(d, "input_size", isz); Py_DecRef(isz);
        auto opts = PyUnicode_FromString(toStringz(optToStr(ai.opt))); PyDict_SetItemString(d, "optimizer_name", opts); Py_DecRef(opts);
        return d;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_export_weights"); return null; }
}

PyObject* py_ml_get_meta(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto d  = PyDict_New();
        auto nm = PyUnicode_FromString(toStringz(ai.name)); PyDict_SetItemString(d, "model_name", nm); Py_DecRef(nm);
        auto al_outer = PyList_New(ai.actionLists.length);
        foreach (i, al; ai.actionLists) { auto al_inner = toPyList(al); PyList_SetItem(al_outer, i, al_inner); }
        PyDict_SetItemString(d, "action_lists", al_outer); Py_DecRef(al_outer);
        auto hl = PyList_New(ai.hiddenSizes.length);
        foreach (i, sz; ai.hiddenSizes) PyList_SetItem(hl, i, PyLong_FromLong(sz));
        PyDict_SetItemString(d, "hidden_layers", hl); Py_DecRef(hl);
        auto opts = PyUnicode_FromString(toStringz(optToStr(ai.opt))); PyDict_SetItemString(d, "optimizer_name", opts); Py_DecRef(opts);
        return d;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_get_meta"); return null; }
}

// ─────────────────────────────────────────────
// Embedded Python class
// ─────────────────────────────────────────────
private enum string PY_CLASS_CODE = `
# SPDX-License-Identifier: GPL-2.0-only
import os

COS = "cos"          # make() 에 actions 대신 넘기면 숫자(연속값) 모드


class Step:
    """rl() 이 돌려주는 (입력, 출력) 쌍. 그냥 데이터."""
    __slots__ = ("input", "output", "_unit")

    def __init__(self, inp, out, unit=0):
        self.input, self.output, self._unit = inp, out, unit

    def __iter__(self):        return iter((self.input, self.output))
    def __len__(self):         return 2
    def __getitem__(self, i):  return (self.input, self.output)[i]
    def __eq__(self, o):
        if isinstance(o, Step):  return (self.input, self.output) == (o.input, o.output)
        if isinstance(o, tuple): return (self.input, self.output) == o
        return NotImplemented
    def __repr__(self):
        return f"Step(input={self.input!r}, output={self.output!r})"


class Scored:
    """reward() 가 돌려주는 (입력, 출력, 보상). 그냥 데이터."""
    __slots__ = ("input", "output", "point", "_unit")

    def __init__(self, inp, out, point, unit=0):
        self.input, self.output, self.point, self._unit = inp, out, point, unit

    def __iter__(self):        return iter((self.input, self.output, self.point))
    def __len__(self):         return 3
    def __getitem__(self, i):  return (self.input, self.output, self.point)[i]
    def __eq__(self, o):
        if isinstance(o, Scored): return tuple(self) == tuple(o)
        if isinstance(o, tuple):  return tuple(self) == o
        return NotImplemented
    def __repr__(self):
        return f"Scored(input={self.input!r}, output={self.output!r}, point={self.point!r})"


class BlackBoxAI:
    def __init__(self, h, name, actions, cos, out_size):
        self._h       = h
        self._name    = name
        self._actions = list(actions or [])
        self._cos     = cos
        self._out     = out_size

    # ── B1: 순수. 가중치를 바꾸지 않고 (입력, 출력) 만 만든다 ──
    def rl(self, input_list, legal=None):
        inp = [float(x) for x in input_list]
        if self._cos:
            val, unit = _ml_pick_cos(self._h, inp)
            return Step(inp, val, unit)
        legal_names = list(legal) if legal else self._actions
        idx = _ml_pick(self._h, legal_names, inp)
        return Step(inp, self._actions[idx], idx)

    # ── B2: 순수. (입력, 출력) + 점수 → (입력, 출력, 보상) ──
    def reward(self, data, point):
        return Scored(data.input, data.output, float(point), data._unit)

    # ── B7: 유일하게 모델을 바꾸는 함수. 역전파 + 자동 저장 ──
    def save(self, scored):
        if isinstance(scored, Scored):
            batch = [scored]
        else:
            batch = list(scored)
        if not batch:
            return 0
        inputs, chosen, values, points = [], [], [], []
        for s in batch:
            inputs.append([float(x) for x in s.input])
            points.append(float(s.point))
            if self._cos:
                chosen.append(int(s._unit))
                values.append(float(s.output))
            else:
                out = s.output
                idx = out if isinstance(out, int) else self._actions.index(out)
                chosen.append(int(idx))
                values.append(0.0)
        _ml_learn(self._h, inputs, chosen, values, points)
        return len(batch)

    # ── B4: 지도학습. 입력도 리스트로 ──
    def sl(self, input_list, answer=None, legal=None, unit=0):
        inp = [float(x) for x in input_list]
        if self._cos:
            train = answer is not None
            return _ml_sl_cos(self._h, inp, int(unit), float(answer or 0.0), int(train))
        legal_names = list(legal) if legal else self._actions
        ans_idx = self._actions.index(answer) if answer is not None else -1
        idx = _ml_sl(self._h, legal_names, inp, ans_idx)
        return self._actions[idx]

    # ── 예측 전용(샘플링 없음, 학습 없음) ──
    def predict(self, input_list, legal=None):
        inp = [float(x) for x in input_list]
        if self._cos:
            return _ml_sl_cos(self._h, inp, 0, 0.0, 0)
        legal_names = list(legal) if legal else self._actions
        return self._actions[_ml_predict(self._h, legal_names, inp)]

    # ── B5: 에피소드 = 스텝을 모으는 리스트. 상태 없음 ──
    # with 블록 대신 그냥 파이썬 리스트를 쓰면 되므로,
    # 보상만 한꺼번에 매기는 헬퍼로 남긴다.
    def episode(self, steps, point):
        """[Step, ...] 전체에 같은 보상을 매겨 [Scored, ...] 반환. 순수."""
        return [self.reward(s, point) for s in steps]

    @property
    def actions(self): return list(self._actions)
    @property
    def out_size(self): return self._out


def make(model_name, layers, actions=None, optimizer='adam'):
    """
    model_name : 모델 이름 (가중치 파일명)
    layers     : [입력수, 은닉..., 출력수]  예) [12, 128, 128, 2]
    actions    : 액션 이름 리스트 (출력 인덱스와 1:1). "cos" 면 숫자 반환 모드
    optimizer  : 'adam' | 'sgd' | 'rmsprop' | 'adagrad'
    """
    layers = [int(x) for x in layers]
    if len(layers) < 2:
        raise ValueError("layers 는 [입력, ..., 출력] 최소 2개가 필요합니다")
    out_size = layers[-1]

    cos = False
    if isinstance(actions, str):
        if actions.lower() == COS:
            cos, actions = True, []
        else:
            raise ValueError(f"actions 가 문자열이면 '{COS}' 만 허용됩니다")
    elif actions is None:
        cos, actions = True, []
    else:
        actions = list(actions)
        if len(actions) != out_size:
            raise ValueError(
                f"actions 개수({len(actions)}) 와 출력층 크기({out_size}) 가 다릅니다")

    h = _ml_make(model_name, layers, list(actions), int(cos), optimizer)
    return BlackBoxAI(h, model_name, actions, cos, out_size)


def change(model_name):
    """예전 버전에서 만든 가중치 파일을 지금 포맷으로 바꿉니다.
    원본은 .bak 으로 남습니다. 이미 최신이면 아무것도 하지 않습니다.

        change("MyModel")                  # MyModel_ml_memory.pth
        change("weights/MyModel.pth")      # 경로를 직접 줘도 됩니다
    """
    path = model_name if model_name.endswith(".pth") else f"{model_name}_ml_memory.pth"
    if not os.path.exists(path):
        print(f" [{model_name}] {path} 가 없습니다.")
        return False
    r = _ml_change(path)
    if r == "already":
        print(f" [{model_name}] 이미 최신 포맷입니다.")
        return False
    print(f" [{model_name}] 변환 완료 → {r}   (원본: {path}.bak)")
    return True


def gc_disable(): _ml_gc_disable()
def gc_collect(): _ml_gc_collect()
def resset(model_name): _ml_resset(model_name)

import builtins
builtins.resset     = resset
builtins.change     = change
builtins.gc_disable = gc_disable
builtins.gc_collect = gc_collect
`;

// ─────────────────────────────────────────────
// Module init
// ─────────────────────────────────────────────
private __gshared PyMethodDef[15] _methods;
private __gshared PyModuleDef     _moddef;

extern(C) PyObject* PyInit_my_ml() nothrow @trusted {
    try {
        _methods[0]  = PyMethodDef("_ml_make",           &py_ml_make,           METH_VARARGS, null);
        _methods[1]  = PyMethodDef("_ml_pick",           &py_ml_pick,           METH_VARARGS, null);
        _methods[2]  = PyMethodDef("_ml_pick_cos",       &py_ml_pick_cos,       METH_VARARGS, null);
        _methods[3]  = PyMethodDef("_ml_predict",        &py_ml_predict,        METH_VARARGS, null);
        _methods[4]  = PyMethodDef("_ml_learn",          &py_ml_learn,          METH_VARARGS, null);
        _methods[5]  = PyMethodDef("_ml_sl",             &py_ml_sl,             METH_VARARGS, null);
        _methods[6]  = PyMethodDef("_ml_sl_cos",         &py_ml_sl_cos,         METH_VARARGS, null);
        _methods[7]  = PyMethodDef("_ml_save",           &py_ml_save,           METH_VARARGS, null);
        _methods[8]  = PyMethodDef("_ml_resset",         &py_ml_resset,         METH_VARARGS, null);
        _methods[9]  = PyMethodDef("_ml_gc_disable",     &py_ml_gc_disable,     METH_VARARGS, null);
        _methods[10] = PyMethodDef("_ml_gc_collect",     &py_ml_gc_collect,     METH_VARARGS, null);
        _methods[11] = PyMethodDef("_ml_export_weights", &py_ml_export_weights, METH_VARARGS, null);
        _methods[12] = PyMethodDef("_ml_get_meta",       &py_ml_get_meta,       METH_VARARGS, null);
        _methods[13] = PyMethodDef("_ml_change",         &py_ml_change,         METH_VARARGS, null);
        _methods[14] = PyMethodDef(null, null, 0, null);

        _moddef.m_base.ob_base.ob_refcnt = 1;
        _moddef.m_name    = "my_ml";
        _moddef.m_doc     = null;
        _moddef.m_size    = -1;
        _moddef.m_methods = _methods.ptr;

        auto mod = PyModule_Create2(&_moddef, PYTHON_API_VERSION);
        if (!mod) return null;

        auto globals  = PyModule_GetDict(mod);
        auto none_obj = PyRun_String("None", Py_eval_input, globals, globals);
        if (none_obj) { _pyNone = none_obj; Py_IncRef(_pyNone); Py_DecRef(none_obj); }
        auto err_obj  = PyRun_String("RuntimeError", Py_eval_input, globals, globals);
        if (err_obj)  { _pyRuntimeError = err_obj; Py_IncRef(_pyRuntimeError); Py_DecRef(err_obj); }

        auto res = PyRun_String(PY_CLASS_CODE.ptr, Py_file_input, globals, globals);
        if (!res) return null;
        Py_DecRef(res);
        return mod;
    } catch (Throwable) { return null; }
}
