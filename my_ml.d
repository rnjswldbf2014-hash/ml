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
import std.math      : exp, sqrt, pow, log, cos, PI, abs;
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
// ── LayerNorm ─────────────────────────────────
private struct LN {
    int D;
    float[] g, b;             // 배울 값 (gamma, beta)
    float[] gg, gb;           // 기울기
    float[] mg, vg, mb, vb;   // 옵티마이저 상태
    int t;

    // 순전파 때 저장 — 역전파에 필요
    float[] mu, rstd;

    this(int d, int maxT) {
        D = d;
        g = new float[d]; b = new float[d];
        gg = new float[d]; gb = new float[d];
        mg = new float[d]; vg = new float[d];
        mb = new float[d]; vb = new float[d];
        // D 는 new float[] 를 NaN 으로 채운다. 반드시 직접 0 을 넣어야 한다.
        gg[] = 0f; gb[] = 0f; mg[] = 0f; vg[] = 0f; mb[] = 0f; vb[] = 0f;
        foreach (i; 0..d) { g[i] = 1f; b[i] = 0f; }
        mu = new float[maxT]; rstd = new float[maxT];
        mu[] = 0f; rstd[] = 0f;
    }

    // x, y 는 T*D 평면 배열
    void fwd(const(float)[] x, float[] y, int T) nothrow @nogc {
        foreach (t_; 0..T) {
            auto xs = x[t_*D .. t_*D + D];
            float m = 0f;
            foreach (v; xs) m += v;
            m /= D;
            float s = 0f;
            foreach (v; xs) { float d_ = v - m; s += d_*d_; }
            float r = 1f / sqrt(s/D + 1e-5f);
            mu[t_] = m; rstd[t_] = r;
            foreach (i; 0..D) y[t_*D + i] = g[i] * ((xs[i] - m) * r) + b[i];
        }
    }

    // dy -> dx (기울기 gg, gb 에 누적)
    void bwd(const(float)[] x, const(float)[] dy, float[] dx, int T) nothrow @nogc {
        foreach (t_; 0..T) {
            auto xs = x[t_*D .. t_*D + D];
            float m = mu[t_], r = rstd[t_];
            float sum1 = 0f, sum2 = 0f;
            foreach (i; 0..D) {
                float xh = (xs[i] - m) * r;
                float dyg = dy[t_*D + i] * g[i];
                gg[i] += dy[t_*D + i] * xh;
                gb[i] += dy[t_*D + i];
                sum1 += dyg;
                sum2 += dyg * xh;
            }
            sum1 /= D; sum2 /= D;
            foreach (i; 0..D) {
                float xh = (xs[i] - m) * r;
                float dyg = dy[t_*D + i] * g[i];
                dx[t_*D + i] += r * (dyg - sum1 - xh * sum2);
            }
        }
    }

    void zero() nothrow @nogc { gg[] = 0f; gb[] = 0f; }

    void step(Opt o, float lr) nothrow {
        adamVec(g, gg, mg, vg, o, lr, t);
        adamVec(b, gb, mb, vb, o, lr, t);
        t++;
    }
}

// 벡터 하나에 대한 옵티마이저 갱신 (Linear.step 과 같은 규칙)
private void adamVec(float[] w, float[] gr, float[] m, float[] v,
                     Opt o, float lr, int t) nothrow {
    enum float B1=0.9f, B2=0.999f, EPS=1e-8f, RHO=0.99f;
    final switch (o) {
        case Opt.adam:
            float bc1 = 1f - B1^^(t+1), bc2 = 1f - B2^^(t+1);
            foreach (i; 0..w.length) {
                float gv = gr[i];
                m[i] = B1*m[i] + (1-B1)*gv;
                v[i] = B2*v[i] + (1-B2)*gv*gv;
                w[i] -= lr * (m[i]/bc1) / (sqrt(v[i]/bc2) + EPS);
            }
            break;
        case Opt.sgd:
            foreach (i; 0..w.length) w[i] -= lr * gr[i];
            break;
        case Opt.rmsprop:
            foreach (i; 0..w.length) {
                float gv = gr[i];
                v[i] = RHO*v[i] + (1-RHO)*gv*gv;
                w[i] -= lr * gv / (sqrt(v[i]) + EPS);
            }
            break;
        case Opt.adagrad:
            foreach (i; 0..w.length) {
                float gv = gr[i];
                v[i] += gv*gv;
                w[i] -= lr * gv / (sqrt(v[i]) + EPS);
            }
            break;
    }
}


// ─────────────────────────────────────────────
// AttnLayer — 어텐션을 일반 층으로
// ─────────────────────────────────────────────
//   폭 dim 짜리 벡터를 items 조각으로 나눠 서로 참조하게 한다.
//   들어온 폭 그대로 나가므로 Linear 층 사이에 그냥 끼울 수 있다.
//     [38, 128, attn(8), 128]   <- 128 을 8조각(각 16) 으로 보고 섞음
//   순서 개념이 없으므로 마스크를 걸지 않는다 (모두가 모두를 봄).
// ─────────────────────────────────────────────
private struct AttnLayer {
    int dim;        // 전체 폭 (입력 = 출력)
    int items;      // 조각 수
    int w;          // 조각 하나의 폭 = dim/items
    int heads;      // 어텐션 헤드
    int hw;         // 헤드 하나의 폭 = w/heads

    LN     ln;
    Linear wq, wk, wv, wo;

    // 순전파 캐시
    float[] x1, q, k, v, att, ao, po;
    // 역전파 임시
    float[] dq, dk, dv, datt, dao, dx1;

    this(int dim_, int items_, int heads_) {
        dim = dim_; items = items_; w = dim_ / items_; heads = heads_; hw = w / heads_;
        ln = LN(w, items);
        wq = Linear(w, w); wk = Linear(w, w); wv = Linear(w, w); wo = Linear(w, w);

        x1 = new float[dim]; q = new float[dim]; k = new float[dim]; v = new float[dim];
        att = new float[heads*items*items];
        ao = new float[dim]; po = new float[dim];
        dq = new float[dim]; dk = new float[dim]; dv = new float[dim];
        datt = new float[heads*items*items];
        dao = new float[dim]; dx1 = new float[dim];
        foreach (a; [x1,q,k,v,att,ao,po,dq,dk,dv,datt,dao,dx1]) a[] = 0f;
    }

    // y = x + 어텐션(LayerNorm(x))
    void fwd(const(float)[] x, float[] y) nothrow @nogc {
        ln.fwd(x, x1, items);
        foreach (t; 0..items) {
            wq.forward(x1[t*w .. t*w+w], q[t*w .. t*w+w]);
            wk.forward(x1[t*w .. t*w+w], k[t*w .. t*w+w]);
            wv.forward(x1[t*w .. t*w+w], v[t*w .. t*w+w]);
        }
        float scale = 1f / sqrt(cast(float) hw);
        foreach (h; 0..heads) {
            foreach (t; 0..items) {
                float mx = -1e30f;
                foreach (s; 0..items) {
                    float dot = 0f;
                    foreach (i; 0..hw) dot += q[t*w + h*hw + i] * k[s*w + h*hw + i];
                    dot *= scale;
                    att[h*items*items + t*items + s] = dot;
                    if (dot > mx) mx = dot;
                }
                float sum = 0f;
                foreach (s; 0..items) {
                    float e = exp(att[h*items*items + t*items + s] - mx);
                    att[h*items*items + t*items + s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                foreach (s; 0..items) att[h*items*items + t*items + s] *= inv;
                foreach (i; 0..hw) {
                    float acc = 0f;
                    foreach (s; 0..items)
                        acc += att[h*items*items + t*items + s] * v[s*w + h*hw + i];
                    ao[t*w + h*hw + i] = acc;
                }
            }
        }
        foreach (t; 0..items) wo.forward(ao[t*w .. t*w+w], po[t*w .. t*w+w]);
        foreach (i; 0..dim) y[i] = x[i] + po[i];      // 잔차
    }

    // dy -> dx (dx 에 누적)
    void bwd(const(float)[] x, const(float)[] dy, float[] dx) nothrow @nogc {
        dao[] = 0f;
        foreach (t; 0..items)
            wo.accum(ao[t*w .. t*w+w], dy[t*w .. t*w+w], dao[t*w .. t*w+w]);

        dq[] = 0f; dk[] = 0f; dv[] = 0f;
        float scale = 1f / sqrt(cast(float) hw);
        foreach (h; 0..heads) {
            foreach (t; 0..items) {
                foreach (s; 0..items) {
                    float a = att[h*items*items + t*items + s];
                    float d_ = 0f;
                    foreach (i; 0..hw) {
                        d_ += dao[t*w + h*hw + i] * v[s*w + h*hw + i];
                        dv[s*w + h*hw + i] += a * dao[t*w + h*hw + i];
                    }
                    datt[h*items*items + t*items + s] = d_;
                }
                float dot = 0f;
                foreach (s; 0..items)
                    dot += datt[h*items*items + t*items + s] * att[h*items*items + t*items + s];
                foreach (s; 0..items) {
                    float a = att[h*items*items + t*items + s];
                    float ds = a * (datt[h*items*items + t*items + s] - dot) * scale;
                    foreach (i; 0..hw) {
                        dq[t*w + h*hw + i] += ds * k[s*w + h*hw + i];
                        dk[s*w + h*hw + i] += ds * q[t*w + h*hw + i];
                    }
                }
            }
        }

        dx1[] = 0f;
        foreach (t; 0..items) {
            wq.accum(x1[t*w .. t*w+w], dq[t*w .. t*w+w], dx1[t*w .. t*w+w]);
            wk.accum(x1[t*w .. t*w+w], dk[t*w .. t*w+w], dx1[t*w .. t*w+w]);
            wv.accum(x1[t*w .. t*w+w], dv[t*w .. t*w+w], dx1[t*w .. t*w+w]);
        }
        ln.bwd(x, dx1, dx, items);
        foreach (i; 0..dim) dx[i] += dy[i];           // 잔차 통과분
    }

    void zeroGrad() nothrow @nogc {
        ln.zero();
        wq.zeroGrad(); wk.zeroGrad(); wv.zeroGrad(); wo.zeroGrad();
    }

    void step(Opt o, float lr) nothrow {
        ln.step(o, lr);
        wq.step(o, lr); wk.step(o, lr); wv.step(o, lr); wo.step(o, lr);
    }
}

private class Network {
    // 은닉층은 Linear(+ReLU) 와 Attn 이 섞일 수 있다.
    // kinds[i] == 0 이면 lins[slot[i]], 1 이면 attns[slot[i]]
    ubyte[]     kinds;
    int[]       slot;
    Linear[]    lins;
    AttnLayer[] attns;
    Linear[]    heads;
    int         inputSz;

    int[]     _inSz, _outSz;
    float[][] _inp, _pre;
    float[]   _hout;
    float[][] _hd, _dHead;
    float[]   _dA, _dB;
    bool      fwdCached;

    int layerCount() const nothrow @nogc { return cast(int) kinds.length; }

    // specKind[i]: 0=Linear(specA=출력폭), 1=Attn(specA=조각수, specB=헤드수)
    this(int inputSz_, const(ubyte)[] specKind, const(int)[] specA, const(int)[] specB,
         int[] headSizes) {
        inputSz = inputSz_;
        int prev = inputSz;
        foreach (i; 0..specKind.length) {
            if (specKind[i] == 0) {
                lins ~= Linear(prev, specA[i]);
                kinds ~= 0; slot ~= cast(int)(lins.length - 1);
                _inSz ~= prev; _outSz ~= specA[i];
                prev = specA[i];
            } else {
                attns ~= AttnLayer(prev, specA[i], specB[i]);
                kinds ~= 1; slot ~= cast(int)(attns.length - 1);
                _inSz ~= prev; _outSz ~= prev;      // 폭 유지
            }
        }
        foreach (sz; headSizes) heads ~= Linear(prev, sz);
        _allocScratch();
    }

    // 예전 형태(전부 Linear) 로 만들 때
    this(int inputSz_, int[] hiddenSizes, int[] headSizes) {
        auto kk = new ubyte[hiddenSizes.length];
        auto aa = new int[hiddenSizes.length];
        auto bb = new int[hiddenSizes.length];
        foreach (i, sz; hiddenSizes) { kk[i] = 0; aa[i] = sz; bb[i] = 0; }
        this(inputSz_, kk, aa, bb, headSizes);
    }

    private void _allocScratch() {
        int n = layerCount;
        _inp = new float[][n]; _pre = new float[][n];
        foreach (i; 0..n) {
            _inp[i] = new float[_inSz[i]]; _inp[i][] = 0f;
            _pre[i] = new float[_outSz[i]]; _pre[i][] = 0f;
        }
        int houtSz = n > 0 ? _outSz[n-1] : inputSz;
        _hout = new float[houtSz]; _hout[] = 0f;
        _hd = new float[][heads.length]; _dHead = new float[][heads.length];
        foreach (i, ref h; heads) {
            _hd[i] = new float[h.outSz]; _hd[i][] = 0f;
            _dHead[i] = new float[h.outSz]; _dHead[i][] = 0f;
        }
        int maxSz = inputSz;
        foreach (i; 0..n) {
            if (_inSz[i] > maxSz) maxSz = _inSz[i];
            if (_outSz[i] > maxSz) maxSz = _outSz[i];
        }
        _dA = new float[maxSz]; _dB = new float[maxSz];
        _dA[] = 0f; _dB[] = 0f;
        fwdCached = false;
    }

    void forward(const(float)[] x) nothrow @nogc {
        int n = layerCount;
        if (n == 0) {
            foreach (k; 0..x.length) _hout[k] = x[k];
        } else {
            foreach (k; 0..x.length) _inp[0][k] = x[k];
            foreach (i; 0..n) {
                int oS = _outSz[i];
                if (kinds[i] == 0) {
                    lins[slot[i]].forward(_inp[i], _pre[i]);
                    // Linear 뒤에만 ReLU
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k] < 0f ? 0f : _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k] < 0f ? 0f : _pre[i][k];
                } else {
                    attns[slot[i]].fwd(_inp[i], _pre[i]);
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k];
                }
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
        for (int i = layerCount - 1; i >= 0; i--) {
            int oS = _outSz[i], iS = _inSz[i];
            if (kinds[i] == 0) {
                foreach (k; 0..oS) if (_pre[i][k] <= 0f) _dA[k] = 0f;
                _dB[0..iS] = 0f;
                lins[slot[i]].accum(_inp[i], _dA[0..oS], _dB[0..iS]);
            } else {
                _dB[0..iS] = 0f;
                attns[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS]);
            }
            foreach (k; 0..iS) _dA[k] = _dB[k];
        }
        fwdCached = false;
    }

    void zeroGrad() nothrow @nogc {
        foreach (ref h; lins)  h.zeroGrad();
        foreach (ref a; attns) a.zeroGrad();
        foreach (ref h; heads) h.zeroGrad();
    }

    void step(Opt opt, float lr) nothrow {
        foreach (ref h; lins)  h.step(opt, lr);
        foreach (ref a; attns) a.step(opt, lr);
        foreach (ref h; heads) h.step(opt, lr);
    }
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
// BlackBoxAI — 순수 파이프라인 + 다중 출력 헤드
//   헤드마다 이산(액션 선택) / cos(실수) 를 자유롭게 섞을 수 있다.
//   pickAll()   : 순전파 + 샘플링. 모델을 바꾸지 않는다.
//   learnBatch(): (입력, 출력, 보상) 묶음으로 역전파.
// ─────────────────────────────────────────────
private enum float GCLIP = 5.0f;   // cos 기울기 상한

class BlackBoxAI {
    string     name;
    string[][] actionLists;   // 헤드별 액션 이름 (cos 헤드는 빈 배열)
    bool[]     cosModes;      // 헤드별 cos 여부
    int[]      outSizes;      // 헤드별 출력 개수
    int[]      hiddenSizes;   // Linear 층 폭 (호환용)
    ubyte[]    layKind;       // 0=Linear, 1=Attn
    int[]      layA, layB;    // Linear: A=폭 / Attn: A=조각수, B=헤드수
    float      lr = 0.01f;
    float      cosSigma = 1.0f;   // cos 헤드 탐험 폭
    float      entropy  = 0.01f;  // 이산 헤드 엔트로피 보너스
    Opt        opt;
    string     file;

    Network net;
    bool    ready;

    private float[]   _envBuf;
    private float[][] _probBufs;   // 헤드별 확률 스크래치
    private int[][]   _maskBufs;   // 헤드별 마스크 스크래치

    int nHeads() const nothrow @nogc { return cast(int) outSizes.length; }

    this(string name, int inputSz, ubyte[] lk, int[] la, int[] lb, int[] heads,
         string[][] actions, bool[] cos,
         Opt opt = Opt.adam, float sigma = 1.0f, float ent = 0.01f) {
        this.name = name; this.opt = opt;
        cosSigma = sigma; entropy = ent;
        layKind = lk.dup; layA = la.dup; layB = lb.dup;
        hiddenSizes = [];
        foreach (i; 0..lk.length) if (lk[i] == 0) hiddenSizes ~= la[i];
        outSizes    = heads.dup;
        actionLists = actions.dup;
        cosModes    = cos.dup;
        file = name ~ "_ml_memory.pth";

        string 층설명() {
            string r = "[";
            foreach (i; 0..layKind.length) {
                if (i) r ~= ", ";
                r ~= layKind[i] == 0 ? to!string(layA[i])
                                     : "attn(" ~ to!string(layA[i]) ~ "," ~ to!string(layB[i]) ~ ")";
            }
            return r ~ "]";
        }

        if (exists(file)) {
            try {
                load();
                bool same = (net.inputSz == inputSz)
                         && (outSizes.length == heads.length)
                         && (layKind.length == lk.length);
                if (same) foreach (i; 0..lk.length)
                    if (layKind[i] != lk[i] || layA[i] != la[i] || layB[i] != lb[i]) { same = false; break; }
                if (same) foreach (i, sz; outSizes) if (sz != heads[i]) { same = false; break; }
                if (!same) {
                    string 저장된 = 층설명();
                    layKind = lk.dup; layA = la.dup; layB = lb.dup;
                    hiddenSizes = [];
                    foreach (i; 0..lk.length) if (lk[i] == 0) hiddenSizes ~= la[i];
                    writefln(" [%s] 저장된 구조 %s 가 요청한 %s 와 다릅니다. 새로 만듭니다.",
                             name, 저장된, 층설명());
                    outSizes = heads.dup;
                    actionLists = actions.dup; cosModes = cos.dup;
                    net = null; ready = false;
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
            net = new Network(inputSz, layKind, layA, layB, outSizes);
            writefln(" [%s] 새로 생성되었습니다. %s->%s", name, 층설명(), outSizes);
            ready = true;
        }
        _allocBufs();
    }

    private void _allocBufs() {
        _envBuf   = new float[net.inputSz];
        _probBufs = new float[][nHeads];
        _maskBufs = new int[][nHeads];
        foreach (i; 0..nHeads) {
            _probBufs[i] = new float[outSizes[i]];
            _maskBufs[i] = new int[outSizes[i]];
        }
    }

    private int maskOf(int h, string[] legal) {
        if (cosModes[h] || legal.length == 0) {
            foreach (k; 0..outSizes[h]) _maskBufs[h][k] = k;
            return outSizes[h];
        }
        return buildMaskInto(actionLists[h], legal, _maskBufs[h]);
    }

    private float gauss() {
        float u1 = cast(float) uniform01!double(rng);
        float u2 = cast(float) uniform01!double(rng);
        if (u1 < 1e-9f) u1 = 1e-9f;
        return sqrt(-2f * log(u1)) * cos(2f * PI * u2);
    }

    // ── 모든 헤드에서 한 번에 뽑는다. 모델은 그대로. ──
    //   이산 헤드 : chosen[h] = 액션 인덱스,  value[h] = 0
    //   cos  헤드 : chosen[h] = 유닛 번호,    value[h] = 뽑은 실수
    void pickAll(string[][] legal, float[] input, int[] chosen, float[] value) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        foreach (h; 0..nHeads) {
            if (cosModes[h]) {
                int unit = outSizes[h] > 1 ? uniform(0, outSizes[h], rng) : 0;
                chosen[h] = unit;
                value[h] = net._hd[h][unit] + cosSigma * gauss();
            } else {
                auto lg = (h < legal.length) ? legal[h] : null;
                int mlen = maskOf(h, lg);
                foreach (k; 0..mlen) _probBufs[h][k] = net._hd[h][_maskBufs[h][k]];
                softmaxInPlace(_probBufs[h][0..mlen]);
                double r = uniform01!double(rng), cum = 0.0;
                int c = mlen - 1;
                foreach (k; 0..mlen) { cum += _probBufs[h][k]; if (r < cum) { c = k; break; } }
                chosen[h] = _maskBufs[h][c];
                value[h]  = 0f;
            }
        }
    }

    // ── 예측(샘플링 없음) ──
    void predictAll(string[][] legal, float[] input, int[] chosen, float[] value) {
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.forward(_envBuf);
        foreach (h; 0..nHeads) {
            if (cosModes[h]) {
                chosen[h] = 0;
                value[h]  = net._hd[h][0];
            } else {
                auto lg = (h < legal.length) ? legal[h] : null;
                int mlen = maskOf(h, lg);
                int best = 0;
                foreach (k; 1..mlen)
                    if (net._hd[h][_maskBufs[h][k]] > net._hd[h][_maskBufs[h][best]]) best = k;
                chosen[h] = _maskBufs[h][best];
                value[h]  = 0f;
            }
        }
    }

    // ── 역전파 ──
    //   scores[i][h] 가 NaN 이면 그 스텝에서 그 헤드는 건너뛴다.
    void learnBatch(float[][] inputs, int[][] chosen, float[][] values, float[][] scores) {
        if (inputs.length == 0) return;
        net.zeroGrad();
        // 배치 크기로 나눠 평균 기울기를 쓴다.
        // (합산만 하면 배치가 커질수록 갱신 폭이 커져 발산한다)
        immutable float inv = 1.0f / cast(float) inputs.length;
        foreach (i; 0..inputs.length) {
            foreach (k; 0..inputs[i].length) _envBuf[k] = inputs[i][k];
            net.forward(_envBuf);
            foreach (h; 0..nHeads) {
                net._dHead[h][] = 0f;
                float sc = scores[i][h];
                if (sc != sc) continue;   // NaN -> 이 헤드는 학습 안 함
                sc *= inv;
                if (cosModes[h]) {
                    int u = chosen[i][h] < outSizes[h] ? chosen[i][h] : 0;
                    float mu = net._hd[h][u];
                    float gr = -sc * (values[i][h] - mu) / (cosSigma * cosSigma);
                    if (gr >  GCLIP) gr =  GCLIP;      // 발산 방지
                    if (gr < -GCLIP) gr = -GCLIP;
                    net._dHead[h][u] = gr;
                } else {
                    int n = outSizes[h];
                    foreach (k; 0..n) _probBufs[h][k] = net._hd[h][k];
                    softmaxInPlace(_probBufs[h][0..n]);
                    // 엔트로피 H = -sum p log p  (탐험이 죽는 것을 막는다)
                    float Hh = 0f;
                    foreach (k; 0..n) {
                        float pk = _probBufs[h][k];
                        if (pk > 1e-8f) Hh -= pk * log(pk);
                    }
                    foreach (k; 0..n) {
                        float pk = _probBufs[h][k];
                        float g  = sc * (pk - (k == chosen[i][h] ? 1f : 0f));
                        if (entropy != 0f && pk > 1e-8f)
                            g += entropy * inv * pk * (log(pk) + Hh);
                        net._dHead[h][k] = g;
                    }
                }
            }
            net.backward();
        }
        net.step(opt, lr);
    }

    // ── 지도학습 1스텝 (헤드별 정답; 이산은 인덱스, cos 는 목표값) ──
    void slBatch(float[] input, string[][] legal, int[] ansIdx, float[] ansVal, bool[] use) {
        bool any = false;
        foreach (u; use) if (u) { any = true; break; }
        if (!any) return;
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        net.zeroGrad();
        net.forward(_envBuf);
        foreach (h; 0..nHeads) {
            net._dHead[h][] = 0f;
            if (!use[h]) continue;
            if (cosModes[h]) {
                net._dHead[h][0] = net._hd[h][0] - ansVal[h];
            } else {
                auto lg = (h < legal.length) ? legal[h] : null;
                int mlen = maskOf(h, lg);
                foreach (k; 0..mlen) _probBufs[h][k] = net._hd[h][_maskBufs[h][k]];
                softmaxInPlace(_probBufs[h][0..mlen]);
                int tgt = -1;
                foreach (k; 0..mlen) if (_maskBufs[h][k] == ansIdx[h]) { tgt = k; break; }
                foreach (k; 0..mlen)
                    net._dHead[h][_maskBufs[h][k]] = _probBufs[h][k] - (k == tgt ? 1f : 0f);
            }
        }
        net.backward();
        net.step(opt, lr);
    }

    void save() {
        if (!ready) return;
        auto f = File(file, "wb");
        void wu(uint v)  { f.rawWrite((&v)[0..1]); }
        void wf(float v) { f.rawWrite((&v)[0..1]); }
        wu(0xBEEFCAFE); wu(7); wu(cast(uint)opt);
        wu(cast(uint)net.inputSz);
        wu(cast(uint)layKind.length);
        foreach (i; 0..layKind.length) {
            wu(cast(uint)layKind[i]); wu(cast(uint)layA[i]); wu(cast(uint)layB[i]);
        }
        wu(cast(uint)nHeads);
        foreach (h; 0..nHeads) {
            wu(cast(uint)outSizes[h]);
            wu(cosModes[h] ? 1u : 0u);
            wu(cast(uint)actionLists[h].length);
            foreach (a; actionLists[h]) {
                auto bytes = cast(ubyte[])a; wu(cast(uint)bytes.length); f.rawWrite(bytes);
            }
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
        void wn(ref LN n) {
            foreach (v; n.g) wf(v); foreach (v; n.mg) wf(v); foreach (v; n.vg) wf(v);
            foreach (v; n.b) wf(v); foreach (v; n.mb) wf(v); foreach (v; n.vb) wf(v);
            wu(cast(uint)n.t);
        }
        foreach (i; 0..net.layerCount) {
            if (net.kinds[i] == 0) wl(net.lins[net.slot[i]]);
            else {
                auto a = &net.attns[net.slot[i]];
                wn(a.ln); wl(a.wq); wl(a.wk); wl(a.wv); wl(a.wo);
            }
        }
        foreach (ref h; net.heads) wl(h);
    }

    private void load() {
        auto f = File(file, "rb");
        uint  ru()  { uint v;  f.rawRead((&v)[0..1]); return v; }
        float rf()  { float v; f.rawRead((&v)[0..1]); return v; }
        if (ru() != 0xBEEFCAFE) throw new Exception("magic mismatch");
        uint ver = ru();
        if (ver < 7) throw new Exception("예전 포맷입니다. change() 로 변환하세요");
        opt = cast(Opt)ru();
        int inputSz = ru();
        int nLay = ru();
        layKind = new ubyte[nLay]; layA = new int[nLay]; layB = new int[nLay];
        hiddenSizes = [];
        foreach (i; 0..nLay) {
            layKind[i] = cast(ubyte)ru(); layA[i] = ru(); layB[i] = ru();
            if (layKind[i] == 0) hiddenSizes ~= layA[i];
        }
        int nL = ru();
        outSizes = new int[nL]; cosModes = new bool[nL]; actionLists = new string[][nL];
        foreach (i; 0..nL) {
            outSizes[i] = ru();
            cosModes[i] = ru() != 0;
            int nA = ru(); actionLists[i] = new string[nA];
            foreach (j; 0..nA) { auto buf = new ubyte[ru()]; f.rawRead(buf); actionLists[i][j] = cast(string)buf.dup; }
        }
        net = new Network(inputSz, layKind, layA, layB, outSizes);
        void rl_(ref Linear l) {
            foreach (ref row; l.w)  foreach (ref v; row) v = rf();
            foreach (ref v; l.b)    v = rf();
            foreach (ref row; l.mW) foreach (ref v; row) v = rf();
            foreach (ref row; l.vW) foreach (ref v; row) v = rf();
            foreach (ref v; l.mB)   v = rf();
            foreach (ref v; l.vB)   v = rf();
            l.t = ru();
        }
        void rn_(ref LN n) {
            foreach (ref v; n.g) v = rf(); foreach (ref v; n.mg) v = rf(); foreach (ref v; n.vg) v = rf();
            foreach (ref v; n.b) v = rf(); foreach (ref v; n.mb) v = rf(); foreach (ref v; n.vb) v = rf();
            n.t = ru();
        }
        foreach (i; 0..net.layerCount) {
            if (net.kinds[i] == 0) rl_(net.lins[net.slot[i]]);
            else {
                auto a = &net.attns[net.slot[i]];
                rn_(a.ln); rl_(a.wq); rl_(a.wk); rl_(a.wv); rl_(a.wo);
            }
        }
        foreach (ref h; net.heads) rl_(h);
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
// change — 예전 포맷 파일을 현재 포맷(ver 4)으로 변환
//   ver 2 : 다중 헤드 → 헤드 전부 보존
//   ver 3 : 단일 헤드 + cos 플래그
//   원본은 .bak 으로 남긴다.
// ─────────────────────────────────────────────
private int linearBytes(int inSz, int outSz) pure nothrow @nogc {
    // 저장 순서: w, b, mW, vW, mB, vB, t
    return cast(int)((3L*outSz*inSz + 3L*outSz) * 4 + 4);
}

string changeFile(string path) {
    if (!exists(path)) throw new Exception(path ~ " 파일이 없습니다");
    if (isTorchFile(path))
        throw new Exception("PyTorch(zip) 포맷입니다. 이 함수는 my_ml 자체 포맷만 변환합니다");

    Opt        o = Opt.adam;
    int        inputSz, nH, nL;
    int[]      hid;
    int[]      outSizes;
    bool[]     cosModes;
    string[][] als;
    ubyte[]    blob;

    {
        auto f = File(path, "rb");
        uint ru() { uint v; f.rawRead((&v)[0..1]); return v; }
        if (ru() != 0xBEEFCAFE) throw new Exception("my_ml 포맷이 아닙니다");
        uint ver = ru();
        if (ver >= 7) return "already";
        if (ver >= 2) o = cast(Opt) ru();

        bool cos3 = false; int out3 = 0;
        if (ver == 3) { cos3 = ru() != 0; out3 = ru(); }
        bool ver4 = (ver >= 4);
        bool ver5 = (ver == 5);

        inputSz = ru();
        nH = ru(); hid = new int[nH];
        foreach (i; 0..nH) hid[i] = ru();
        nL = ru();
        als = new string[][nL]; outSizes = new int[nL]; cosModes = new bool[nL];
        foreach (i; 0..nL) {
            if (ver4) { outSizes[i] = ru(); cosModes[i] = ru() != 0; }
            if (ver5) { ru(); ru(); }      // 옛 cos 범위 자리 — 버린다
            int nA = ru(); als[i] = new string[nA];
            foreach (j; 0..nA) { auto b = new ubyte[ru()]; f.rawRead(b); als[i][j] = cast(string) b.dup; }
            if (!ver4) {
                outSizes[i] = (ver == 3) ? out3 : nA;
                cosModes[i] = (ver == 3) ? cos3 : false;
            }
        }
        if (nL == 0) throw new Exception("헤드가 없습니다");

        // 가중치는 배치가 동일하므로 통째로 옮긴다 (헤드 전부 보존)
        int prev = inputSz, total = 0;
        foreach (sz; hid) { total += linearBytes(prev, sz); prev = sz; }
        foreach (sz; outSizes) total += linearBytes(prev, sz);
        blob = new ubyte[total];
        if (total > 0) f.rawRead(blob);
    }

    string bak = path ~ ".bak";
    if (exists(bak)) remove(bak);
    rename(path, bak);

    auto w = File(path, "wb");
    void wu(uint v) { w.rawWrite((&v)[0..1]); }
    wu(0xBEEFCAFE); wu(7); wu(cast(uint) o);
    wu(cast(uint) inputSz);
    wu(cast(uint) nH);
    foreach (sz; hid) { wu(0u); wu(cast(uint) sz); wu(0u); }   // 전부 Linear 층
    wu(cast(uint) nL);
    foreach (i; 0..nL) {
        wu(cast(uint) outSizes[i]);
        wu(cosModes[i] ? 1u : 0u);
        wu(cast(uint) als[i].length);
        foreach (a; als[i]) { auto b = cast(ubyte[]) a; wu(cast(uint) b.length); w.rawWrite(b); }
    }
    w.rawWrite(blob);
    w.close();

    return to!string(inputSz) ~ "->" ~ to!string(hid) ~ "->" ~ to!string(outSizes);
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
// GPT — 트랜스포머 언어 모델
// ─────────────────────────────────────────────
//   토큰 열 -> 다음 토큰 확률
//   구조: 임베딩 + (LayerNorm -> 어텐션 -> LayerNorm -> FFN) x L -> 출력
//   전부 손으로 역전파한다. 기울기는 수치미분으로 검증했다.
// ─────────────────────────────────────────────

// ── 트랜스포머 블록 하나 ──────────────────────
private final class Block {
    int D, H, dh, F, maxT;
    LN ln1, ln2;
    Linear wq, wk, wv, wo;   // 어텐션
    Linear fc1, fc2;         // FFN

    // 순전파 캐시
    float[] x1;      // ln1 출력          T*D
    float[] q, k, v; // T*D
    float[] att;     // H*T*T  (softmax 결과)
    float[] ao;      // 어텐션 출력 (헤드 합침) T*D
    float[] po;      // wo 통과            T*D
    float[] r1;      // 잔차 1             T*D
    float[] x2;      // ln2 출력          T*D
    float[] h1;      // fc1 출력          T*F
    float[] h2;      // relu(h1)          T*F
    float[] fo;      // fc2 출력          T*D

    // 역전파 임시
    float[] dx1, dq, dk, dv, datt, dao, dpo, dx2, dh1, dh2;

    this(int d, int h, int maxT_) {
        D = d; H = h; dh = d / h; F = 4 * d; maxT = maxT_;
        ln1 = LN(d, maxT); ln2 = LN(d, maxT);
        wq = Linear(d, d); wk = Linear(d, d); wv = Linear(d, d); wo = Linear(d, d);
        fc1 = Linear(d, F); fc2 = Linear(F, d);

        x1 = new float[maxT*d]; q = new float[maxT*d];
        k  = new float[maxT*d]; v = new float[maxT*d];
        att = new float[h*maxT*maxT];
        ao = new float[maxT*d]; po = new float[maxT*d]; r1 = new float[maxT*d];
        x2 = new float[maxT*d]; h1 = new float[maxT*F]; h2 = new float[maxT*F];
        fo = new float[maxT*d];

        dx1 = new float[maxT*d]; dq = new float[maxT*d];
        dk  = new float[maxT*d]; dv = new float[maxT*d];
        datt = new float[h*maxT*maxT];
        dao = new float[maxT*d]; dpo = new float[maxT*d];
        dx2 = new float[maxT*d]; dh1 = new float[maxT*F]; dh2 = new float[maxT*F];
        // NaN 으로 시작하지 않도록 전부 0
        foreach (a; [x1,q,k,v,att,ao,po,r1,x2,h1,h2,fo,
                     dx1,dq,dk,dv,datt,dao,dpo,dx2,dh1,dh2]) a[] = 0f;
    }

    // 입력 x (T*D) -> 출력 y (T*D). x 는 역전파에서 다시 쓰므로 보존해야 한다.
    void fwd(const(float)[] x, float[] y, int T) nothrow @nogc {
        ln1.fwd(x, x1, T);
        foreach (t_; 0..T) {
            wq.forward(x1[t_*D .. t_*D+D], q[t_*D .. t_*D+D]);
            wk.forward(x1[t_*D .. t_*D+D], k[t_*D .. t_*D+D]);
            wv.forward(x1[t_*D .. t_*D+D], v[t_*D .. t_*D+D]);
        }

        float scale = 1f / sqrt(cast(float) dh);
        foreach (hh; 0..H) {
            foreach (t_; 0..T) {
                // 인과 마스크: s <= t 만 본다
                float mx = -1e30f;
                foreach (s; 0..t_+1) {
                    float dot = 0f;
                    foreach (i; 0..dh)
                        dot += q[t_*D + hh*dh + i] * k[s*D + hh*dh + i];
                    dot *= scale;
                    att[hh*maxT*maxT + t_*maxT + s] = dot;
                    if (dot > mx) mx = dot;
                }
                float sum = 0f;
                foreach (s; 0..t_+1) {
                    float e = exp(att[hh*maxT*maxT + t_*maxT + s] - mx);
                    att[hh*maxT*maxT + t_*maxT + s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                foreach (s; 0..t_+1) att[hh*maxT*maxT + t_*maxT + s] *= inv;
                // 가중합
                foreach (i; 0..dh) {
                    float acc = 0f;
                    foreach (s; 0..t_+1)
                        acc += att[hh*maxT*maxT + t_*maxT + s] * v[s*D + hh*dh + i];
                    ao[t_*D + hh*dh + i] = acc;
                }
            }
        }

        foreach (t_; 0..T) wo.forward(ao[t_*D .. t_*D+D], po[t_*D .. t_*D+D]);
        foreach (i; 0..T*D) r1[i] = x[i] + po[i];      // 잔차 1

        ln2.fwd(r1, x2, T);
        foreach (t_; 0..T) {
            fc1.forward(x2[t_*D .. t_*D+D], h1[t_*F .. t_*F+F]);
            foreach (i; 0..F) h2[t_*F + i] = h1[t_*F + i] < 0f ? 0f : h1[t_*F + i];
            fc2.forward(h2[t_*F .. t_*F+F], fo[t_*D .. t_*D+D]);
        }
        foreach (i; 0..T*D) y[i] = r1[i] + fo[i];      // 잔차 2
    }

    // dy -> dx
    void bwd(const(float)[] x, const(float)[] dy, float[] dx, int T) nothrow @nogc {
        // 잔차 2: dr1 += dy, dfo = dy
        dx2[0..T*D] = 0f;
        foreach (t_; 0..T) {
            dh2[t_*F .. t_*F+F] = 0f;
            fc2.accum(h2[t_*F .. t_*F+F], dy[t_*D .. t_*D+D], dh2[t_*F .. t_*F+F]);
            foreach (i; 0..F) dh1[t_*F + i] = h1[t_*F + i] > 0f ? dh2[t_*F + i] : 0f;
            fc1.accum(x2[t_*D .. t_*D+D], dh1[t_*F .. t_*F+F], dx2[t_*D .. t_*D+D]);
        }
        // ln2 를 통과해 r1 로
        auto dr1 = dpo;                 // 임시 재사용
        dr1[0..T*D] = 0f;
        ln2.bwd(r1, dx2[0..T*D], dr1[0..T*D], T);
        foreach (i; 0..T*D) dr1[i] += dy[i];     // 잔차 2 의 통과분

        // 잔차 1: dx += dr1, dpo_att = dr1
        dao[0..T*D] = 0f;
        foreach (t_; 0..T)
            wo.accum(ao[t_*D .. t_*D+D], dr1[t_*D .. t_*D+D], dao[t_*D .. t_*D+D]);

        // 어텐션 역전파
        dq[0..T*D] = 0f; dk[0..T*D] = 0f; dv[0..T*D] = 0f;
        float scale = 1f / sqrt(cast(float) dh);
        foreach (hh; 0..H) {
            foreach (t_; 0..T) {
                // dao -> datt, dv
                foreach (s; 0..t_+1) {
                    float a = att[hh*maxT*maxT + t_*maxT + s];
                    float d_ = 0f;
                    foreach (i; 0..dh) {
                        d_ += dao[t_*D + hh*dh + i] * v[s*D + hh*dh + i];
                        dv[s*D + hh*dh + i] += a * dao[t_*D + hh*dh + i];
                    }
                    datt[hh*maxT*maxT + t_*maxT + s] = d_;
                }
                // softmax 역전파
                float dot = 0f;
                foreach (s; 0..t_+1)
                    dot += datt[hh*maxT*maxT + t_*maxT + s] * att[hh*maxT*maxT + t_*maxT + s];
                foreach (s; 0..t_+1) {
                    float a = att[hh*maxT*maxT + t_*maxT + s];
                    float ds = a * (datt[hh*maxT*maxT + t_*maxT + s] - dot) * scale;
                    foreach (i; 0..dh) {
                        dq[t_*D + hh*dh + i] += ds * k[s*D + hh*dh + i];
                        dk[s*D + hh*dh + i] += ds * q[t_*D + hh*dh + i];
                    }
                }
            }
        }

        dx1[0..T*D] = 0f;
        foreach (t_; 0..T) {
            wq.accum(x1[t_*D .. t_*D+D], dq[t_*D .. t_*D+D], dx1[t_*D .. t_*D+D]);
            wk.accum(x1[t_*D .. t_*D+D], dk[t_*D .. t_*D+D], dx1[t_*D .. t_*D+D]);
            wv.accum(x1[t_*D .. t_*D+D], dv[t_*D .. t_*D+D], dx1[t_*D .. t_*D+D]);
        }
        ln1.bwd(x, dx1[0..T*D], dx, T);
        foreach (i; 0..T*D) dx[i] += dr1[i];     // 잔차 1 의 통과분
    }

    void zero() nothrow @nogc {
        ln1.zero(); ln2.zero();
        wq.zeroGrad(); wk.zeroGrad(); wv.zeroGrad(); wo.zeroGrad();
        fc1.zeroGrad(); fc2.zeroGrad();
    }

    void step(Opt o, float lr) nothrow {
        ln1.step(o, lr); ln2.step(o, lr);
        wq.step(o, lr); wk.step(o, lr); wv.step(o, lr); wo.step(o, lr);
        fc1.step(o, lr); fc2.step(o, lr);
    }
}

// ── GPT 본체 ──────────────────────────────────
final class GPT {
    string name, file;
    int V, D, H, L, maxT;
    float lr = 3e-4f;
    Opt opt;

    float[] tok, pos;            // 임베딩: V*D, maxT*D
    float[] gTok, gPos;
    float[] mTok, vTok, mPos, vPos;
    int embT;

    Block[] blocks;
    LN lnf;
    Linear head;                 // D -> V

    // 활성값 (층마다 입력을 남겨야 역전파가 된다)
    float[][] xs;                // (L+1) 개, 각 maxT*D
    float[] xf;                  // lnf 출력
    float[] logits;              // maxT*V
    float[] probs;               // maxT*V
    float[] dx, dxf, dlogits;

    this(string name_, int vocab, int dim, int heads, int layers, int ctx,
         Opt o = Opt.adam, float lr_ = 3e-4f) {
        name = name_; file = name_ ~ "_lm.pth";
        V = vocab; D = dim; H = heads; L = layers; maxT = ctx;
        opt = o; lr = lr_;

        tok = new float[V*D]; pos = new float[maxT*D];
        gTok = new float[V*D]; gPos = new float[maxT*D];
        mTok = new float[V*D]; vTok = new float[V*D];
        mPos = new float[maxT*D]; vPos = new float[maxT*D];
        gTok[] = 0f; gPos[] = 0f;
        mTok[] = 0f; vTok[] = 0f; mPos[] = 0f; vPos[] = 0f;
        float s = 0.02f;
        foreach (i; 0..V*D)    tok[i] = uniform(-s, s, rng);
        foreach (i; 0..maxT*D) pos[i] = uniform(-s, s, rng);

        blocks = new Block[layers];
        foreach (i; 0..layers) blocks[i] = new Block(dim, heads, ctx);
        lnf = LN(dim, ctx);
        head = Linear(dim, vocab);

        xs = new float[][](layers+1, ctx*dim);
        foreach (a; xs) a[] = 0f;
        xf = new float[ctx*dim];
        logits = new float[ctx*vocab];
        probs  = new float[ctx*vocab];
        dx = new float[ctx*dim]; dxf = new float[ctx*dim];
        dlogits = new float[ctx*vocab];
        xf[] = 0f; logits[] = 0f; probs[] = 0f;
        dx[] = 0f; dxf[] = 0f; dlogits[] = 0f;
    }

    long paramCount() const {
        long n = cast(long)V*D + cast(long)maxT*D;          // 임베딩
        n += cast(long)L * (12L*D*D + 13L*D);               // 블록 (대략)
        n += 2L*D + cast(long)D*V + V;                      // lnf + head
        return n;
    }

    // 순전파. toks 길이 T -> logits (T*V)
    void forward(const(int)[] toks, int T) nothrow @nogc {
        foreach (t_; 0..T) {
            int id = toks[t_];
            foreach (i; 0..D) xs[0][t_*D + i] = tok[id*D + i] + pos[t_*D + i];
        }
        foreach (l; 0..L) blocks[l].fwd(xs[l][0..T*D], xs[l+1][0..T*D], T);
        lnf.fwd(xs[L][0..T*D], xf, T);
        foreach (t_; 0..T) head.forward(xf[t_*D .. t_*D+D], logits[t_*V .. t_*V+V]);
    }

    // 다음 토큰 맞히기 손실. targets[t] = toks[t+1]
    // 반환: 평균 교차엔트로피. 기울기는 누적된다.
    float backward(const(int)[] toks, const(int)[] targets, int T) nothrow @nogc {
        float loss = 0f;
        foreach (t_; 0..T) {
            auto lg = logits[t_*V .. t_*V+V];
            float mx = lg[0];
            foreach (v_; lg) if (v_ > mx) mx = v_;
            float sum = 0f;
            foreach (i; 0..V) { float e = exp(lg[i] - mx); probs[t_*V + i] = e; sum += e; }
            float inv = 1f / sum;
            foreach (i; 0..V) probs[t_*V + i] *= inv;
            int tgt = targets[t_];
            loss -= log(probs[t_*V + tgt] + 1e-12f);
            foreach (i; 0..V)
                dlogits[t_*V + i] = (probs[t_*V + i] - (i == tgt ? 1f : 0f)) / T;
        }

        dxf[0..T*D] = 0f;
        foreach (t_; 0..T)
            head.accum(xf[t_*D .. t_*D+D], dlogits[t_*V .. t_*V+V], dxf[t_*D .. t_*D+D]);

        dx[0..T*D] = 0f;
        lnf.bwd(xs[L][0..T*D], dxf[0..T*D], dx[0..T*D], T);

        for (int l = L-1; l >= 0; l--) {
            dxf[0..T*D] = 0f;               // Block.bwd 는 누적하므로 먼저 비운다
            blocks[l].bwd(xs[l][0..T*D], dx[0..T*D], dxf[0..T*D], T);
            dx[0..T*D] = dxf[0..T*D];
        }

        foreach (t_; 0..T) {
            int id = toks[t_];
            foreach (i; 0..D) {
                gTok[id*D + i] += dx[t_*D + i];
                gPos[t_*D + i] += dx[t_*D + i];
            }
        }
        return loss / T;
    }

    void zeroGrad() nothrow @nogc {
        gTok[] = 0f; gPos[] = 0f;
        foreach (b; blocks) b.zero();
        lnf.zero(); head.zeroGrad();
    }

    void step() nothrow {
        adamVec(tok, gTok, mTok, vTok, opt, lr, embT);
        adamVec(pos, gPos, mPos, vPos, opt, lr, embT);
        embT++;
        foreach (b; blocks) b.step(opt, lr);
        lnf.step(opt, lr);
        head.step(opt, lr);
    }

    // 한 덩어리 학습. 반환 = 손실
    float trainOn(const(int)[] toks, const(int)[] targets, int T) {
        zeroGrad();
        forward(toks, T);
        float l = backward(toks, targets, T);
        step();
        return l;
    }

    // 다음 토큰 하나 뽑기 (temperature, top-k)
    int sample(const(int)[] ctx, int T, float temp, int topk) {
        forward(ctx, T);
        auto lg = logits[(T-1)*V .. T*V];
        auto p = new float[V];
        float mx = -1e30f;
        foreach (i; 0..V) { p[i] = lg[i] / (temp <= 0f ? 1e-6f : temp); if (p[i] > mx) mx = p[i]; }
        float sum = 0f;
        foreach (i; 0..V) { p[i] = exp(p[i] - mx); sum += p[i]; }
        foreach (i; 0..V) p[i] /= sum;

        if (topk > 0 && topk < V) {
            // topk 밖은 버린다
            auto idx = new int[V];
            foreach (i; 0..V) idx[i] = i;
            // 부분 선택 정렬 (topk 가 작으니 충분)
            foreach (a; 0..topk) {
                int best = a;
                foreach (b2; a+1..V) if (p[idx[b2]] > p[idx[best]]) best = b2;
                auto tmp = idx[a]; idx[a] = idx[best]; idx[best] = tmp;
            }
            float keep = 0f;
            foreach (a; 0..topk) keep += p[idx[a]];
            foreach (a; topk..V) p[idx[a]] = 0f;
            foreach (i; 0..V) p[i] /= keep;
        }

        double r = uniform01!double(rng), cum = 0.0;
        foreach (i; 0..V) { cum += p[i]; if (r < cum) return i; }
        return V-1;
    }
}

// ── GPT 저장 / 불러오기 ───────────────────────
// 파일: 이름_lm.pth   (magic 0x11ADCAFE)
private void wLinear(ref File f, ref Linear l) {
    void wf(float v) { f.rawWrite((&v)[0..1]); }
    void wu(uint v)  { f.rawWrite((&v)[0..1]); }
    foreach (row; l.w)  foreach (v; row) wf(v);
    foreach (row; l.mW) foreach (v; row) wf(v);
    foreach (row; l.vW) foreach (v; row) wf(v);
    foreach (v; l.b)  wf(v);
    foreach (v; l.mB) wf(v);
    foreach (v; l.vB) wf(v);
    wu(cast(uint) l.t);
}

private void rLinear(ref File f, ref Linear l) {
    float rf() { float v; f.rawRead((&v)[0..1]); return v; }
    uint  ru() { uint v;  f.rawRead((&v)[0..1]); return v; }
    foreach (ref row; l.w)  foreach (ref v; row) v = rf();
    foreach (ref row; l.mW) foreach (ref v; row) v = rf();
    foreach (ref row; l.vW) foreach (ref v; row) v = rf();
    foreach (ref v; l.b)  v = rf();
    foreach (ref v; l.mB) v = rf();
    foreach (ref v; l.vB) v = rf();
    l.t = ru();
}

private void wVec(ref File f, float[] a) { foreach (v; a) f.rawWrite((&v)[0..1]); }
private void rVec(ref File f, float[] a) { foreach (ref v; a) { float t; f.rawRead((&t)[0..1]); v = t; } }

private void wLN(ref File f, ref LN n) {
    void wu(uint v) { f.rawWrite((&v)[0..1]); }
    wVec(f, n.g); wVec(f, n.mg); wVec(f, n.vg);
    wVec(f, n.b); wVec(f, n.mb); wVec(f, n.vb);
    wu(cast(uint) n.t);
}
private void rLN(ref File f, ref LN n) {
    uint ru() { uint v; f.rawRead((&v)[0..1]); return v; }
    rVec(f, n.g); rVec(f, n.mg); rVec(f, n.vg);
    rVec(f, n.b); rVec(f, n.mb); rVec(f, n.vb);
    n.t = ru();
}

void saveGPT(GPT g) {
    auto f = File(g.file, "wb");
    void wu(uint v) { f.rawWrite((&v)[0..1]); }
    void wf(float v) { f.rawWrite((&v)[0..1]); }
    wu(0x11ADCAFE); wu(1); wu(cast(uint) g.opt); wf(g.lr);
    wu(cast(uint) g.V); wu(cast(uint) g.D); wu(cast(uint) g.H);
    wu(cast(uint) g.L); wu(cast(uint) g.maxT);
    wVec(f, g.tok); wVec(f, g.mTok); wVec(f, g.vTok);
    wVec(f, g.pos); wVec(f, g.mPos); wVec(f, g.vPos);
    wu(cast(uint) g.embT);
    foreach (b; g.blocks) {
        wLN(f, b.ln1); wLN(f, b.ln2);
        wLinear(f, b.wq); wLinear(f, b.wk); wLinear(f, b.wv); wLinear(f, b.wo);
        wLinear(f, b.fc1); wLinear(f, b.fc2);
    }
    wLN(f, g.lnf);
    wLinear(f, g.head);
}

void loadGPT(string name, out GPT g) {
    string path = name ~ "_lm.pth";
    auto f = File(path, "rb");
    uint ru() { uint v; f.rawRead((&v)[0..1]); return v; }
    float rf() { float v; f.rawRead((&v)[0..1]); return v; }
    if (ru() != 0x11ADCAFE) throw new Exception("lm 포맷이 아닙니다");
    uint ver = ru();
    if (ver != 1) throw new Exception("모르는 lm 버전");
    Opt o = cast(Opt) ru();
    float lr = rf();
    int V = ru(), D = ru(), H = ru(), L = ru(), T = ru();
    g = new GPT(name, V, D, H, L, T, o, lr);
    rVec(f, g.tok); rVec(f, g.mTok); rVec(f, g.vTok);
    rVec(f, g.pos); rVec(f, g.mPos); rVec(f, g.vPos);
    g.embT = ru();
    foreach (b; g.blocks) {
        rLN(f, b.ln1); rLN(f, b.ln2);
        rLinear(f, b.wq); rLinear(f, b.wk); rLinear(f, b.wv); rLinear(f, b.wo);
        rLinear(f, b.fc1); rLinear(f, b.fc2);
    }
    rLN(f, g.lnf);
    rLinear(f, g.head);
}

// ─────────────────────────────────────────────
extern(C) nothrow @trusted:

PyObject* py_ml_make(PyObject* self, PyObject* args) {
    try {
        PyObject* nm; int inputSz; PyObject* hid; PyObject* heads;
        PyObject* acts; PyObject* coss; PyObject* opt;
        PyObject* lk; PyObject* lb2; double sigma, ent;
        if (!PyArg_ParseTuple(args, "OiOOOOOOOdd", &nm, &inputSz, &lk, &hid, &lb2,
                              &heads, &acts, &coss, &opt, &sigma, &ent))
            return null;
        string name   = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        string optStr = fromStringz(PyUnicode_AsUTF8(opt)).idup;
        int[] kindsI  = pyIntList(lk);
        int[] la      = pyIntList(hid);
        int[] lbv     = pyIntList(lb2);
        auto  lkb     = new ubyte[kindsI.length];
        foreach (i, v; kindsI) lkb[i] = cast(ubyte) v;
        int[] hsz     = pyIntList(heads);
        string[][] al = pyLals(acts);
        int[] cosI    = pyIntList(coss);
        auto cosB = new bool[cosI.length];
        foreach (i, v; cosI) cosB[i] = v != 0;
        auto ai = new BlackBoxAI(name, inputSz, lkb, la, lbv, hsz, al, cosB, parseOpt(optStr),
                                 cast(float)sigma, cast(float)ent);
        GC.addRoot(cast(void*) ai);
        return PyCapsule_New(cast(void*) ai, "BlackBoxAI", &bbai_dtor);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_make"); return null; }
}

// 헤드별 [선택인덱스, 실수값] 을 평탄한 리스트로 돌려준다: [i0,v0, i1,v1, ...]
private PyObject* packPick(BlackBoxAI ai, int[] chosen, float[] value) {
    int n = ai.nHeads;
    auto lst = PyList_New(n * 2);
    foreach (h; 0..n) {
        PyList_SetItem(lst, h*2,   PyLong_FromLong(chosen[h]));
        PyList_SetItem(lst, h*2+1, PyFloat_FromDouble(cast(double) value[h]));
    }
    return lst;
}

PyObject* py_ml_pick(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &legal, &inp)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto chosen = new int[ai.nHeads];
        auto value  = new float[ai.nHeads];
        ai.pickAll(pyLals(legal), pyFloatList(inp), chosen, value);
        return packPick(ai, chosen, value);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_pick"); return null; }
}

PyObject* py_ml_predict(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &legal, &inp)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto chosen = new int[ai.nHeads];
        auto value  = new float[ai.nHeads];
        ai.predictAll(pyLals(legal), pyFloatList(inp), chosen, value);
        return packPick(ai, chosen, value);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_predict"); return null; }
}

// inputs[i], chosen[i][h], values[i][h], scores[i][h]  (score 가 NaN 이면 그 헤드는 제외)
PyObject* py_ml_learn(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inps; PyObject* chos; PyObject* vals; PyObject* scrs;
        if (!PyArg_ParseTuple(args, "OOOOO", &cap, &inps, &chos, &vals, &scrs)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        Py_ssize_t n = PyList_Size(inps);
        auto inputs = new float[][n];
        auto chosen = new int[][n];
        auto values = new float[][n];
        auto score  = new float[][n];
        foreach (i; 0..n) {
            inputs[i] = pyFloatList(PyList_GetItem(inps, i));
            chosen[i] = pyIntList(PyList_GetItem(chos, i));
            values[i] = pyFloatList(PyList_GetItem(vals, i));
            score[i]  = pyFloatList(PyList_GetItem(scrs, i));
        }
        ai.learnBatch(inputs, chosen, values, score);
        ai.save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_learn"); return null; }
}

PyObject* py_ml_sl(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inp;
        PyObject* ansI; PyObject* ansV; PyObject* useL;
        if (!PyArg_ParseTuple(args, "OOOOOO", &cap, &legal, &inp, &ansI, &ansV, &useL)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto useI = pyIntList(useL);
        auto use  = new bool[useI.length];
        foreach (i, v; useI) use[i] = v != 0;
        auto input = pyFloatList(inp);
        auto lals  = pyLals(legal);
        ai.slBatch(input, lals, pyIntList(ansI), pyFloatList(ansV), use);
        auto chosen = new int[ai.nHeads];
        auto value  = new float[ai.nHeads];
        ai.predictAll(lals, input, chosen, value);
        return packPick(ai, chosen, value);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_sl"); return null; }
}

PyObject* py_ml_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_save"); return null; }
}

// ─────────────────────────────────────────────
// LM (GPT) C API
// ─────────────────────────────────────────────
extern(C) void gpt_dtor(PyObject* cap) nothrow @trusted {
    auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
    if (g) try { GC.removeRoot(cast(void*) g); } catch (Throwable) {}
}

PyObject* py_lm_make(PyObject* self, PyObject* args) {
    try {
        PyObject* nm; int V, D, H, L, T; PyObject* opt; double lr;
        if (!PyArg_ParseTuple(args, "OiiiiiOd", &nm, &V, &D, &H, &L, &T, &opt, &lr))
            return null;
        string name   = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        string optStr = fromStringz(PyUnicode_AsUTF8(opt)).idup;
        if (D % H != 0) {
            PyErr_SetString(_pyRuntimeError, "my_ml: 폭(dim)은 헤드 수로 나누어떨어져야 합니다");
            return null;
        }
        auto g = new GPT(name, V, D, H, L, T, parseOpt(optStr), cast(float) lr);
        GC.addRoot(cast(void*) g);
        return PyCapsule_New(cast(void*) g, "GPT", &gpt_dtor);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_make"); return null; }
}

// 저장된 파일에서 불러오기. 없으면 None. 성공하면 (캡슐, 어휘목록, 설정)
PyObject* py_lm_load(PyObject* self, PyObject* args) {
    try {
        PyObject* nm;
        if (!PyArg_ParseTuple(args, "O", &nm)) return null;
        string name = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        if (!exists(name ~ "_lm.pth")) { Py_IncRef(_pyNone); return _pyNone; }
        GPT g;
        try { loadGPT(name, g); }
        catch (Exception e) { Py_IncRef(_pyNone); return _pyNone; }
        GC.addRoot(cast(void*) g);
        auto cap = PyCapsule_New(cast(void*) g, "GPT", &gpt_dtor);
        auto tup = PyList_New(2);
        PyList_SetItem(tup, 0, cap);
        auto cfg = PyList_New(5);
        PyList_SetItem(cfg, 0, PyLong_FromLong(g.V));
        PyList_SetItem(cfg, 1, PyLong_FromLong(g.D));
        PyList_SetItem(cfg, 2, PyLong_FromLong(g.H));
        PyList_SetItem(cfg, 3, PyLong_FromLong(g.L));
        PyList_SetItem(cfg, 4, PyLong_FromLong(g.maxT));
        PyList_SetItem(tup, 1, cfg);
        return tup;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_load"); return null; }
}

// 토큰 열 하나로 한 번 학습. 반환 = 손실
PyObject* py_lm_train(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* toks;
        if (!PyArg_ParseTuple(args, "OO", &cap, &toks)) return null;
        auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
        auto ids = pyIntList(toks);
        int n = cast(int) ids.length;
        if (n < 2) return PyFloat_FromDouble(0.0);
        int T = n - 1;
        if (T > g.maxT) T = g.maxT;
        auto inp = ids[0 .. T];
        auto tgt = ids[1 .. T+1];
        float l = g.trainOn(inp, tgt, T);
        return PyFloat_FromDouble(cast(double) l);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_train"); return null; }
}

// 손실만 계산 (학습 안 함)
PyObject* py_lm_loss(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* toks;
        if (!PyArg_ParseTuple(args, "OO", &cap, &toks)) return null;
        auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
        auto ids = pyIntList(toks);
        int n = cast(int) ids.length;
        if (n < 2) return PyFloat_FromDouble(0.0);
        int T = n - 1;
        if (T > g.maxT) T = g.maxT;
        g.forward(ids[0 .. T], T);
        double loss = 0;
        foreach (t_; 0..T) {
            auto lg = g.logits[t_*g.V .. t_*g.V + g.V];
            double mx = lg[0];
            foreach (v_; lg) if (v_ > mx) mx = v_;
            double sum = 0;
            foreach (i; 0..g.V) sum += exp(cast(double) lg[i] - mx);
            loss -= (cast(double) lg[ids[t_+1]] - mx) - log(sum);
        }
        return PyFloat_FromDouble(loss / T);
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_loss"); return null; }
}

// 다음 토큰 하나 뽑기
PyObject* py_lm_sample(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* toks; double temp; int topk;
        if (!PyArg_ParseTuple(args, "OOdi", &cap, &toks, &temp, &topk)) return null;
        auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
        auto ids = pyIntList(toks);
        int T = cast(int) ids.length;
        if (T == 0) { PyErr_SetString(_pyRuntimeError, "my_ml: 입력이 비었습니다"); return null; }
        if (T > g.maxT) { ids = ids[$-g.maxT .. $]; T = g.maxT; }
        return PyLong_FromLong(g.sample(ids, T, cast(float) temp, topk));
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_sample"); return null; }
}

PyObject* py_lm_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
        saveGPT(g);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_save"); return null; }
}

PyObject* py_lm_info(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto g = cast(GPT) PyCapsule_GetPointer(cap, "GPT");
        auto d = PyDict_New();
        void put(string k, long v) {
            auto o = PyLong_FromLong(v);
            PyDict_SetItemString(d, toStringz(k), o); Py_DecRef(o);
        }
        put("vocab", g.V); put("dim", g.D); put("heads", g.H);
        put("layers", g.L); put("ctx", g.maxT); put("params", g.paramCount());
        return d;
    } catch (Throwable) { PyErr_SetString(_pyRuntimeError, "my_ml: exception in _lm_info"); return null; }
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
        foreach (i, ref h; net.lins) {
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

class _Cos:
    """숫자(연속값) 출력 표시. 값의 범위는 쓰는 쪽에서 정한다."""
    __slots__ = ()
    def __repr__(self): return "cos"

cos = _Cos()
COS = cos          # 옛 이름


class _Attn:
    """어텐션 층 표시. 폭은 그대로 두고, 값들끼리 서로 참조하게 한다.

        attn(8)      -> 폭을 8조각으로 나눠 서로 참조 (헤드 1)
        attn(8, 2)   -> 조각 8, 헤드 2
    """
    __slots__ = ("items", "heads")

    def __init__(self, items=4, heads=1):
        self.items, self.heads = int(items), int(heads)

    def __call__(self, items, heads=1):
        return _Attn(items, heads)

    def __repr__(self):
        return f"attn({self.items}, {self.heads})"

attn = _Attn()


class Step:
    """rl() 이 돌려주는 (입력, 출력) 쌍. 그냥 데이터."""
    __slots__ = ("input", "output", "_units", "_raw")

    def __init__(self, inp, out, units, raw):
        self.input, self.output, self._units, self._raw = inp, out, units, raw

    def __iter__(self):        return iter((self.input, self.output))
    def __len__(self):         return 2
    def __getitem__(self, i):  return (self.input, self.output)[i]
    def __repr__(self):
        return f"Step(input={self.input!r}, output={self.output!r})"


class Scored:
    """reward() 가 돌려주는 (입력, 출력, 보상). 그냥 데이터.
    point 는 숫자 하나이거나, 헤드별 리스트(None 이면 그 헤드는 학습 제외)."""
    __slots__ = ("input", "output", "point", "_units", "_raw")

    def __init__(self, inp, out, point, units, raw):
        self.input, self.output, self.point = inp, out, point
        self._units, self._raw = units, raw

    def __iter__(self):        return iter((self.input, self.output, self.point))
    def __len__(self):         return 3
    def __getitem__(self, i):  return (self.input, self.output, self.point)[i]
    def __repr__(self):
        return f"Scored(input={self.input!r}, output={self.output!r}, point={self.point!r})"


_NAN = float("nan")


class BlackBoxAI:
    def __init__(self, h, name, heads):
        self._h     = h
        self._name  = name
        self._heads = heads          # [(actions or None, cos:bool), ...]
        self._n     = len(heads)

    # 헤드별 원시값 → 사람이 쓰는 출력
    def _decode(self, flat):
        out, units, raw = [], [], []
        for i, (acts, cos) in enumerate(self._heads):
            idx, val = int(flat[i*2]), float(flat[i*2+1])
            units.append(idx)
            raw.append(val)
            out.append(val if cos else acts[idx])
        return out, units, raw

    def _legal_arg(self, legal):
        if legal is None:
            return [[] for _ in range(self._n)]
        if self._n == 1 and legal and isinstance(legal[0], str):
            return [list(legal)]
        return [list(x) if x else [] for x in legal]

    def _one(self, lst):
        return lst[0] if self._n == 1 else lst

    # ── 순수: (입력, 출력) 데이터만 ──
    def rl(self, input_list, legal=None):
        inp  = [float(x) for x in input_list]
        flat = _ml_pick(self._h, self._legal_arg(legal), inp)
        out, units, raw = self._decode(flat)
        return Step(inp, self._one(out), units, raw)

    # ── 순수: 보상 붙이기 ──
    def reward(self, data, point):
        return Scored(data.input, data.output, point, data._units, data._raw)

    # ── 여기서만 학습 + 저장 ──
    def save(self, scored=None):
        if scored is None:
            _ml_save(self._h); return 0
        batch = [scored] if isinstance(scored, Scored) else list(scored)
        if not batch:
            return 0
        inputs, chosen, values, points = [], [], [], []
        for s in batch:
            inputs.append([float(x) for x in s.input])
            chosen.append([int(u) for u in s._units])
            outs = s.output if self._n > 1 else [s.output]
            values.append([float(v) if c else 0.0
                           for v, (_, c) in zip(s._raw, self._heads)])
            p = s.point
            if isinstance(p, (list, tuple)):
                points.append([_NAN if x is None else float(x) for x in p])
            else:
                points.append([float(p)] * self._n)
        _ml_learn(self._h, inputs, chosen, values, points)
        return len(batch)

    # ── 지도학습 ──
    def sl(self, input_list, answer=None, legal=None):
        inp  = [float(x) for x in input_list]
        lals = self._legal_arg(legal)
        ansI = [0] * self._n
        ansV = [0.0] * self._n
        use  = [0] * self._n
        if answer is not None:
            answers = answer if (self._n > 1 and isinstance(answer, (list, tuple))) else [answer]
            for i, a in enumerate(answers):
                if a is None or i >= self._n: continue
                acts, cos = self._heads[i]
                use[i] = 1
                if cos: ansV[i] = float(a)
                else:   ansI[i] = acts.index(a)
        flat = _ml_sl(self._h, lals, inp, ansI, ansV, use)
        out, _, _ = self._decode(flat)
        return self._one(out)

    # ── 예측(샘플링 없음) ──
    def predict(self, input_list, legal=None):
        inp  = [float(x) for x in input_list]
        flat = _ml_predict(self._h, self._legal_arg(legal), inp)
        out, _, _ = self._decode(flat)
        return self._one(out)

    # ── 스텝 묶음에 같은 보상 (순수) ──
    def episode(self, steps, point):
        return [self.reward(s, point) for s in steps]

    @property
    def heads(self):    return self._n
    @property
    def actions(self):  return self._one([a for a, _ in self._heads])


def _헤드해석(spec):
    """스펙 하나 → (액션목록 또는 None, cos여부, 출력개수)"""
    if isinstance(spec, _Cos):
        return None, True, 1
    if isinstance(spec, str) and spec.lower() == "cos":
        return None, True, 1
    if isinstance(spec, (list, tuple)):
        names = list(spec)
        if len(names) < 2:
            raise ValueError(f"고를 것이 2개 이상이어야 합니다: {names}")
        if not all(isinstance(a, str) for a in names):
            raise ValueError(f"액션 이름은 문자열이어야 합니다: {names}")
        return names, False, len(names)
    raise ValueError(f"알 수 없는 출력 스펙: {spec!r}  (액션 리스트 또는 cos)")


def _헤드스펙인가(x):
    return isinstance(x, (_Cos, list, tuple)) or (isinstance(x, str) and x.lower() == "cos")


def make(model_name, layers, outputs, optimizer='adam', sigma=1.0, entropy=0.01):
    """
    model_name : 모델 이름 (가중치 파일명)
    layers     : [입력수, 은닉...]   출력은 outputs 에서 정해진다
                 은닉 자리에 attn(조각수) 를 넣으면 어텐션 층이 된다
                 예) [38, 128, attn(8), 128]
    outputs    : 출력 하나면  ["A","B"]  또는  cos
                 여러 개면   [cos, ["A","B"], cos, ...]
                 cos 값의 범위는 쓰는 쪽에서 알아서 정한다
    sigma      : cos 가 값을 얼마나 넓게 탐험할지 (기본 1.0)
    entropy    : 고르는 쪽이 한 답으로 굳는 것을 막는 힘 (기본 0.01)
    """
    if len(layers) < 1:
        raise ValueError("layers 는 [입력수, 은닉...] 형태입니다")
    inputSz = int(layers[0])
    lay_kind, lay_a, lay_b = [], [], []
    폭 = inputSz
    for i, L in enumerate(layers[1:], 1):
        if isinstance(L, _Attn):
            if 폭 % L.items:
                raise ValueError(
                    f"{i}번째 층 attn({L.items}): 폭 {폭} 이 조각 {L.items} 로 나뉘지 않습니다")
            조각폭 = 폭 // L.items
            if 조각폭 % L.heads:
                raise ValueError(
                    f"{i}번째 층 attn: 조각폭 {조각폭} 이 헤드 {L.heads} 로 나뉘지 않습니다")
            lay_kind.append(1); lay_a.append(L.items); lay_b.append(L.heads)
            # 폭 그대로
        else:
            폭 = int(L)
            lay_kind.append(0); lay_a.append(폭); lay_b.append(0)

    # 출력 스펙을 헤드 목록으로
    if _헤드스펙인가(outputs) and isinstance(outputs, (list, tuple))             and not isinstance(outputs, _Cos) and outputs and all(_헤드스펙인가(o) for o in outputs):
        specs = list(outputs)          # 헤드 여러 개
    else:
        specs = [outputs]              # 헤드 하나

    heads, al_arg, cos_arg, sizes = [], [], [], []
    for spec in specs:
        names, is_cos, n = _헤드해석(spec)
        heads.append((names, is_cos))
        al_arg.append(names or [])
        cos_arg.append(1 if is_cos else 0)
        sizes.append(n)

    h = _ml_make(model_name, inputSz, lay_kind, lay_a, lay_b, sizes, al_arg, cos_arg,
                 optimizer, float(sigma), float(entropy))
    return BlackBoxAI(h, model_name, heads)


class LM:
    """토큰 열을 이어 쓰는 모델 (트랜스포머).

    토큰은 0 이상 vocab 미만의 정수다. 글자/단어를 번호로 바꾸는 건 쓰는 쪽 몫이다.

        chars = sorted(set(텍스트))
        번호  = {c: i for i, c in enumerate(chars)}

        lm = make_lm("내모델", vocab=len(chars))
        lm.sl([번호[c] for c in 텍스트], steps=2000)
        나온것 = lm.gen([번호["안"]], 40)
        print("".join(chars[i] for i in 나온것))
    """

    def __init__(self, name, vocab, dim=128, layers=4, heads=4, ctx=64,
                 optimizer='adam', lr=3e-4):
        self._name = name
        self._h    = None
        self._cfg  = dict(vocab=int(vocab), dim=dim, layers=layers, heads=heads,
                          ctx=ctx, optimizer=optimizer, lr=lr)

        got = _lm_load(name)
        if got:
            self._h, cfg = got[0], got[1]
            self._cfg.update(vocab=cfg[0], dim=cfg[1], heads=cfg[2],
                             layers=cfg[3], ctx=cfg[4])
            if cfg[0] != int(vocab):
                print(f" [{name}] 저장된 어휘 {cfg[0]} 이 요청한 {vocab} 과 다릅니다. 저장된 쪽을 씁니다.")
            print(f" [{name}] 이전 학습 데이터를 불러왔습니다.")

    def _준비(self):
        if self._h is None:
            c = self._cfg
            self._h = _lm_make(self._name, c['vocab'], c['dim'], c['heads'],
                               c['layers'], c['ctx'], c['optimizer'], float(c['lr']))
            n = _lm_info(self._h)['params']
            print(f" [{self._name}] 새로 만들었습니다. "
                  f"어휘 {c['vocab']}, 파라미터 {n:,}개")

    def _확인(self, ids):
        V = self._cfg['vocab']
        out = [int(t) for t in ids]
        for t in out:
            if not (0 <= t < V):
                raise ValueError(f"토큰 {t} 가 어휘 범위(0~{V-1}) 밖입니다")
        return out

    # ── 학습 ──────────────────────────────────
    def sl(self, tokens, steps=None, log=None):
        """토큰 열로 학습. 정답은 '다음 토큰' 이라 따로 줄 필요가 없다.
        steps 를 주면 그만큼 무작위 조각을 뽑아 학습한다."""
        import random
        self._준비()
        ids = self._확인(tokens)
        if len(ids) < 2:
            raise ValueError("토큰이 2개 이상 필요합니다")
        조각 = self._cfg['ctx'] + 1

        if steps is None:
            걸음 = max(1, self._cfg['ctx'] // 2)
            자리 = list(range(0, max(1, len(ids) - 조각 + 1), 걸음))
        else:
            상한 = max(1, len(ids) - 조각)
            자리 = [random.randrange(상한) for _ in range(steps)]

        총, 수 = 0.0, 0
        for s in 자리:
            묶음 = ids[s:s + 조각]
            if len(묶음) < 2:
                continue
            총 += _lm_train(self._h, 묶음)
            수 += 1
            if log and 수 % log == 0:
                print(f"   {수}/{len(자리)}  손실 {총/수:.4f}")
        return 총 / max(1, 수)

    def loss(self, tokens):
        """학습 없이 손실만 (낮을수록 잘 맞힘)"""
        self._준비()
        ids = self._확인(tokens)[:self._cfg['ctx'] + 1]
        if len(ids) < 2:
            return 0.0
        return _lm_loss(self._h, ids)

    # ── 생성 ──────────────────────────────────
    def gen(self, tokens, n=100, temp=1.0, topk=0):
        """tokens 뒤에 n 개를 이어 만든다. 새로 만든 것만 반환한다.
        temp 낮으면 뻔하게, 높으면 엉뚱하게. topk>0 이면 상위 k개 중에서만."""
        import random
        self._준비()
        ids = self._확인(tokens)
        if not ids:
            ids = [random.randrange(self._cfg['vocab'])]
        나온것 = []
        for _ in range(n):
            다음 = _lm_sample(self._h, ids, float(temp), int(topk))
            나온것.append(다음)
            ids.append(다음)
            if len(ids) > self._cfg['ctx']:
                ids = ids[-self._cfg['ctx']:]
        return 나온것

    def save(self):
        self._준비()
        _lm_save(self._h)
        print(f" [{self._name}] 저장 완료.")

    @property
    def info(self):
        self._준비()
        return _lm_info(self._h)


def make_lm(name, vocab, dim=128, layers=4, heads=4, ctx=64,
            optimizer='adam', lr=3e-4):
    """
    name      : 모델 이름 (가중치 파일 이름_lm.pth)
    vocab     : 토큰 종류 수. 토큰은 0 ~ vocab-1 의 정수
    dim       : 폭. heads 로 나누어떨어져야 한다
    layers    : 층 수
    heads     : 어텐션 헤드 수
    ctx       : 한 번에 보는 토큰 수
    """
    if dim % heads:
        raise ValueError(f"dim({dim}) 은 heads({heads}) 로 나누어떨어져야 합니다")
    if int(vocab) < 2:
        raise ValueError("vocab 은 2 이상이어야 합니다")
    return LM(name, vocab, dim, layers, heads, ctx, optimizer, lr)


def change(model_name):
    """예전 버전에서 만든 가중치 파일을 지금 포맷으로 바꿉니다.
    원본은 .bak 으로 남습니다."""
    path = model_name if model_name.endswith(".pth") else f"{model_name}_ml_memory.pth"
    if not os.path.exists(path):
        print(f" [{model_name}] {path} 가 없습니다.")
        return False
    r = _ml_change(path)
    if r == "already":
        print(f" [{model_name}] 이미 최신 포맷입니다.")
        return False
    print(f" [{model_name}] 변환 완료 -> {r}   (원본: {path}.bak)")
    return True


def gc_disable(): _ml_gc_disable()
def gc_collect(): _ml_gc_collect()
def resset(model_name): _ml_resset(model_name)

import builtins
builtins.resset     = resset
builtins.change     = change
builtins.make_lm    = make_lm
builtins.gc_disable = gc_disable
builtins.gc_collect = gc_collect
`;

// ─────────────────────────────────────────────
// Module init
// ─────────────────────────────────────────────
private __gshared PyMethodDef[20] _methods;
private __gshared PyModuleDef     _moddef;

extern(C) PyObject* PyInit_my_ml() nothrow @trusted {
    try {
        _methods[0 ] = PyMethodDef("_ml_make",            &py_ml_make,            METH_VARARGS, null);
        _methods[1 ] = PyMethodDef("_ml_pick",            &py_ml_pick,            METH_VARARGS, null);
        _methods[2 ] = PyMethodDef("_ml_predict",         &py_ml_predict,         METH_VARARGS, null);
        _methods[3 ] = PyMethodDef("_ml_learn",           &py_ml_learn,           METH_VARARGS, null);
        _methods[4 ] = PyMethodDef("_ml_sl",              &py_ml_sl,              METH_VARARGS, null);
        _methods[5 ] = PyMethodDef("_ml_save",            &py_ml_save,            METH_VARARGS, null);
        _methods[6 ] = PyMethodDef("_ml_resset",          &py_ml_resset,          METH_VARARGS, null);
        _methods[7 ] = PyMethodDef("_ml_gc_disable",      &py_ml_gc_disable,      METH_VARARGS, null);
        _methods[8 ] = PyMethodDef("_ml_gc_collect",      &py_ml_gc_collect,      METH_VARARGS, null);
        _methods[9 ] = PyMethodDef("_ml_export_weights",  &py_ml_export_weights,  METH_VARARGS, null);
        _methods[10] = PyMethodDef("_ml_get_meta",        &py_ml_get_meta,        METH_VARARGS, null);
        _methods[11] = PyMethodDef("_ml_change",          &py_ml_change,          METH_VARARGS, null);
        _methods[12] = PyMethodDef("_lm_make",            &py_lm_make,            METH_VARARGS, null);
        _methods[13] = PyMethodDef("_lm_load",            &py_lm_load,            METH_VARARGS, null);
        _methods[14] = PyMethodDef("_lm_train",           &py_lm_train,           METH_VARARGS, null);
        _methods[15] = PyMethodDef("_lm_loss",            &py_lm_loss,            METH_VARARGS, null);
        _methods[16] = PyMethodDef("_lm_sample",          &py_lm_sample,          METH_VARARGS, null);
        _methods[17] = PyMethodDef("_lm_save",            &py_lm_save,            METH_VARARGS, null);
        _methods[18] = PyMethodDef("_lm_info",            &py_lm_info,            METH_VARARGS, null);
        _methods[19] = PyMethodDef(null, null, 0, null);

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
