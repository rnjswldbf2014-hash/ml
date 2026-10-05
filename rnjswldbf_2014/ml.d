// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 rnjswldbf2014-hash
module ml;

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
import std.math      : exp, sqrt, pow, log, cos, tanh, PI, abs;
import std.random    : Random, uniform, uniform01, unpredictableSeed;
import std.file      : exists, remove, rename;
import std.algorithm : countUntil, min;
import std.conv      : to;
import std.process   : environment;
import std.parallelism : taskPool, defaultPoolThreads, totalCPUs, parallel;
import std.range     : iota;
import gpucl;

__gshared bool _noBatch = false;
// 묶음을 하나의 사슬로 보고 시간을 거슬러 올라간다 (BPTT). **기본 켜짐.**
// 1스텝 절단이 진짜 기울기와 얼마나 다른지 재봤는데 최대 기울기의 60% 였다 —
// 대안이 아니라 그냥 틀린 값이다. MYML_BPTT=0 으로 예전 동작(1스텝 절단)으로
// 돌릴 수 있고, 그건 "묶음 == 직렬" 동치 검증용이다 (직렬 경로는 순전파와
// 역전파를 샘플마다 번갈아 하므로 사슬 전체를 들고 있을 수가 없어서 BPTT 가
// 구조적으로 불가능하다. 그래서 둘을 비교할 때는 양쪽 다 꺼야 한다).
__gshared bool _bptt = true;
__gshared int  _nThreads = 1;
// GPU 사용 여부: "0"=완전 비활성(OpenCL.dll 프로브도 안 함), "1"=강제(문턱값 무시),
// "auto"=문턱값 넘는 큰 배치에서만 지연 프로브.
//
// 기본값은 "0" 이다. 지금 GPU 경로는 이 기계(AMD gfx1035 내장 GPU)에서 CPU 배치
// 경로보다 17~25배 "느리다" — 커널이 타일링도 병합 접근도 없는 순진한 행렬곱이라
// 이론 성능의 0.2% 밖에 못 낸다. 예전엔 기본이 "auto" 여서, OpenCL 이 깔린 기계에서
// 묶음이 문턱값을 넘으면 사용자가 영문도 모르고 20배 느려졌다. 결과는 맞으므로
// 알아챌 방법도 없었다. 어디서든 빠르다는 걸 보이기 전까지는 켜지 않는다.
// 켜보려면 MYML_GPU=1 (강제) 또는 MYML_GPU=auto (문턱값).
__gshared string _gpuMode = "0";
__gshared long   _gpuMinFlops = 50_000_000;
__gshared int    _gpuMinB = 64;
shared static this() {
    try { _noBatch = environment.get("MYML_NOBATCH", "") == "1"; } catch (Exception e) {}
    try { _bptt = environment.get("MYML_BPTT", "1") != "0"; } catch (Exception e) {}
    try {
        auto s = environment.get("MYML_THREADS", "");
        _nThreads = (s.length ? s.to!int : totalCPUs);
    } catch (Exception e) { _nThreads = totalCPUs; }
    try {
        auto g = environment.get("MYML_GPU", "");
        if (g.length) _gpuMode = g;
    } catch (Exception e) {}
    try {
        auto f = environment.get("MYML_GPU_MIN_FLOPS", "");
        if (f.length) _gpuMinFlops = f.to!long;
    } catch (Exception e) {}
    try {
        auto b = environment.get("MYML_GPU_MIN_B", "");
        if (b.length) _gpuMinB = b.to!int;
    } catch (Exception e) {}
    if (_nThreads < 1) _nThreads = 1;
    try { defaultPoolThreads(cast(size_t)(_nThreads > 0 ? _nThreads - 1 : 0)); } catch (Exception e) {}
}

// 병렬화 문턱값 — 일이 이보다 작으면 스레드를 띄우는 비용이 일보다 크다.
//
// 예전엔 **행 개수만** 봤다 (n < 2*T). 행 하나가 얼마나 무거운지는 안 봤다는
// 뜻인데, 가중치 갱신이 벡터화돼서 10배 싸진 뒤로 그게 손해로 바뀌었다.
// 실측 (배치 1, [입력,은닉...] 한 장 학습, 12스레드 vs 1스레드):
//
//     가중치 행렬   1스레드    12스레드
//        2,048      0.0087     0.0311   <- 3.6배 손해
//       16,384      0.0396     0.0782   <- 2.0배 손해
//       65,536      0.1862     0.2332
//      262,144      1.2692     0.7705   <- 여기서 뒤집힌다
//    1,048,576      4.6884     4.2656
//
// 그래서 131,072 로 잡았다 (65,536 과 262,144 사이). 묶음으로 학습할 때는
// 일이 B 배라 어느 크기든 문턱값을 넘는다 (실측 B=64 에서 3~5배 이득).
//
// 문턱값은 **결과를 바꾸지 않는다** — 조각들이 겹치지 않는 출력을 담당하므로
// 병렬이든 직렬이든 합산 순서가 같다. regression.py 가 그걸 지킨다.
enum long PAR_MIN_WORK = 131_072;

// [0,n) 을 _nThreads 조각으로 나눠 병렬 실행. 각 조각은 겹치지 않는 출력을 담당한다
// (합산 순서가 보존되므로 직렬과 비트 단위로 동일하다).
// work = 이 루프가 건드리는 원소 수. 안 주면 "충분히 크다" 로 본다.
void _parChunk(int n, scope void delegate(int lo, int hi) body) {
    _parChunk(n, long.max, body);
}

void _parChunk(int n, long work, scope void delegate(int lo, int hi) body) {
    int T = _nThreads;
    if (T <= 1 || n < 2 * T || work < PAR_MIN_WORK) { body(0, n); return; }
    int per = (n + T - 1) / T;
    foreach (t; taskPool.parallel(iota(T), 1)) {
        int lo = cast(int)t * per; int hi = lo + per;
        if (hi > n) hi = n;
        if (lo < hi) body(lo, hi);
    }
}

// nothrow 컨텍스트용 _parChunk. std.parallelism 자체는 nothrow 가 아니라서
// (그리고 Task 할당 때문에 @nogc 도 아니라서), 병렬 실행 중 예외가 나면
// 그 청크는 포기하지 않고 직렬로 다시 돌려 결과를 보존한다.
void _parChunkNT(int n, scope void delegate(int lo, int hi) nothrow body) nothrow {
    try { _parChunk(n, long.max, body); } catch (Throwable) { body(0, n); }
}

void _parChunkNT(int n, long work, scope void delegate(int lo, int hi) nothrow body) nothrow {
    try { _parChunk(n, work, body); } catch (Throwable) { body(0, n); }
}

private Random rng;
static this() { rng = Random(unpredictableSeed); }

// 안내 메시지. 출력 실패를 삼킨다.
//
// stdout 이 막혀 있을 수 있다 — 파이프가 닫혔다거나, 부모 프로세스가 출력을
// 안 읽어서 버퍼가 찼다거나. 그때 writefln 은 예외를 던지는데, 예전엔 그게
// 생성자 밖으로 그대로 나가서 "모델 생성 실패" 가 됐다. 안내 한 줄 때문에
// 모델을 못 만드는 건 말이 안 된다.
private void 알림(Args...)(string fmt, Args args) nothrow {
    try { writefln(fmt, args); } catch (Throwable) {}
}

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

@target("avx512f") private float dot_avx512(const(float)[] a, const(float)[] b) pure nothrow @nogc
{ float s = 0f; foreach (i; 0..a.length) s += a[i]*b[i]; return s; }

@target("avx2,fma") private void saxpy_avx2 (float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }
@target("avx512f") private void saxpy_avx512(float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }
@target("sse4.1")   private void saxpy_sse41(float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }
                    private void saxpy_base (float[] d, const(float)[] s, float sc) pure nothrow @nogc
{ foreach (i; 0..d.length) d[i] += sc*s[i]; }

// ── 옵티마이저 커널 (행 하나씩) ──────────────────────────────────────
// 요점은 식이 아니라 **행 슬라이스를 미리 뽑아서 넘긴다**는 것이다. 내부 루프
// 안에서 w[j][k] 처럼 간접참조하면 LLVM 이 w/m/v/grad 의 행들이 겹치지 않는다는
// 걸 증명할 수 없어서 벡터화를 포기한다. 실측 (가중치 65,536개, 1스레드, Adam):
//
//     지금처럼 w[j][k]              368 us
//     @target("avx2,fma") 만 붙임   309 us   <- 1.09배. 거의 효과 없다
//     행을 루프 밖으로 빼기          70 us   <- 5.3배. 여기가 거의 전부다
//     + @target("avx2,fma")          34 us   <- 뺀 뒤에야 AVX2 가 먹는다
//
// 식은 한 글자도 바꾸지 않았다 — 나눗셈을 미리 계산한 역수로 바꾸면 29us 까지
// 가지만(16% 더) 결과값이 미세하게 달라진다. 곱해서 10.7배면 충분하다.
//
// 이 기계(znver3 = Zen 3+)는 AVX-512 를 지원하지 않는다 (LLVM 이 -avx512f 로
// 보고한다). AVX2 가 상한이다. SSE4.1 변종은 두지 않았다 — 호이스트만 하면
// 기본 타깃(SSE2)도 4개씩 묶어서 70us 를 내므로 중간 단계가 의미가 없다.
private enum float OB1=0.9f, OB2=0.999f, OEPS=1e-8f, ORHO=0.99f;

private enum string ADAM_ROW = q{
    foreach (k; 0 .. w.length) {
        float g = gw[k];
        float mm = OB1*m[k] + (1-OB1)*g;      m[k] = mm;
        float vv = OB2*v[k] + (1-OB2)*g*g;    v[k] = vv;
        w[k] -= lr * (mm/bc1) / (sqrt(vv/bc2) + OEPS);
    }
};
private enum string RMS_ROW = q{
    foreach (k; 0 .. w.length) {
        float g = gw[k];
        float vv = ORHO*v[k] + (1-ORHO)*g*g;  v[k] = vv;
        w[k] -= lr * g / (sqrt(vv) + OEPS);
    }
};
private enum string ADAGRAD_ROW = q{
    foreach (k; 0 .. w.length) {
        float g = gw[k];
        float vv = v[k] + g*g;                v[k] = vv;
        w[k] -= lr * g / (sqrt(vv) + OEPS);
    }
};
private enum string SGD_ROW = q{
    foreach (k; 0 .. w.length) w[k] -= lr * gw[k];
};

private alias AdamFn = void function(float[] w, float[] m, float[] v,
                                     const(float)[] gw, float lr, float bc1, float bc2) nothrow @nogc;
private alias VFn    = void function(float[] w, float[] v,
                                     const(float)[] gw, float lr) nothrow @nogc;
private alias SgdFn  = void function(float[] w, const(float)[] gw, float lr) nothrow @nogc;

@target("avx2,fma") private void adamRow_avx2(float[] w, float[] m, float[] v,
    const(float)[] gw, float lr, float bc1, float bc2) nothrow @nogc { mixin(ADAM_ROW); }
@target("avx512f")  private void adamRow_avx512(float[] w, float[] m, float[] v,
    const(float)[] gw, float lr, float bc1, float bc2) nothrow @nogc { mixin(ADAM_ROW); }
                    private void adamRow_base(float[] w, float[] m, float[] v,
    const(float)[] gw, float lr, float bc1, float bc2) nothrow @nogc { mixin(ADAM_ROW); }

@target("avx2,fma") private void rmsRow_avx2(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(RMS_ROW); }
@target("avx512f")  private void rmsRow_avx512(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(RMS_ROW); }
                    private void rmsRow_base(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(RMS_ROW); }

@target("avx2,fma") private void adagradRow_avx2(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(ADAGRAD_ROW); }
@target("avx512f")  private void adagradRow_avx512(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(ADAGRAD_ROW); }
                    private void adagradRow_base(float[] w, float[] v,
    const(float)[] gw, float lr) nothrow @nogc { mixin(ADAGRAD_ROW); }

@target("avx2,fma") private void sgdRow_avx2(float[] w, const(float)[] gw, float lr)
    nothrow @nogc { mixin(SGD_ROW); }
@target("avx512f")  private void sgdRow_avx512(float[] w, const(float)[] gw, float lr)
    nothrow @nogc { mixin(SGD_ROW); }
                    private void sgdRow_base(float[] w, const(float)[] gw, float lr)
    nothrow @nogc { mixin(SGD_ROW); }

@target("avx2,fma") private void scaleRow_avx2(float[] w, float sc) nothrow @nogc
{ foreach (k; 0 .. w.length) w[k] *= sc; }
@target("avx512f")  private void scaleRow_avx512(float[] w, float sc) nothrow @nogc
{ foreach (k; 0 .. w.length) w[k] *= sc; }
                    private void scaleRow_base(float[] w, float sc) nothrow @nogc
{ foreach (k; 0 .. w.length) w[k] *= sc; }

private __gshared string _simd = "base";   // 실제로 고른 SIMD 수준
private __gshared AdamFn _adamRow;
private __gshared VFn    _rmsRow, _adagradRow;
private __gshared SgdFn  _sgdRow;
private __gshared void function(float[], float) nothrow @nogc _scaleRow;

shared static this() {
    import cpu = core.cpuid;
    // ── SIMD 수준 선택 ───────────────────────────────────────────────
    // MYML_SIMD 로 내려서 쓸 수 있다 (base|sse|avx2|avx512). 올려 쓰는 건 막는다 —
    // 하드웨어가 없는데 강제하면 바로 폴트다.
    //
    // **avx512 는 반드시 cpu.avx2 와 함께 확인한다.** core.cpuid 의 avx512f 는
    // CPUID 비트만 보고 **OS 가 ZMM 상태를 켰는지(XCR0)는 안 본다** — avx2 쪽은
    // 보는데 avx512f 쪽은 빠져 있다. 비트만 믿고 ZMM 을 쓰면 OS 가 안 켜준
    // 환경에서 #UD 로 죽는다. avx2 가 켜져 있으면 XCR0 의 SSE/YMM 이 켜진 것이고,
    // 실제 시스템에서 ZMM 상태는 그것들과 같이 켜진다. 완벽한 보장은 아니라서
    // MYML_SIMD=avx2 로 내릴 길을 같이 열어뒀다.
    immutable bool okAvx512 = cpu.avx2 && cpu.avx512f;
    immutable bool okAvx2   = cpu.avx2;
    immutable bool okSse    = cpu.sse42;

    string want;
    try { want = environment.get("MYML_SIMD", ""); } catch (Exception e) { want = ""; }
    if      (want == "base")                  { _simd = "base"; }
    else if (want == "sse"    && okSse)       { _simd = "sse"; }
    else if (want == "avx2"   && okAvx2)      { _simd = "avx2"; }
    else if (want == "avx512" && okAvx512)    { _simd = "avx512"; }
    else if (okAvx512)                        { _simd = "avx512"; }
    else if (okAvx2)                          { _simd = "avx2"; }
    else if (okSse)                           { _simd = "sse"; }
    else                                      { _simd = "base"; }

    switch (_simd) {
        case "avx512":
            _dot = &dot_avx512; _saxpy = &saxpy_avx512;
            _adamRow = &adamRow_avx512; _rmsRow = &rmsRow_avx512;
            _adagradRow = &adagradRow_avx512; _sgdRow = &sgdRow_avx512;
            _scaleRow = &scaleRow_avx512;
            break;
        case "avx2":
            _dot = &dot_avx2; _saxpy = &saxpy_avx2;
            _adamRow = &adamRow_avx2; _rmsRow = &rmsRow_avx2;
            _adagradRow = &adagradRow_avx2; _sgdRow = &sgdRow_avx2;
            _scaleRow = &scaleRow_avx2;
            break;
        case "sse":
            _dot = &dot_sse41; _saxpy = &saxpy_sse41;
            goto default;          // 옵티마이저 커널은 sse 변종을 두지 않았다
        default:
            if (_dot is null)  { _dot = &dot_base; _saxpy = &saxpy_base; }
            _adamRow = &adamRow_base; _rmsRow = &rmsRow_base;
            _adagradRow = &adagradRow_base; _sgdRow = &sgdRow_base;
            _scaleRow = &scaleRow_base;
            break;
    }
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

    // decay = 가중치 감쇠. 0 이면 아무 일도 안 한다 (기본).
    //
    // 가중치 루프는 **행마다 커널을 부른다** (위의 _adamRow 등). 행 슬라이스를
    // 미리 뽑아서 넘기는 게 요점이다 — 예전엔 여기서 w[j][k] 로 직접 훑었고
    // 그게 벡터화를 막아서 Adam 갱신이 10.7배 느렸다. 식은 그대로라 결과값은
    // 비트 단위로 같다. 편향(b)은 outSz 개뿐이라 그대로 둔다.
    void step(Opt opt, float lr, float decay = 0f) nothrow {
        enum float B1=0.9f, B2=0.999f, EPS=1e-8f, RHO=0.99f;
        final switch (opt) {
            case Opt.adam:
                t++;
                float bc1 = 1f - B1^^t, bc2 = 1f - B2^^t;
                _parChunkNT(outSz, cast(long) outSz * inSz, (int jlo, int jhi) nothrow {
                    foreach (j; jlo .. jhi) {
                        _adamRow(w[j], mW[j], vW[j], gradW[j], lr, bc1, bc2);
                        float gb = gradB[j];
                        mB[j] = B1*mB[j] + (1-B1)*gb; vB[j] = B2*vB[j] + (1-B2)*gb*gb;
                        b[j] -= lr * (mB[j]/bc1) / (sqrt(vB[j]/bc2) + EPS);
                    }
                });
                break;
            case Opt.sgd:
                _parChunkNT(outSz, cast(long) outSz * inSz, (int jlo, int jhi) nothrow {
                    foreach (j; jlo .. jhi) {
                        _sgdRow(w[j], gradW[j], lr);
                        b[j] -= lr * gradB[j];
                    }
                });
                break;
            case Opt.rmsprop:
                _parChunkNT(outSz, cast(long) outSz * inSz, (int jlo, int jhi) nothrow {
                    foreach (j; jlo .. jhi) {
                        _rmsRow(w[j], vW[j], gradW[j], lr);
                        float gb = gradB[j];
                        vB[j] = RHO*vB[j] + (1-RHO)*gb*gb;
                        b[j] -= lr * gb / (sqrt(vB[j]) + EPS);
                    }
                });
                break;
            case Opt.adagrad:
                _parChunkNT(outSz, cast(long) outSz * inSz, (int jlo, int jhi) nothrow {
                    foreach (j; jlo .. jhi) {
                        _adagradRow(w[j], vW[j], gradW[j], lr);
                        float gb = gradB[j];
                        vB[j] += gb*gb;
                        b[j] -= lr * gb / (sqrt(vB[j]) + EPS);
                    }
                });
                break;
        }

        // 가중치 감쇠. 기울기에 더하지 않고 따로 곱한다 (AdamW 가 하는 방식) —
        // Adam 은 기울기를 제 나름으로 정규화하므로, 거기에 감쇠를 섞으면 실제로
        // 걸리는 힘이 가중치마다 달라져 예측이 안 된다.
        //
        // 편향(b)에는 안 건다. 편향은 "전체적으로 얼마나 밀어올릴지" 라서 0 으로
        // 당기면 맞춰야 할 값을 못 맞춘다 (관례도 그렇다).
        if (decay != 0f) {
            immutable float keep = 1f - lr * decay;
            _parChunkNT(outSz, cast(long) outSz * inSz, (int jlo, int jhi) nothrow {
                foreach (j; jlo .. jhi) _scaleRow(w[j], keep);
            });
        }
    }

    // ── 배치 코어 (count 개 항목 — Network 의 샘플 배치든 EachLayer 의 항목 배치든) ──
    // pre[c*outSz+j] = b[j] + dot(w[j], xs[c*inSz..]).  j 기준 병렬 (w[j] 재사용).
    void batchForward(const(float)[] xs, float[] pre, int count) {
        _parChunk(outSz, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) {
                auto wj = w[j]; float bj = b[j];
                foreach (c; 0..count)
                    pre[c*outSz + j] = bj + _dot(wj, xs[c*inSz .. c*inSz + inSz]);
            }
        });
    }

    // dZ(count*outSz, 활성함수까지 적용된 기울기) -> gradW/gradB 누적, 선택적으로 dIn 전파.
    // ① gradW: j 기준 병렬 (gradW[j] 재사용).  ② dIn: c 기준 병렬 (w[j] 스트리밍).
    // zeroDIn=false 면 dIn 을 먼저 비우지 않고 누적만 한다 (여러 헤드가 같은 dHout
    // 버퍼를 공유해서 더해 넣어야 할 때 — 그 경우 호출자가 미리 한 번 0으로 채워둔다).
    void batchAccum(const(float)[] xs, const(float)[] dZ, float[] dIn, int count,
                     bool needDIn = true, bool zeroDIn = true) {
        _parChunk(outSz, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) {
                auto gj = gradW[j]; float gb = 0f;
                foreach (c; 0..count) {
                    float d = dZ[c*outSz + j];
                    if (d == 0f) continue;
                    _saxpy(gj, xs[c*inSz .. c*inSz+inSz], d);
                    gb += d;
                }
                gradB[j] += gb;
            }
        });
        if (!needDIn) return;
        _parChunk(count, (int clo, int chi) {
            foreach (c; clo .. chi) {
                auto o = dIn[c*inSz .. c*inSz+inSz];
                if (zeroDIn) o[] = 0f;
                foreach (j; 0..outSz) {
                    float d = dZ[c*outSz + j];
                    if (d != 0f) _saxpy(o, w[j], d);
                }
            }
        });
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

    this(int d) {
        D = d;
        g = new float[d]; b = new float[d];
        gg = new float[d]; gb = new float[d];
        mg = new float[d]; vg = new float[d];
        mb = new float[d]; vb = new float[d];
        // D 는 new float[] 를 NaN 으로 채운다. 반드시 직접 0 을 넣어야 한다.
        gg[] = 0f; gb[] = 0f; mg[] = 0f; vg[] = 0f; mb[] = 0f; vb[] = 0f;
        foreach (i; 0..d) { g[i] = 1f; b[i] = 0f; }
    }

    // mu, rstd, xh 는 전부 호출자(AttnLayer) 소유의 스크래치다 (per-sample 용과
    // fwdBatch/bwdBatch 용이 서로 다른 버퍼 — 절대 안 섞이도록 여기서는 그냥 넘겨받기만
    // 한다. 예전에는 LN 이 내부에서 알아서 키우는 공유 버퍼였는데, per-sample 경로가
    // 배치 경로가 키워둔 버퍼를 재사용하다가 두 경로의 예측 결과가 갈라지는 문제가 있어
    // 아예 분리했다).

    // x, y 는 T*D 평면 배열, mu/rstd 는 [T] 이상 — t_ 마다 서로 겹치지 않아 병렬 안전
    void fwd(const(float)[] x, float[] y, float[] mu, float[] rstd, int T) {
        _parChunk(T, (int tlo, int thi) {
            foreach (t_; tlo .. thi) {
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
        });
    }

    // dy -> dx (기울기 gg, gb 에 누적). mu/rstd 는 직전 fwd() 가 채운 것, xh 는 [T*D] 이상
    // (1단계에서 계산해 2단계로 넘기는 캐시).
    // gg/gb 는 D축(i) 로만 인덱싱되고 모든 t_ 가 같은 자리에 누적하므로 t_ 기준
    // 병렬화가 안전하지 않다. dx 계산은 t_ 마다 disjoint 라 안전 — 그래서 2단계로 나눈다:
    // 1단계(t_ 기준 병렬) dx 계산 + xh 캐시,  2단계(i 기준 병렬) gg/gb 를 t_ 오름차순으로 누적
    // (원래 직렬 코드와 같은 순서로 더해야 부동소수 결과가 비트 단위로 같다).
    void bwd(const(float)[] x, const(float)[] dy, float[] dx,
             const(float)[] mu, const(float)[] rstd, float[] xh, int T) {
        _parChunk(T, (int tlo, int thi) {
            foreach (t_; tlo .. thi) {
                auto xs = x[t_*D .. t_*D + D];
                float m = mu[t_], r = rstd[t_];
                float sum1 = 0f, sum2 = 0f;
                foreach (i; 0..D) {
                    float xhv = (xs[i] - m) * r;
                    xh[t_*D + i] = xhv;
                    float dyg = dy[t_*D + i] * g[i];
                    sum1 += dyg;
                    sum2 += dyg * xhv;
                }
                sum1 /= D; sum2 /= D;
                foreach (i; 0..D) {
                    float dyg = dy[t_*D + i] * g[i];
                    dx[t_*D + i] += r * (dyg - sum1 - xh[t_*D + i] * sum2);
                }
            }
        });
        _parChunk(D, (int ilo, int ihi) {
            foreach (i; ilo .. ihi) {
                foreach (t_; 0..T) {
                    gg[i] += dy[t_*D + i] * xh[t_*D + i];
                    gb[i] += dy[t_*D + i];
                }
            }
        });
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
            _parChunkNT(cast(int) w.length, (int ilo, int ihi) nothrow {
                foreach (i; ilo .. ihi) {
                    float gv = gr[i];
                    m[i] = B1*m[i] + (1-B1)*gv;
                    v[i] = B2*v[i] + (1-B2)*gv*gv;
                    w[i] -= lr * (m[i]/bc1) / (sqrt(v[i]/bc2) + EPS);
                }
            });
            break;
        case Opt.sgd:
            _parChunkNT(cast(int) w.length, (int ilo, int ihi) nothrow {
                foreach (i; ilo .. ihi) w[i] -= lr * gr[i];
            });
            break;
        case Opt.rmsprop:
            _parChunkNT(cast(int) w.length, (int ilo, int ihi) nothrow {
                foreach (i; ilo .. ihi) {
                    float gv = gr[i];
                    v[i] = RHO*v[i] + (1-RHO)*gv*gv;
                    w[i] -= lr * gv / (sqrt(v[i]) + EPS);
                }
            });
            break;
        case Opt.adagrad:
            _parChunkNT(cast(int) w.length, (int ilo, int ihi) nothrow {
                foreach (i; ilo .. ihi) {
                    float gv = gr[i];
                    v[i] += gv*gv;
                    w[i] -= lr * gv / (sqrt(v[i]) + EPS);
                }
            });
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
// EachLayer — 항목마다 따로 도는 층
// ─────────────────────────────────────────────
//   같은 가중치 하나를 항목 수만큼 돌려쓴다.
//   일반 Linear 는 전체를 한 덩어리로 섞어서 항목 구분이 사라지는데,
//   이 층은 항목 경계를 유지한다. attn 사이에 끼우면 조각이 안 무너진다.
//
//     폭 128, 항목 8 (조각당 16) 에서 each(64):
//       16칸 -> 64칸 을 8번  =>  나오는 폭 512
//       가중치는 16x64 하나뿐 (8항목이 공유)
// ─────────────────────────────────────────────
private struct EachLayer {
    int items;      // 항목 수
    int inW, outW;  // 항목 하나의 입력폭 / 출력폭
    Linear lin;
    float[] pre;    // ReLU 전 값 (역전파에 필요) — 단일 샘플(serial) 경로용

    int _bcap;
    float[] preB;   // [_bcap*items*outW] — fwdBatch/bwdBatch 용 (ReLU 전 값)

    this(int items_, int inW_, int outW_) {
        items = items_; inW = inW_; outW = outW_;
        lin = Linear(inW_, outW_);
        pre = new float[items_ * outW_];
        pre[] = 0f;
    }

    void _allocBatch(int B) {
        if (B <= _bcap) return;
        preB = new float[B * items * outW];
        preB[] = 0f;
        _bcap = B;
    }

    // X: [B*items*inW] (샘플마다 items*inW). Y: [B*items*outW].
    // items 뿐 아니라 B 도 전부 같은 lin 을 공유하는 독립 항목이라 count=B*items 로
    // batchForward/batchAccum 을 한 번에 돌린다 (EachLayer 안에는 항목간 상호작용이 없다
    // — 그게 AttnLayer 와 다른 점이라 배치 처리가 더 간단하다).
    void fwdBatch(const(float)[] X, float[] Y, int B) {
        _allocBatch(B);
        int count = B * items;
        lin.batchForward(X, preB, count);
        _parChunk(count, (int clo, int chi) {
            foreach (c; clo .. chi) foreach (i; 0..outW) {
                float v = preB[c*outW + i];
                Y[c*outW + i] = v < 0f ? 0f : v;
            }
        });
    }

    // dY -> dX. tmp 는 호출자 스크래치, 최소 [B*items*outW] 필요
    // (Network._bDZ[i] 가 이미 B*_outSz[i] = B*items*outW 로 잡혀 있어 그대로 쓸 수 있다).
    void bwdBatch(const(float)[] X, const(float)[] dY, float[] dX, float[] tmp, int B) {
        int count = B * items;
        _parChunk(count, (int clo, int chi) {
            foreach (c; clo .. chi) foreach (i; 0..outW)
                tmp[c*outW + i] = preB[c*outW + i] > 0f ? dY[c*outW + i] : 0f;
        });
        lin.batchAccum(X, tmp[0 .. count*outW], dX, count);
    }

    int inSize()  const nothrow @nogc { return items * inW; }
    int outSize() const nothrow @nogc { return items * outW; }

    // x (items*inW) -> y (items*outW), ReLU 포함.
    // items 개 항목이 전부 같은 lin 을 공유하므로(가중치 재사용), Linear 를 "샘플 B" 대신
    // "항목 count=items" 로 배치 처리하는 것과 수학적으로 같다 — batchForward 재사용.
    void fwd(const(float)[] x, float[] y) {
        lin.batchForward(x, pre, items);
        _parChunk(items, (int tlo, int thi) {
            foreach (t; tlo .. thi) foreach (i; 0..outW) {
                float v = pre[t*outW + i];
                y[t*outW + i] = v < 0f ? 0f : v;
            }
        });
    }

    // dy -> dx. tmp 는 [items*outW] 이상 (호출자의 스크래치 버퍼가 이미 그만큼 크다 —
    // Network._dC 는 각 층의 outSize() 전체를 기준으로 잡혀 있다).
    void bwd(const(float)[] x, const(float)[] dy, float[] dx, float[] tmp) {
        _parChunk(items, (int tlo, int thi) {
            foreach (t; tlo .. thi) foreach (i; 0..outW)
                tmp[t*outW + i] = pre[t*outW + i] > 0f ? dy[t*outW + i] : 0f;
        });
        lin.batchAccum(x, tmp[0 .. items*outW], dx, items);
    }

    void zeroGrad() nothrow @nogc { lin.zeroGrad(); }
    void step(Opt o, float lr, float decay = 0f) nothrow { lin.step(o, lr, decay); }
}

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
    float[] mu, rstd;      // LN 순전파 캐시 (per-sample 전용 — LN 은 이제 내부 상태가 없다)
    // 역전파 임시
    float[] dq, dk, dv, datt, dao, dx1;
    float[] xh;            // LN 역전파 캐시 (per-sample 전용)
    // bwd() 1단계(datt+dot 계산)에서 구해 2·3단계(dq, dk/dv)로 넘기는 스칼라 캐시 [heads*items]
    float[] dotBuf;

    // fwdBatch/bwdBatch 용 배치 스크래치 — B 개 샘플, 어텐션은 같은 샘플의 항목끼리만
    // 섞이므로(다른 샘플과는 절대 안 섞임) 위 필드들을 그냥 B배 키운 버전이다.
    // LN 의 mu/rstd/xh 도 여기 포함 — per-sample 쪽(mu/rstd/xh)과 절대 같은 버퍼를
    // 공유하지 않는다 (예전엔 LN 내부에서 공유·재할당하다가 두 경로 예측 결과가
    // 미세하게 갈라지는 문제가 있었다).
    int _bcap;
    float[] x1B, qB, kB, vB, attB, aoB, poB;
    float[] muB, rstdB;
    float[] dqB, dkB, dvB, dattB, daoB, dx1B, dotBufB;
    float[] xhB;

    void _allocBatch(int B) {
        if (B <= _bcap) return;
        x1B = new float[B*dim]; qB = new float[B*dim]; kB = new float[B*dim]; vB = new float[B*dim];
        attB = new float[B*heads*items*items];
        aoB = new float[B*dim]; poB = new float[B*dim];
        muB = new float[B*items]; rstdB = new float[B*items];
        dqB = new float[B*dim]; dkB = new float[B*dim]; dvB = new float[B*dim];
        dattB = new float[B*heads*items*items];
        daoB = new float[B*dim]; dx1B = new float[B*dim];
        dotBufB = new float[B*heads*items];
        xhB = new float[B*dim];   // T*D = (B*items)*w = B*(items*w) = B*dim
        foreach (a; [x1B,qB,kB,vB,attB,aoB,poB,muB,rstdB,dqB,dkB,dvB,dattB,daoB,dx1B,dotBufB,xhB])
            a[] = 0f;
        _bcap = B;
    }

    this(int dim_, int items_, int heads_) {
        dim = dim_; items = items_; w = dim_ / items_; heads = heads_; hw = w / heads_;
        ln = LN(w);
        wq = Linear(w, w); wk = Linear(w, w); wv = Linear(w, w); wo = Linear(w, w);

        x1 = new float[dim]; q = new float[dim]; k = new float[dim]; v = new float[dim];
        att = new float[heads*items*items];
        ao = new float[dim]; po = new float[dim];
        mu = new float[items]; rstd = new float[items];
        dq = new float[dim]; dk = new float[dim]; dv = new float[dim];
        datt = new float[heads*items*items];
        dao = new float[dim]; dx1 = new float[dim];
        xh = new float[dim];   // T*D = items*w = dim
        dotBuf = new float[heads*items];
        foreach (a; [x1,q,k,v,att,ao,po,mu,rstd,dq,dk,dv,datt,dao,dx1,xh,dotBuf]) a[] = 0f;
    }

    // y = x + 어텐션(LayerNorm(x))
    // 항목(t)별로 q/k/v/ao/po 를 서로 겹치지 않게 쓰므로 t (또는 h*items+t) 기준
    // 병렬화가 안전하다. forward() 는 가중치를 읽기만 해서 항목끼리 공유해도 된다.
    void fwd(const(float)[] x, float[] y) {
        ln.fwd(x, x1, mu, rstd, items);
        _parChunk(items, (int tlo, int thi) {
            foreach (t; tlo .. thi) {
                wq.forward(x1[t*w .. t*w+w], q[t*w .. t*w+w]);
                wk.forward(x1[t*w .. t*w+w], k[t*w .. t*w+w]);
                wv.forward(x1[t*w .. t*w+w], v[t*w .. t*w+w]);
            }
        });
        float scale = 1f / sqrt(cast(float) hw);
        _parChunk(heads * items, (int lo, int hi) {
            foreach (ht; lo .. hi) {
                int h = ht / items, t = ht % items;
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
        });
        _parChunk(items, (int tlo, int thi) {
            foreach (t; tlo .. thi) wo.forward(ao[t*w .. t*w+w], po[t*w .. t*w+w]);
        });
        foreach (i; 0..dim) y[i] = x[i] + po[i];      // 잔차
    }

    // dy -> dx (dx 에 누적).
    // wq/wk/wv/wo 는 items 개 항목이 전부 같은 가중치를 공유하므로 batchAccum(count=items)
    // 재사용 (fwd 와 같은 이유). q/k/v 로의 역전파(dq/dk/dv)는 datt[h,t,s] 를 dq 는 t 로,
    // dk/dv 는 s 로 읽으므로 그대로 t 기준 병렬화하면 dk/dv 가 모든 t 에서 겹쳐 써진다
    // (레이스) — 그래서 3단계로 나눈다: 1단계(t 기준) datt+dot 계산, 2단계(t 기준) dq 누적,
    // 3단계(s 기준, 루프 순서를 h·t 로 뒤집어서) dk/dv 누적. 각 단계는 원래 직렬 코드와
    // 같은 합산 순서를 지켜서 결과가 비트 단위로 같다.
    void bwd(const(float)[] x, const(float)[] dy, float[] dx) {
        wo.batchAccum(ao, dy, dao, items);

        float scale = 1f / sqrt(cast(float) hw);

        // 1단계: datt[h,t,s] 계산 + dot[h,t] 스칼라를 dotBuf 에 캐시 (dq/dk/dv 는 아직 안 건드림)
        _parChunk(heads * items, (int lo, int hi) {
            foreach (ht; lo .. hi) {
                int h = ht / items, t = ht % items;
                float dot = 0f;
                foreach (s; 0..items) {
                    float d_ = 0f;
                    foreach (i; 0..hw) d_ += dao[t*w + h*hw + i] * v[s*w + h*hw + i];
                    datt[h*items*items + t*items + s] = d_;
                    dot += d_ * att[h*items*items + t*items + s];
                }
                dotBuf[h*items + t] = dot;
            }
        });

        // 2단계: dq[t...] 는 오직 t 자기 반복 안에서만 쓰인다 — t(또는 h*items+t) 기준 안전
        dq[] = 0f;
        _parChunk(heads * items, (int lo, int hi) {
            foreach (ht; lo .. hi) {
                int h = ht / items, t = ht % items;
                float dot = dotBuf[h*items + t];
                foreach (s; 0..items) {
                    float a = att[h*items*items + t*items + s];
                    float ds = a * (datt[h*items*items + t*items + s] - dot) * scale;
                    foreach (i; 0..hw) dq[t*w + h*hw + i] += ds * k[s*w + h*hw + i];
                }
            }
        });

        // 3단계: dk[s...]/dv[s...] 는 모든 t 에서 누적되므로 s 기준으로 뒤집어서 병렬화
        // (안쪽은 원래와 같은 h-먼저-t 순서로 돌아 합산 순서를 그대로 보존한다)
        dk[] = 0f; dv[] = 0f;
        _parChunk(items, (int slo, int shi) {
            foreach (s; slo .. shi) {
                foreach (h; 0..heads) foreach (t; 0..items) {
                    float a = att[h*items*items + t*items + s];
                    foreach (i; 0..hw) dv[s*w + h*hw + i] += a * dao[t*w + h*hw + i];
                    float dot = dotBuf[h*items + t];
                    float ds = a * (datt[h*items*items + t*items + s] - dot) * scale;
                    foreach (i; 0..hw) dk[s*w + h*hw + i] += ds * q[t*w + h*hw + i];
                }
            }
        });

        dx1[] = 0f;
        wq.batchAccum(x1, dq, dx1, items, true, false);
        wk.batchAccum(x1, dk, dx1, items, true, false);
        wv.batchAccum(x1, dv, dx1, items, true, false);
        ln.bwd(x, dx1, dx, mu, rstd, xh, items);
        foreach (i; 0..dim) dx[i] += dy[i];           // 잔차 통과분
    }

    // X, Y: [B*dim] — B 개 샘플. LayerNorm 과 wq/wk/wv/wo 투영은 B*items 개
    // 항목이 전부 독립(가중치만 공유)이라 fwd() 와 완전히 같은 방식으로 count=B*items
    // 로 한 번에 처리한다. 어텐션(QK^T/softmax/weighted-V) 만은 같은 샘플의 항목끼리만
    // 섞여야 하므로 (b,h,t) 3중 인덱스로 병렬화하고 s 는 그 샘플 안에서만 돈다.
    void fwdBatch(const(float)[] X, float[] Y, int B) {
        _allocBatch(B);
        int BT = B * items;
        ln.fwd(X, x1B, muB, rstdB, BT);
        wq.batchForward(x1B, qB, BT);
        wk.batchForward(x1B, kB, BT);
        wv.batchForward(x1B, vB, BT);

        float scale = 1f / sqrt(cast(float) hw);
        _parChunk(B * heads * items, (int lo, int hi) {
            foreach (bht; lo .. hi) {
                int b = bht / (heads*items);
                int rem = bht % (heads*items);
                int h = rem / items, t = rem % items;
                int off = b*dim;              // qB/kB/vB/aoB 안에서 샘플 b 의 시작
                int aoff = b*heads*items*items;  // attB 안에서 샘플 b 의 시작
                float mx = -1e30f;
                foreach (s; 0..items) {
                    float dot = 0f;
                    foreach (i; 0..hw) dot += qB[off+t*w+h*hw+i] * kB[off+s*w+h*hw+i];
                    dot *= scale;
                    attB[aoff + h*items*items + t*items + s] = dot;
                    if (dot > mx) mx = dot;
                }
                float sum = 0f;
                foreach (s; 0..items) {
                    float e = exp(attB[aoff + h*items*items + t*items + s] - mx);
                    attB[aoff + h*items*items + t*items + s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                foreach (s; 0..items) attB[aoff + h*items*items + t*items + s] *= inv;
                foreach (i; 0..hw) {
                    float acc = 0f;
                    foreach (s; 0..items)
                        acc += attB[aoff + h*items*items + t*items + s] * vB[off+s*w+h*hw+i];
                    aoB[off + t*w + h*hw + i] = acc;
                }
            }
        });
        wo.batchForward(aoB, poB, BT);
        foreach (c; 0 .. B*dim) Y[c] = X[c] + poB[c];      // 잔차
    }

    // dY -> dX (dX 에 누적). bwd() 의 3단계 재구성을 그대로 샘플 차원 b 를 하나 더 얹어 반복한다.
    void bwdBatch(const(float)[] X, const(float)[] dY, float[] dX, int B) {
        int BT = B * items;
        wo.batchAccum(aoB, dY, daoB, BT);

        float scale = 1f / sqrt(cast(float) hw);

        _parChunk(B * heads * items, (int lo, int hi) {
            foreach (bht; lo .. hi) {
                int b = bht / (heads*items);
                int rem = bht % (heads*items);
                int h = rem / items, t = rem % items;
                int off = b*dim; int aoff = b*heads*items*items;
                float dot = 0f;
                foreach (s; 0..items) {
                    float d_ = 0f;
                    foreach (i; 0..hw) d_ += daoB[off+t*w+h*hw+i] * vB[off+s*w+h*hw+i];
                    dattB[aoff + h*items*items + t*items + s] = d_;
                    dot += d_ * attB[aoff + h*items*items + t*items + s];
                }
                dotBufB[b*heads*items + h*items + t] = dot;
            }
        });

        dqB[0 .. B*dim] = 0f;
        _parChunk(B * heads * items, (int lo, int hi) {
            foreach (bht; lo .. hi) {
                int b = bht / (heads*items);
                int rem = bht % (heads*items);
                int h = rem / items, t = rem % items;
                int off = b*dim; int aoff = b*heads*items*items;
                float dot = dotBufB[b*heads*items + h*items + t];
                foreach (s; 0..items) {
                    float a = attB[aoff + h*items*items + t*items + s];
                    float ds = a * (dattB[aoff + h*items*items + t*items + s] - dot) * scale;
                    foreach (i; 0..hw) dqB[off+t*w+h*hw+i] += ds * kB[off+s*w+h*hw+i];
                }
            }
        });

        dkB[0 .. B*dim] = 0f; dvB[0 .. B*dim] = 0f;
        _parChunk(B * items, (int lo, int hi) {
            foreach (bs; lo .. hi) {
                int b = bs / items, s = bs % items;
                int off = b*dim; int aoff = b*heads*items*items;
                foreach (h; 0..heads) foreach (t; 0..items) {
                    float a = attB[aoff + h*items*items + t*items + s];
                    foreach (i; 0..hw) dvB[off+s*w+h*hw+i] += a * daoB[off+t*w+h*hw+i];
                    float dot = dotBufB[b*heads*items + h*items + t];
                    float ds = a * (dattB[aoff + h*items*items + t*items + s] - dot) * scale;
                    foreach (i; 0..hw) dkB[off+s*w+h*hw+i] += ds * qB[off+t*w+h*hw+i];
                }
            }
        });

        dx1B[0 .. B*dim] = 0f;
        wq.batchAccum(x1B, dqB, dx1B, BT, true, false);
        wk.batchAccum(x1B, dkB, dx1B, BT, true, false);
        wv.batchAccum(x1B, dvB, dx1B, BT, true, false);
        ln.bwd(X, dx1B, dX, muB, rstdB, xhB, BT);
        foreach (c; 0 .. B*dim) dX[c] += dY[c];           // 잔차 통과분
    }

    void zeroGrad() nothrow @nogc {
        ln.zero();
        wq.zeroGrad(); wk.zeroGrad(); wv.zeroGrad(); wo.zeroGrad();
    }

    void step(Opt o, float lr, float decay = 0f) nothrow {
        ln.step(o, lr);          // LayerNorm 의 scale/shift 에는 감쇠를 걸지 않는다
        wq.step(o, lr, decay); wk.step(o, lr, decay);
        wv.step(o, lr, decay); wo.step(o, lr, decay);
    }
}

// ─────────────────────────────────────────────
// LogicLayer — 논리 게이트를 학습하는 층 (뉴로심볼릭)
// ─────────────────────────────────────────────
// 출력 유닛 하나하나가 "논리 게이트 한 개" 다. 유닛 j 는 입력 두 칸(ia[j], ib[j])을
// 보고, 그 둘로 뭘 할지를 16가지 중에서 고른다 (AND, OR, XOR, NAND, "a 를 그대로",
// "항상 참" ...). 고르는 방식이 학습 대상이다.
//
// 어떻게 미분 가능하게 만드나:
//   게이트 선택을 softmax 로 부드럽게 섞는다. y = Σ_g p_g · g(a,b).
//   학습 중에는 "70% AND, 20% OR, ..." 같은 상태로 있다가 한쪽으로 몰린다.
//   다 끝나면 제일 큰 것만 남겨서 읽으면 그게 진짜 논리식이다 (rules() 참고).
//   이게 이 층을 쓰는 이유다 — Linear 로는 "무슨 규칙을 배웠는지" 를 못 읽는다.
//
// 참/거짓 대신 0~1 사이 값을 쓰고, 게이트는 확률곱으로 완화한다
// (a∧b → a·b, a∨b → a+b-ab, ¬a → 1-a). a, b 가 딱 0/1 이면 보통 논리와 같아진다.
//
// 입력 두 칸은 생성할 때 무작위로 정하고 고정한다 (학습하지 않는다 — 연결까지
// 학습시키면 탐색 공간이 터진다. Petersen 등의 differentiable logic gate network
// 와 같은 선택이다). 무작위이므로 파일에 같이 저장해야 한다.
private enum int NGATE = 16;

// 16가지 게이트. 인덱스가 곧 진리표다 (하위 비트가 (a,b) = 00,01,10,11).
private immutable string[NGATE] GATE_NAME = [
    "거짓", "a∧b", "a∧¬b", "a", "¬a∧b", "b", "a⊕b", "a∨b",
    "¬(a∨b)", "a↔b", "¬b", "a∨¬b", "¬a", "¬a∨b", "¬(a∧b)", "참"
];

// g(a,b) 를 확률곱으로 완화한 값. 16개를 한 번에 채운다.
private void gateValues(float a, float b, ref float[NGATE] g) pure nothrow @nogc {
    immutable float ab = a * b;
    g[0]  = 0f;
    g[1]  = ab;                 // a∧b
    g[2]  = a - ab;             // a∧¬b
    g[3]  = a;
    g[4]  = b - ab;             // ¬a∧b
    g[5]  = b;
    g[6]  = a + b - 2f*ab;      // a⊕b
    g[7]  = a + b - ab;         // a∨b
    g[8]  = 1f - g[7];          // ¬(a∨b)
    g[9]  = 1f - g[6];          // a↔b
    g[10] = 1f - b;
    g[11] = 1f - b + ab;        // a∨¬b
    g[12] = 1f - a;
    g[13] = 1f - a + ab;        // ¬a∨b
    g[14] = 1f - ab;            // ¬(a∧b)
    g[15] = 1f;
}

// ∂g/∂a 와 ∂g/∂b. 위 식을 손으로 미분한 것 — 바꿀 때 둘을 같이 고쳐야 한다.
private void gateGrads(float a, float b, ref float[NGATE] da, ref float[NGATE] db)
        pure nothrow @nogc {
    da[0]=0f;      db[0]=0f;
    da[1]=b;       db[1]=a;
    da[2]=1f-b;    db[2]=-a;
    da[3]=1f;      db[3]=0f;
    da[4]=-b;      db[4]=1f-a;
    da[5]=0f;      db[5]=1f;
    da[6]=1f-2f*b; db[6]=1f-2f*a;
    da[7]=1f-b;    db[7]=1f-a;
    da[8]=b-1f;    db[8]=a-1f;
    da[9]=2f*b-1f; db[9]=2f*a-1f;
    da[10]=0f;     db[10]=-1f;
    da[11]=b;      db[11]=a-1f;
    da[12]=-1f;    db[12]=0f;
    da[13]=b-1f;   db[13]=a;
    da[14]=-b;     db[14]=-a;
    da[15]=0f;     db[15]=0f;
}

private struct LogicLayer {
    int inSz, outSz;
    int[] ia, ib;        // 유닛 j 가 보는 두 입력 칸 (고정, 파일에 저장)
    float[] gw;          // [outSz*NGATE] 게이트 선택 가중치 (학습 대상)
    float[] ggw;         // 기울기
    float[] mg, vg;      // Adam 상태
    int t;

    // 들어온 값을 0~1 로 눌러야 하는가. 바로 앞이 또 로직 층이면 이미 0~1 이라
    // 다시 누르면 [0.5, 0.73] 로 찌그러져서 신호가 죽는다. 그래서 생성 시 정한다.
    bool squash;

    float[] prob;        // [outSz*NGATE] softmax 결과 (순전파에서 채움)
    float[] sx;          // [inSz] 눌러놓은 입력 — 역전파에서 미분계수에 필요
    float[] outCache;    // [outSz] 출력값 — softmax 역전파에 필요
    float[] dsx;         // [inSz] 입력에 대한 기울기 모으는 곳

    int _bcap;
    float[] sxB, outB, dsxB;   // 배치용 (각각 B*inSz, B*outSz, B*inSz)

    this(int inSz_, int outSz_, bool squash_) {
        inSz = inSz_; outSz = outSz_; squash = squash_;
        ia = new int[outSz]; ib = new int[outSz];
        foreach (j; 0..outSz) {
            ia[j] = uniform(0, inSz, rng);
            // 두 입력이 같으면 게이트 절반이 의미를 잃는다 (a∧a == a). 피한다.
            if (inSz > 1) {
                int b2 = uniform(0, inSz - 1, rng);
                ib[j] = (b2 >= ia[j]) ? b2 + 1 : b2;
            } else ib[j] = 0;
        }
        gw  = new float[outSz*NGATE]; gw[]  = 0f;   // 전부 0 = 처음엔 16개 균등
        ggw = new float[outSz*NGATE]; ggw[] = 0f;
        mg  = new float[outSz*NGATE]; mg[]  = 0f;
        vg  = new float[outSz*NGATE]; vg[]  = 0f;
        prob = new float[outSz*NGATE]; prob[] = 0f;
        sx = new float[inSz]; sx[] = 0f;
        outCache = new float[outSz]; outCache[] = 0f;
        dsx = new float[inSz]; dsx[] = 0f;
    }

    private void _allocBatch(int B) {
        if (B <= _bcap) return;
        sxB  = new float[B*inSz];  sxB[]  = 0f;
        outB = new float[B*outSz]; outB[] = 0f;
        dsxB = new float[B*inSz];  dsxB[] = 0f;
        _bcap = B;
    }

    // 게이트 선택 확률. gw 가 바뀔 때마다 다시 구해야 한다.
    private void _softmaxGates() {
        _parChunk(outSz, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) {
                auto z = gw[j*NGATE .. (j+1)*NGATE];
                auto p = prob[j*NGATE .. (j+1)*NGATE];
                float mx = z[0];
                foreach (v; z) if (v > mx) mx = v;
                float s = 0f;
                foreach (k; 0..NGATE) { p[k] = exp(z[k] - mx); s += p[k]; }
                float inv = 1f / s;
                foreach (k; 0..NGATE) p[k] *= inv;
            }
        });
    }

    private static float _sig(float v) pure nothrow @nogc {
        return 1f / (1f + exp(-v));
    }

    void fwd(const(float)[] x, float[] y) {
        _softmaxGates();
        foreach (i; 0..inSz) sx[i] = squash ? _sig(x[i]) : x[i];
        _parChunk(outSz, (int jlo, int jhi) {
            float[NGATE] g;
            foreach (j; jlo .. jhi) {
                gateValues(sx[ia[j]], sx[ib[j]], g);
                auto p = prob[j*NGATE .. (j+1)*NGATE];
                float acc = 0f;
                foreach (k; 0..NGATE) acc += p[k] * g[k];
                outCache[j] = acc;
                y[j] = acc;
            }
        });
    }

    // dy -> dx.  ggw 에도 누적한다.
    //
    // 2단계로 나눈다. 1단계는 j 로 쪼개도 안전하다 (유닛마다 자기 16칸만 건드린다).
    // 2단계는 안 된다 — 서로 다른 유닛이 같은 입력 칸을 볼 수 있어서 dsx 가 겹친다.
    // 그래서 j 오름차순으로 직렬 누적한다 (순서를 고정해야 직렬과 비트가 같다).
    void bwd(const(float)[] x, const(float)[] dy, float[] dx) {
        _parChunk(outSz, (int jlo, int jhi) {
            float[NGATE] g;
            foreach (j; jlo .. jhi) {
                gateValues(sx[ia[j]], sx[ib[j]], g);
                auto p = prob[j*NGATE .. (j+1)*NGATE];
                auto gg = ggw[j*NGATE .. (j+1)*NGATE];
                // y = Σ p_k g_k, p = softmax(z) 이므로 dy/dz_m = p_m (g_m - y)
                float yv = outCache[j], d = dy[j];
                foreach (k; 0..NGATE) gg[k] += d * p[k] * (g[k] - yv);
            }
        });

        dsx[] = 0f;
        float[NGATE] da, db;
        foreach (j; 0..outSz) {
            int A = ia[j], B2 = ib[j];
            gateGrads(sx[A], sx[B2], da, db);
            auto p = prob[j*NGATE .. (j+1)*NGATE];
            float sa = 0f, sb = 0f;
            foreach (k; 0..NGATE) { sa += p[k]*da[k]; sb += p[k]*db[k]; }
            dsx[A]  += dy[j] * sa;
            dsx[B2] += dy[j] * sb;
        }
        foreach (i; 0..inSz)
            dx[i] += squash ? dsx[i] * sx[i] * (1f - sx[i]) : dsx[i];
    }

    // X: [B*inSz] -> Y: [B*outSz]
    void fwdBatch(const(float)[] X, float[] Y, int B) {
        _allocBatch(B);
        _softmaxGates();
        _parChunk(B, (int blo, int bhi) {
            float[NGATE] g;
            foreach (b; blo .. bhi) {
                auto s = sxB[b*inSz .. (b+1)*inSz];
                foreach (i; 0..inSz) s[i] = squash ? _sig(X[b*inSz + i]) : X[b*inSz + i];
                foreach (j; 0..outSz) {
                    gateValues(s[ia[j]], s[ib[j]], g);
                    auto p = prob[j*NGATE .. (j+1)*NGATE];
                    float acc = 0f;
                    foreach (k; 0..NGATE) acc += p[k] * g[k];
                    outB[b*outSz + j] = acc;
                    Y[b*outSz + j] = acc;
                }
            }
        });
    }

    // dY -> dX (누적). 1단계는 j 로, 2단계는 b 로 쪼갠다 — 둘 다 겹치지 않는다.
    void bwdBatch(const(float)[] X, const(float)[] dY, float[] dX, int B) {
        _parChunk(outSz, (int jlo, int jhi) {
            float[NGATE] g;
            foreach (j; jlo .. jhi) {
                auto p = prob[j*NGATE .. (j+1)*NGATE];
                auto gg = ggw[j*NGATE .. (j+1)*NGATE];
                foreach (b; 0..B) {         // b 오름차순 고정 = 결정적
                    auto s = sxB[b*inSz .. (b+1)*inSz];
                    gateValues(s[ia[j]], s[ib[j]], g);
                    float yv = outB[b*outSz + j], d = dY[b*outSz + j];
                    foreach (k; 0..NGATE) gg[k] += d * p[k] * (g[k] - yv);
                }
            }
        });

        _parChunk(B, (int blo, int bhi) {
            float[NGATE] da, db;
            foreach (b; blo .. bhi) {
                auto s  = sxB[b*inSz .. (b+1)*inSz];
                auto ds = dsxB[b*inSz .. (b+1)*inSz];
                ds[] = 0f;
                foreach (j; 0..outSz) {
                    int A = ia[j], B2 = ib[j];
                    gateGrads(s[A], s[B2], da, db);
                    auto p = prob[j*NGATE .. (j+1)*NGATE];
                    float sa = 0f, sb = 0f;
                    foreach (k; 0..NGATE) { sa += p[k]*da[k]; sb += p[k]*db[k]; }
                    float d = dY[b*outSz + j];
                    ds[A]  += d * sa;
                    ds[B2] += d * sb;
                }
                foreach (i; 0..inSz)
                    dX[b*inSz + i] += squash ? ds[i] * s[i] * (1f - s[i]) : ds[i];
            }
        });
    }

    void zeroGrad() nothrow @nogc { ggw[] = 0f; }

    void step(Opt o, float lr) nothrow {
        adamVec(gw, ggw, mg, vg, o, lr, t);
        t++;
    }

    // 학습된 회로를 사람이 읽을 수 있게. 유닛마다 "제일 유력한 게이트" 와
    // 그 확신도를 돌려준다. 이 층을 쓰는 목적이 사실 이것이다.
    string[] rules() {
        _softmaxGates();
        auto r = new string[outSz];
        foreach (j; 0..outSz) {
            auto p = prob[j*NGATE .. (j+1)*NGATE];
            int best = 0;
            foreach (k; 1..NGATE) if (p[k] > p[best]) best = k;
            string nm = GATE_NAME[best];
            // a, b 를 실제 입력 번호로 바꿔 끼운다
            string s;
            foreach (dchar c; nm) {
                if      (c == 'a') s ~= "x" ~ to!string(ia[j]);
                else if (c == 'b') s ~= "x" ~ to!string(ib[j]);
                else               s ~= to!string(c);
            }
            r[j] = s ~ "   (" ~ to!string(cast(int)(p[best]*100f + 0.5f)) ~ "%)";
        }
        return r;
    }
}

// ─────────────────────────────────────────────
// RnnLayer — 기억(메모장)을 들고 다니는 층
// ─────────────────────────────────────────────
// 신경망 안에 숫자 n개짜리 메모장(h)을 둔다. 매번 호출될 때마다 "이번 입력과 지금
// 메모를 보고 메모를 어떻게 고칠지" 를 계산한다. 그래서 입력을 늘리지 않고도 과거를
// 볼 수 있고, 몇 스텝 전까지 볼지 미리 정할 필요가 없다.
//
// 갱신식 (게이트 하나짜리 — GRU 를 절반으로 줄인 형태):
//     z  = sigmoid(zx(x) + zh(h))        게이트: 이번에 얼마나 고칠까
//     c  = tanh(gx(x) + gh(h))           고칠 후보값
//     h' = (1-z)*h + z*c
//
// 게이트가 있는 이유: 게이트 없이 h' = tanh(...) 로만 쓰면 매 스텝 tanh 로 눌려서
// 옛 기억이 금방 사라진다. z 가 0 에 가까우면 h 가 **그대로 통과**하므로 오래 간다.
// 이 라이브러리가 노리는 "몇십 스텝 전을 기억" 에 그 차이가 크다.
//
// ── 도달 거리는 "어떻게 주느냐" 가 정한다 ────────────────────────────
// 기울기가 과거로 몇 스텝 거슬러 가는지가 전부인데, 그게 호출 모양에 달려있다.
//
// ① **한 스텝씩** sl() 을 부르면 1스텝 절단이다. 그 호출에 사슬이라고 할 게
//    없으니(B=1) 어쩔 수 없다. 중간 스텝은 정답이 없어서(sl(x,[None])) 기울기가
//    0 이고 그 0 이 더 거슬러 가지도 않는다. 그래서 **게이트가 "언제 열고 닫을지"
//    를 못 배우고** z 는 초기값 0.5 근처에 머문다. 배우는 건 마지막 스텝의
//    "메모 읽는 법" 뿐이고, 도달 거리는 학습이 아니라 forward 감쇠(≈0.5^K)가
//    정한다.
//
// ② **에피소드를 묶음으로** 주면 사슬 전체를 거슬러 간다 (BPTT). fwdBatch 가
//    샘플마다 hPrevB/cB/zB 를 남기므로 테이프가 이미 있고, bwdBatch 가 b 를
//    내림차순으로 돌면서 h 로 가는 기울기를 이어 넘긴다 (_bwdBatchBptt).
//    이제 게이트도 배운다.
//
//        with ai.round():
//            ai.sl(한_에피소드_입력들, 한_에피소드_정답들)
//
// 실측 (단서가 t=0 에만 보이고 K 스텝 뒤에 답해야 하는 문제, 10시드,
// [2, memory(16), 16], lr=0.02, 4000판):
//        K          4       8      16      32
//     ① 한 스텝씩  9/10   1/10    0/10    0/10
//     ② 묶음+BPTT  10/10  10/10   5/10    1/10
// 묶어서 주되 BPTT 를 끄면 ① 과 같다 — 이득은 묶음이 아니라 BPTT 가 낸다.
// 비용은 학습 시간 +15~25%. 쓸모 있는 범위가 K≈4 에서 K≈8(확실)~16(반반)로 늘었다.
// K=32 는 여전히 안 된다. 결과가 양극단으로 갈리는 성질은 그대로다.
//
// 1스텝 절단이 진짜 기울기와 얼마나 다른지도 재봤다: **최대 기울기의 60%**.
// 유한차분 ↔ 독립 파이썬 역전파 ↔ 이 코드 3중으로 확인했다 (각각 1e-5, 1e-7).
// 임시 프로브로 한 번 보고 지웠다 — 상시 감시는 tests/memory.py 의 K=8 과
// "B=1 이면 BPTT 를 켜도 꺼도 같은 값" 이다.
private struct RnnLayer {
    int inSz, outSz;
    Linear gx, gh;        // 후보값: 입력 쪽 / 메모 쪽
    Linear zx, zh;        // 게이트: 입력 쪽 / 메모 쪽

    float[] h;            // 메모장. 호출 사이에 남는다 (학습된 값이 아니라 현재 상태)
    float[] hSnap;        // "다시 돌릴 때 출발할 메모". 아래 설명 참고.
    float[] hPrev, cc, c, cz, z, ccH, czH;
    float[] dcc, dcz, dxTmp, dhSink;

    // 배치용 — 샘플마다 역전파에 필요한 값(지난 메모, 후보값, 게이트)을 남겨둔다.
    int _bcap;
    float[] hPrevB, cB, zB;
    float[] dhCarry;      // BPTT: 다음(시간상 뒤) 스텝에서 흘러온 기울기

    this(int inSz_, int outSz_) {
        inSz = inSz_; outSz = outSz_;
        gx = Linear(inSz_, outSz_); gh = Linear(outSz_, outSz_);
        zx = Linear(inSz_, outSz_); zh = Linear(outSz_, outSz_);
        // 게이트 편향은 0 으로 둔다 (Linear 의 기본값 그대로) = z 가 0.5 에서 출발.
        //
        // LSTM 의 forget-gate-bias 관례를 따라 "유지" 쪽(음수)으로 밀어봤는데
        // 실측으로 더 나빴다. z 는 유지만 조절하는 게 아니라 쓰기도 같이 조절하기
        // 때문이다 (h' = (1-z)h + z·c) — z 를 작게 하면 오래 남지만 애초에 적히는
        // 양도 그만큼 적어진다. 5시드 × K 측정 (90% 넘게 푼 횟수):
        //       편향   K=2    K=4    K=8
        //         0    5/5    5/5    0/5
        //        -1    5/5    4/5    0/5
        //        -2    5/5    4/5    1/5
        //        -3    3/5    0/5    0/5
        // 바꾸려면 이 표를 다시 재고 나서 바꿀 것.
        foreach (ref a; [&h, &hSnap, &hPrev, &cc, &c, &cz, &z, &ccH, &czH,
                         &dcc, &dcz, &dhSink, &dhCarry])
            { *a = new float[outSz_]; (*a)[] = 0f; }
        dxTmp = new float[inSz_]; dxTmp[] = 0f;
    }

    private static float _sig(float v) nothrow @nogc {
        return 1f / (1f + exp(-v));
    }

    // 메모를 비운다. 에피소드가 바뀔 때 부른다. 재생 출발점도 같이 비운다 —
    // 판이 처음부터 시작하면 다시 돌릴 때도 처음부터 출발해야 한다.
    void forget() nothrow @nogc { h[] = 0f; hSnap[] = 0f; }

    // ── 재생 출발점 (hSnap) ──────────────────────────────────────────
    // rl() 로 모아서 save() 로 학습하는 경로는 학습할 때 그 스텝들을 **다시**
    // 순전파한다. 그때 출발 메모가 "수집할 때 그 첫 스텝이 보던 메모" 여야 한다.
    // 그런데 수집이 이미 메모를 끝까지 밀어놨으므로 h 는 쓸 수 없다.
    //
    // 그래서 "아직 학습 안 한 가장 오래된 스텝이 보던 메모" 를 따로 들고 있는다.
    // 갱신 시점은 두 곳뿐이다: forget() 과 학습이 끝난 직후. rl()/predict() 는
    // h 만 밀고 hSnap 은 건드리지 않는다 — 그게 이 버퍼의 요점이다.
    //
    // 이러면 두 패턴이 다 맞는다:
    //   판 끝에 한 번 저장:  forget→snap=0, rl×N(h→N), save(0 에서 재생→N, snap=N)
    //   한 스텝씩 저장:      forget→snap=0, rl(h→m1), save(0 에서 재생→m1, snap=m1),
    //                        rl(h→m2), save(m1 에서 재생→m2, snap=m2)  ...
    // 어느 쪽이든 재생이 끝난 h 가 수집이 남긴 h 와 같아진다 (가중치 갱신은
    // 순전파 뒤에 일어나므로 같은 가중치로 같은 입력을 돈다).
    void snapshotMemo() nothrow @nogc { hSnap[] = h[]; }
    void restoreMemo()  nothrow @nogc { h[] = hSnap[]; }

    void fwd(const(float)[] x, float[] y) {
        hPrev[] = h[];
        gx.forward(x, cc);  gh.forward(hPrev, ccH);
        zx.forward(x, cz);  zh.forward(hPrev, czH);
        foreach (i; 0..outSz) {
            cc[i] += ccH[i];  c[i] = tanh(cc[i]);
            cz[i] += czH[i];  z[i] = _sig(cz[i]);
            h[i] = (1f - z[i])*hPrev[i] + z[i]*c[i];
            y[i] = h[i];
        }
    }

    // dy -> dx (누적). 가중치 기울기도 네 Linear 에 누적한다.
    void bwd(const(float)[] x, const(float)[] dy, float[] dx) {
        foreach (i; 0..outSz) {
            float dz = dy[i] * (c[i] - hPrev[i]);
            float dcv = dy[i] * z[i];
            dcc[i] = dcv * (1f - c[i]*c[i]);          // tanh'
            dcz[i] = dz  * z[i] * (1f - z[i]);        // sigmoid'
        }
        // 입력 쪽: 두 Linear 가 같은 버퍼에 누적한다 (accum 은 더하기만 한다)
        dxTmp[] = 0f;
        gx.accum(x, dcc, dxTmp);
        zx.accum(x, dcz, dxTmp);
        foreach (k; 0..inSz) dx[k] += dxTmp[k];

        // 메모 쪽: 가중치 기울기는 받되 입력(=지난 메모)으로 가는 기울기는 버린다.
        // 이게 "1스텝 절단" 이다 — 여기서 안 끊으면 과거 전체를 들고 있어야 한다.
        dhSink[] = 0f; gh.accum(hPrev, dcc, dhSink);
        dhSink[] = 0f; zh.accum(hPrev, dcz, dhSink);
    }

    private void _allocBatch(int B) {
        if (B <= _bcap) return;
        hPrevB = new float[B*outSz]; hPrevB[] = 0f;
        cB     = new float[B*outSz]; cB[]     = 0f;
        zB     = new float[B*outSz]; zB[]     = 0f;
        _bcap = B;
    }

    // 배치여도 메모는 순서대로 이어져야 하므로 b 를 차례로 돈다.
    // 묶음의 이득은 이 층 앞뒤가 받는다 — 거기는 샘플이 서로 독립이라 그대로
    // 묶여서 돌고, 보통 계산량의 대부분이 그쪽이다.
    //
    // 중요: 도는 순서가 per-sample 경로와 같다 (b 오름차순으로 메모가 한 칸씩
    // 나아간다). 그래서 묶음과 직렬이 같은 답을 낸다 — 이 라이브러리의
    // MYML_NOBATCH 동치 불변조건을 깨지 않는다.
    void fwdBatch(const(float)[] X, float[] Y, int B) {
        _allocBatch(B);
        foreach (b; 0..B) {
            auto x  = X[b*inSz .. (b+1)*inSz];
            auto hp = hPrevB[b*outSz .. (b+1)*outSz];
            auto cb = cB[b*outSz .. (b+1)*outSz];
            auto zb = zB[b*outSz .. (b+1)*outSz];
            hp[] = h[];
            gx.forward(x, cc);  gh.forward(hp, ccH);
            zx.forward(x, cz);  zh.forward(hp, czH);
            foreach (i; 0..outSz) {
                cb[i] = tanh(cc[i] + ccH[i]);
                zb[i] = _sig(cz[i] + czH[i]);
                h[i]  = (1f - zb[i])*hp[i] + zb[i]*cb[i];
                Y[b*outSz + i] = h[i];
            }
        }
    }

    // dY -> dX (누적). _bptt 가 꺼져 있을 때만 쓰는 1스텝 절단 경로다.
    // b 오름차순 고정 — 가중치 기울기 합산 순서가 고정돼야 직렬 경로와
    // 비트 단위로 맞는다 (MYML_BPTT=0 동치 검증).
    void bwdBatch(const(float)[] X, const(float)[] dY, float[] dX, int B) {
        if (_bptt) { _bwdBatchBptt(X, dY, dX, B); return; }
        foreach (b; 0..B) {
            auto x  = X[b*inSz .. (b+1)*inSz];
            auto hp = hPrevB[b*outSz .. (b+1)*outSz];
            auto cb = cB[b*outSz .. (b+1)*outSz];
            auto zb = zB[b*outSz .. (b+1)*outSz];
            foreach (i; 0..outSz) {
                float d   = dY[b*outSz + i];
                float dz  = d * (cb[i] - hp[i]);
                float dcv = d * zb[i];
                dcc[i] = dcv * (1f - cb[i]*cb[i]);
                dcz[i] = dz  * zb[i] * (1f - zb[i]);
            }
            dxTmp[] = 0f;
            gx.accum(x, dcc, dxTmp);
            zx.accum(x, dcz, dxTmp);
            foreach (k; 0..inSz) dX[b*inSz + k] += dxTmp[k];
            // 1스텝 절단 — 지난 메모로 가는 기울기는 버린다 (per-sample 과 동일)
            dhSink[] = 0f; gh.accum(hp, dcc, dhSink);
            dhSink[] = 0f; zh.accum(hp, dcz, dhSink);
        }
    }

    // 묶음 전체를 하나의 사슬로 보고 시간을 거슬러 간다 (BPTT).
    // b 를 내림차순으로 돌면서 h 로 가는 기울기를 dhCarry 에 이어 넘긴다.
    //   h_b = (1-z_b)*h_{b-1} + z_b*c_b   이므로 h_{b-1} 로 가는 길이 셋이다:
    //     ① 직접 통과:  d_b * (1-z_b)
    //     ② 후보값 경유: gh 의 입력 기울기
    //     ③ 게이트 경유: zh 의 입력 기울기
    // 1스텝 절단은 ①을 버리고 ②③의 입력 기울기를 버리는 것이었다.
    private void _bwdBatchBptt(const(float)[] X, const(float)[] dY, float[] dX, int B) {
        dhCarry[] = 0f;
        foreach_reverse (b; 0..B) {
            auto x  = X[b*inSz .. (b+1)*inSz];
            auto hp = hPrevB[b*outSz .. (b+1)*outSz];
            auto cb = cB[b*outSz .. (b+1)*outSz];
            auto zb = zB[b*outSz .. (b+1)*outSz];
            foreach (i; 0..outSz) {
                float d   = dY[b*outSz + i] + dhCarry[i];
                float dz  = d * (cb[i] - hp[i]);
                float dcv = d * zb[i];
                dcc[i] = dcv * (1f - cb[i]*cb[i]);
                dcz[i] = dz  * zb[i] * (1f - zb[i]);
                dhCarry[i] = d * (1f - zb[i]);          // ① 직접 통과
            }
            dxTmp[] = 0f;
            gx.accum(x, dcc, dxTmp);
            zx.accum(x, dcz, dxTmp);
            foreach (k; 0..inSz) dX[b*inSz + k] += dxTmp[k];
            // ②③ — 지난 메모로 가는 기울기를 버리지 않고 모아서 넘긴다
            dhSink[] = 0f;
            gh.accum(hp, dcc, dhSink);
            zh.accum(hp, dcz, dhSink);
            foreach (i; 0..outSz) dhCarry[i] += dhSink[i];
        }
    }

    void zeroGrad() nothrow @nogc {
        gx.zeroGrad(); gh.zeroGrad(); zx.zeroGrad(); zh.zeroGrad();
    }

    void step(Opt o, float lr, float decay = 0f) nothrow {
        gx.step(o, lr, decay); gh.step(o, lr, decay);
        zx.step(o, lr, decay); zh.step(o, lr, decay);
    }
}

// ─────────────────────────────────────────────
// ConvLayer — 창을 미끄러뜨리며 같은 무늬를 찾는 층 (1차원 합성곱)
// ─────────────────────────────────────────────
// 입력을 "항목 items 개 × 채널 inCh 개" 로 보고, 창(win) 하나를 항목 축으로 쭉
// 미끄러뜨리면서 같은 가중치를 돌려 쓴다. "왼쪽 끝의 무늬" 와 "오른쪽 끝의 무늬" 를
// 따로 배우지 않고 **무늬 찾는 법 하나**를 배워서 전체에 적용하는 것이 요점이다.
// 그래서 가중치가 훨씬 적고 위치가 바뀌어도 통한다.
//
// 길이는 유지한다 (출력도 항목 items 개). 양끝은 0 으로 메운다 — 그래야 attn/each
// 처럼 항목 구조를 지키는 층들과 섞어 쓸 수 있다.
//
// 구현은 im2col + Linear 이다. 창 안의 값들을 한 줄로 늘어놓은 행렬을 만들면
// 합성곱이 그냥 행렬곱이 되고, 그러면 Linear 를 그대로 재사용할 수 있다
// (SIMD dot/saxpy, Adam 상태, 가중치 감쇠, 저장/불러오기가 전부 공짜로 따라온다).
// EachLayer 가 Linear.batchForward 를 재사용하는 것과 같은 수법이다.
private struct ConvLayer {
    int items, inCh, outCh, win, pad;
    Linear lin;              // (inCh*win) -> outCh

    float[] col, pre;        // per-sample: [items*inCh*win], [items*outCh]
    int _bcap;
    float[] colB, preB;      // 배치용

    int inSize()  const nothrow @nogc { return items * inCh; }
    int outSize() const nothrow @nogc { return items * outCh; }
    int K()       const nothrow @nogc { return inCh * win; }

    this(int items_, int inCh_, int outCh_, int win_) {
        items = items_; inCh = inCh_; outCh = outCh_; win = win_;
        pad = (win_ - 1) / 2;          // 창이 짝수면 왼쪽을 한 칸 덜 메운다
        lin = Linear(inCh_ * win_, outCh_);
        col = new float[items_ * inCh_ * win_]; col[] = 0f;
        pre = new float[items_ * outCh_];       pre[] = 0f;
    }

    private void _allocBatch(int B) {
        if (B <= _bcap) return;
        colB = new float[B * items * K]; colB[] = 0f;
        preB = new float[B * items * outCh]; preB[] = 0f;
        _bcap = B;
    }

    // 창 안의 값들을 한 줄로 늘어놓는다. 범위를 벗어나면 0.
    // c[i*K + ic*win + d] = x[(i+d-pad)*inCh + ic]
    private void _im2col(const(float)[] x, float[] c, int bOff, int cOff) nothrow {
        int k = K;
        _parChunkNT(items, (int lo, int hi) nothrow {
            foreach (i; lo .. hi) {
                auto dst = c[cOff + i*k .. cOff + (i+1)*k];
                foreach (d; 0..win) {
                    int p = i + d - pad;
                    if (p < 0 || p >= items)
                        foreach (ic; 0..inCh) dst[ic*win + d] = 0f;
                    else
                        foreach (ic; 0..inCh) dst[ic*win + d] = x[bOff + p*inCh + ic];
                }
            }
        });
    }

    // im2col 의 역. 모으는 쪽으로 쓴다 — 입력 위치 p 마다 "그 값을 가져다 쓴 창들"
    // 을 훑어서 합친다. 흩뿌리는 쪽으로 쓰면 이웃한 창이 같은 입력 칸에 동시에
    // 써서 레이스가 난다 (그리고 합산 순서가 흔들려 결정성도 깨진다).
    private void _col2im(const(float)[] dc, float[] dx, int cOff, int bOff) nothrow {
        int k = K;
        _parChunkNT(items, (int lo, int hi) nothrow {
            foreach (p; lo .. hi) {
                foreach (ic; 0..inCh) {
                    float s = 0f;
                    foreach (d; 0..win) {
                        int i = p - d + pad;      // 이 입력을 쓴 출력 위치
                        if (i >= 0 && i < items) s += dc[cOff + i*k + ic*win + d];
                    }
                    dx[bOff + p*inCh + ic] += s;
                }
            }
        });
    }

    void fwd(const(float)[] x, float[] y) {
        _im2col(x, col, 0, 0);
        lin.batchForward(col, pre, items);
        _parChunk(items, (int lo, int hi) {            // ReLU 는 안에서
            foreach (i; lo .. hi) foreach (oc; 0..outCh) {
                float v = pre[i*outCh + oc];
                y[i*outCh + oc] = v < 0f ? 0f : v;
            }
        });
    }

    // tmp 는 [items*outCh] 이상 (호출자 스크래치)
    void bwd(const(float)[] x, const(float)[] dy, float[] dx, float[] tmp) {
        _parChunk(items, (int lo, int hi) {
            foreach (i; lo .. hi) foreach (oc; 0..outCh) {
                int c = i*outCh + oc;
                tmp[c] = pre[c] > 0f ? dy[c] : 0f;
            }
        });
        // col 버퍼를 dCol 로 재사용한다 — 여기서부터 col 의 내용은 더 필요 없다.
        lin.batchAccum(col, tmp[0 .. items*outCh], col, items);
        _col2im(col, dx, 0, 0);
    }

    // 항목뿐 아니라 샘플도 전부 같은 가중치를 쓰는 독립 창이라 count=B*items 로
    // 한 번에 돌린다 (EachLayer 와 같은 이유). 단 im2col 은 샘플 경계를 넘지
    // 않아야 하므로 샘플마다 따로 만든다 — 넘으면 앞 샘플의 끝을 읽어버린다.
    void fwdBatch(const(float)[] X, float[] Y, int B) {
        _allocBatch(B);
        foreach (b; 0..B) _im2col(X, colB, b*items*inCh, b*items*K);
        int count = B * items;
        lin.batchForward(colB, preB, count);
        _parChunk(count, (int lo, int hi) {
            foreach (c; lo .. hi) foreach (oc; 0..outCh) {
                float v = preB[c*outCh + oc];
                Y[c*outCh + oc] = v < 0f ? 0f : v;
            }
        });
    }

    // dX 에 누적한다 (호출자가 먼저 비운다). tmp 는 [B*items*outCh] 이상.
    void bwdBatch(const(float)[] X, const(float)[] dY, float[] dX, float[] tmp, int B) {
        int count = B * items;
        _parChunk(count, (int lo, int hi) {
            foreach (c; lo .. hi) foreach (oc; 0..outCh) {
                int o = c*outCh + oc;
                tmp[o] = preB[o] > 0f ? dY[o] : 0f;
            }
        });
        lin.batchAccum(colB, tmp[0 .. count*outCh], colB, count);
        foreach (b; 0..B) _col2im(colB, dX, b*items*K, b*items*inCh);
    }

    void zeroGrad() nothrow @nogc { lin.zeroGrad(); }
    void step(Opt o, float lr, float decay = 0f) nothrow { lin.step(o, lr, decay); }
}

// ─────────────────────────────────────────────
// VICReg — "다 같은 값으로 뭉개지는 것"(collapse) 막기
// ─────────────────────────────────────────────
// jepa 처럼 "요약끼리 비교" 하는 학습은 요약기가 잔머리를 굴릴 수 있다: 입력이 뭐든
// 늘 같은 값을 뱉으면 예측이 항상 맞아서 손실이 0 이 된다. 아무것도 안 배우고 만점.
// 그래서 요약값들이 (1) 차원마다 실제로 값이 변하고 (2) 차원끼리 같은 정보를 중복해서
// 담지 않도록 벌점을 준다.
//
// 이 계산은 이 라이브러리의 다른 모든 계산과 성격이 다르다 — 한 샘플만 보고는 못 구하고
// 배치 전체를 한꺼번에 봐야 한다 ("이 묶음 안에서 서로 다른가" 가 질문이라서).
// 그래서 배치가 1이면 아예 건너뛴다.
private struct VicReg {
    float varW  = 25.0f;   // 분산 항 가중치
    float covW  = 1.0f;    // 공분산 항 가중치
    float gamma = 1.0f;    // 목표 표준편차 (이보다 크면 벌점 없음)

    private float[] z;     // [B*D] 중심화한 값
    private float[] mu, sd;// [D]
    private float[] C;     // [D*D] 공분산

    private void _alloc(int B, int D) {
        if (z.length  < cast(size_t)(B*D)) { z  = new float[B*D]; z[]  = 0f; }
        if (mu.length < cast(size_t)D)     { mu = new float[D];   mu[] = 0f;
                                             sd = new float[D];   sd[] = 0f; }
        if (C.length  < cast(size_t)(D*D)) { C  = new float[D*D]; C[]  = 0f; }
    }

    // S: [B*D] 요약값,  dS: [B*D] 여기에 기울기를 더한다. 손실값을 돌려준다.
    float grad(const(float)[] S, float[] dS, int B, int D) {
        if (B < 2 || D < 1) return 0f;   // 배치가 1이면 "서로 다른가" 를 잴 수 없다
        _alloc(B, D);
        immutable float invB = 1f / cast(float)B;
        immutable float n1   = cast(float)(B - 1);

        // 1) 차원마다 평균 -> 중심화 -> 표준편차
        _parChunk(D, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) {
                float s = 0f;
                foreach (b; 0..B) s += S[b*D + j];
                float m = s * invB;
                mu[j] = m;
                float ss = 0f;
                foreach (b; 0..B) { float d = S[b*D + j] - m; z[b*D + j] = d; ss += d*d; }
                sd[j] = sqrt(ss / n1 + 1e-4f);
            }
        });

        // 2) 공분산 C = Zᵀ Z / (B-1)
        _parChunk(D, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) foreach (k; 0..D) {
                float s = 0f;
                foreach (b; 0..B) s += z[b*D + j] * z[b*D + k];
                C[j*D + k] = s / n1;
            }
        });

        // 3) 손실 — 분산은 목표에 못 미친 만큼, 공분산은 대각선 밖(= 차원끼리 닮은 정도)
        float lv = 0f, lc = 0f;
        foreach (j; 0..D) {
            if (sd[j] < gamma) lv += gamma - sd[j];
            foreach (k; 0..D) if (k != j) { float c = C[j*D + k]; lc += c*c; }
        }
        lv /= D; lc /= D;

        // 4) 기울기.  평균(mu)이 S 에 의존하는 항은 Σ_b z[b][k] == 0 이라 정확히 0 이 되므로
        //    중심화한 값 z 로만 계산하면 된다.
        //      분산  : d/dS[b][j] = -varW/(D(B-1)) * z[b][j]/sd[j]   (sd[j] < gamma 일 때만)
        //      공분산: d/dS[b][j] = 4covW/(D(B-1)) * Σ_{k≠j} C[j][k] z[b][k]
        //    j 마다 스레드를 나눈다 — dS[b*D+j] 는 j 별로 겹치지 않고, 합산 순서도
        //    고정이라 직렬 실행과 비트 단위로 같다.
        immutable float gv = -varW / (cast(float)D * n1);
        immutable float gc = 4f * covW / (cast(float)D * n1);
        _parChunk(D, (int jlo, int jhi) {
            foreach (j; jlo .. jhi) {
                float vs = (sd[j] < gamma) ? gv / sd[j] : 0f;
                foreach (b; 0..B) {
                    float s = 0f;
                    foreach (k; 0..D) if (k != j) s += C[j*D + k] * z[b*D + k];
                    dS[b*D + j] += vs * z[b*D + j] + gc * s;
                }
            }
        });
        return varW * lv + covW * lc;
    }
}

private class Network {
    // 은닉층은 Linear(+ReLU) 와 Attn 이 섞일 수 있다.
    // kinds[i] == 0 이면 lins[slot[i]], 1 이면 attns[slot[i]]
    ubyte[]      kinds;
    int[]        slot;
    Linear[]     lins;
    AttnLayer[]  attns;
    EachLayer[]  eachs;
    LogicLayer[] logics;
    RnnLayer[]   rnns;
    ConvLayer[]  convs;
    Linear[]     heads;
    int          inputSz;

    // 모든 메모를 비운다. 에피소드 경계에서 부른다.
    void forget() nothrow @nogc { foreach (ref r; rnns) r.forget(); }
    bool hasMemo() const nothrow @nogc { return rnns.length > 0; }
    void snapshotMemo() nothrow @nogc { foreach (ref r; rnns) r.snapshotMemo(); }
    void restoreMemo()  nothrow @nogc { foreach (ref r; rnns) r.restoreMemo(); }

    int[]     _inSz, _outSz;
    float[][] _inp, _pre;
    float[]   _hout;
    float[][] _hd, _dHead;
    float[]   _dA, _dB, _dC;
    bool      fwdCached;

    // GPU 학습 경로가 가중치를 장치에 남겨두고 갈 수 있다. 그 뒤로 호스트 쪽
    // w/b/m/v 는 낡은 값이므로, 그것들을 만지기 전에 반드시 회수해야 한다.
    //
    // 호출처마다 "여기서 회수" 를 적어두는 방식은 하나만 빠뜨려도 조용히 틀린 답이
    // 나온다. 그래서 여기(가중치를 읽는 입구 전부)에 훅을 두고 BlackBoxAI 가 꽂는다
    // — 새 경로를 나중에 추가해도 forward/backward/step 중 하나는 반드시 지나가므로
    // 저절로 덮인다. 직접 필드를 훑는 save()/export_weights 만 따로 불러준다.
    void delegate() gpuSync;
    private void _sync() { if (gpuSync !is null) gpuSync(); }

    int layerCount() const nothrow @nogc { return cast(int) kinds.length; }

    // specKind[i]: 0=Linear(A=출력폭), 1=Attn(A=조각수, B=헤드수),
    //              2=Each(A=항목수, B=항목당 출력폭), 3=Logic(A=게이트 수),
    //              4=Rnn(A=메모 칸 수), 5=Conv(A=출력채널, B=창, C=항목수)
    this(int inputSz_, const(ubyte)[] specKind, const(int)[] specA, const(int)[] specB,
         const(int)[] specC, int[] headSizes) {
        inputSz = inputSz_;
        int prev = inputSz;
        foreach (i; 0..specKind.length) {
            if (specKind[i] == 0) {
                lins ~= Linear(prev, specA[i]);
                kinds ~= 0; slot ~= cast(int)(lins.length - 1);
                _inSz ~= prev; _outSz ~= specA[i];
                prev = specA[i];
            } else if (specKind[i] == 1) {
                attns ~= AttnLayer(prev, specA[i], specB[i]);
                kinds ~= 1; slot ~= cast(int)(attns.length - 1);
                _inSz ~= prev; _outSz ~= prev;      // 폭 유지
            } else if (specKind[i] == 5) {
                // Conv: specA=출력채널, specB=창, specC=항목수.
                // 입력채널은 들어온 폭에서 나눠서 구한다 (prev = 항목수 * 입력채널).
                int it = specC[i], oc = specA[i], wn = specB[i];
                convs ~= ConvLayer(it, prev / it, oc, wn);
                kinds ~= 5; slot ~= cast(int)(convs.length - 1);
                _inSz ~= prev; _outSz ~= it * oc;
                prev = it * oc;
            } else if (specKind[i] == 4) {
                // Rnn: specA=메모 칸 수 (= 출력 폭)
                rnns ~= RnnLayer(prev, specA[i]);
                kinds ~= 4; slot ~= cast(int)(rnns.length - 1);
                _inSz ~= prev; _outSz ~= specA[i];
                prev = specA[i];
            } else if (specKind[i] == 3) {
                // Logic: specA=게이트(출력) 수.
                // 바로 앞이 또 로직 층이면 입력이 이미 0~1 이라 다시 누르지 않는다.
                bool 앞도로직 = (i > 0 && specKind[i-1] == 3);
                logics ~= LogicLayer(prev, specA[i], !앞도로직);
                kinds ~= 3; slot ~= cast(int)(logics.length - 1);
                _inSz ~= prev; _outSz ~= specA[i];
                prev = specA[i];
            } else {
                // Each: specA=항목수, specB=항목당 출력폭
                int items = specA[i], ow = specB[i];
                eachs ~= EachLayer(items, prev / items, ow);
                kinds ~= 2; slot ~= cast(int)(eachs.length - 1);
                _inSz ~= prev; _outSz ~= items * ow;
                prev = items * ow;
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
        auto cc = new int[hiddenSizes.length];
        foreach (i, sz; hiddenSizes) { kk[i] = 0; aa[i] = sz; bb[i] = 0; cc[i] = 0; }
        this(inputSz_, kk, aa, bb, cc, headSizes);
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
        _dA = new float[maxSz]; _dB = new float[maxSz]; _dC = new float[maxSz];
        _dA[] = 0f; _dB[] = 0f; _dC[] = 0f;
        fwdCached = false;
    }

    void forward(const(float)[] x) {
        _sync();
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
                } else if (kinds[i] == 1) {
                    attns[slot[i]].fwd(_inp[i], _pre[i]);
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k];
                } else if (kinds[i] == 3) {
                    // 출력이 이미 0~1 이라 ReLU 를 걸지 않는다
                    logics[slot[i]].fwd(_inp[i], _pre[i]);
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k];
                } else if (kinds[i] == 5) {
                    convs[slot[i]].fwd(_inp[i], _pre[i]);   // ReLU 는 안에서
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k];
                } else if (kinds[i] == 4) {
                    // 여기서 메모가 한 스텝 나아간다 (tanh 출력이라 ReLU 안 건다)
                    rnns[slot[i]].fwd(_inp[i], _pre[i]);
                    if (i + 1 < n)
                        foreach (k; 0..oS) _inp[i+1][k] = _pre[i][k];
                    else
                        foreach (k; 0..oS) _hout[k] = _pre[i][k];
                } else {
                    eachs[slot[i]].fwd(_inp[i], _pre[i]);   // ReLU 는 안에서
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

    void backward() {
        _sync();
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
            } else if (kinds[i] == 1) {
                _dB[0..iS] = 0f;
                attns[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS]);
            } else if (kinds[i] == 3) {
                _dB[0..iS] = 0f;
                logics[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS]);
            } else if (kinds[i] == 4) {
                _dB[0..iS] = 0f;
                rnns[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS]);
            } else if (kinds[i] == 5) {
                _dB[0..iS] = 0f;
                convs[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS], _dC);
            } else {
                _dB[0..iS] = 0f;
                eachs[slot[i]].bwd(_inp[i], _dA[0..oS], _dB[0..iS], _dC);
            }
            foreach (k; 0..iS) _dA[k] = _dB[k];
        }
        fwdCached = false;
    }

    // ── 묶음(batch) 경로 — 가중치 행을 배치 전체에 재사용해 5배 빠르다 ──
    // Linear/Attn/Each 전부 지원 (kinds[i] 로 분기). 결과는 per-sample 과 동일.

    int       _bcap;
    float[][] _bInp, _bPre, _bDZ;    // 층별 [_bcap*inSz], [_bcap*outSz], dZ[_bcap*outSz]
    float[]   _bHout, _bDHout;       // [_bcap*houtSz]
    float[][] _bHd;                  // 헤드별 [_bcap*outSz]  (출력)
    float[][] _bDHead;               // 헤드별 [_bcap*outSz]  (호출자가 채움)
    // 1번째 층(kind 1 또는 2)의 dIn 을 버릴 때 쓰는 스크래치 — Linear 는 !첫층 으로
    // dIn 계산 자체를 건너뛸 수 있지만 Attn/Each 는 항상 계산하므로 버릴 곳이 필요하다.
    float[]   _bScratch0;

    void _allocBatch(int B) {
        if (B <= _bcap) return;
        int n = layerCount;
        _bInp = new float[][n]; _bPre = new float[][n]; _bDZ = new float[][n];
        foreach (i; 0..n) {
            _bInp[i] = new float[B*_inSz[i]];
            _bPre[i] = new float[B*_outSz[i]];
            _bDZ[i]  = new float[B*_outSz[i]];
        }
        int houtSz = n > 0 ? _outSz[n-1] : inputSz;
        _bHout = new float[B*houtSz]; _bDHout = new float[B*houtSz];
        _bHd = new float[][heads.length]; _bDHead = new float[][heads.length];
        foreach (i, ref h; heads) {
            _bHd[i]    = new float[B*h.outSz];
            _bDHead[i] = new float[B*h.outSz];
        }
        // 첫 층의 dIn 을 버릴 곳. 크기는 그 층의 "입력" 폭 기준이다 (_inSz[0]) — Attn 은
        // 입력폭==출력폭이라 티가 안 나지만 Each 가 첫 층이면 둘이 다르다. max 로 잡아 둔다.
        if (n > 0) _bScratch0 = new float[B * (_inSz[0] > _outSz[0] ? _inSz[0] : _outSz[0])];
        _bcap = B;
    }

    // X: B 개 입력 (각 inputSz). 헤드 출력은 _bHd 에.
    void forwardBatch(float[][] X, int B) {
        _sync();
        int n = layerCount;
        int houtSz = n > 0 ? _outSz[n-1] : inputSz;
        if (n == 0) {
            foreach (b; 0..B) foreach (k; 0..inputSz) _bHout[b*houtSz + k] = X[b][k];
        } else {
            foreach (b; 0..B) foreach (k; 0..inputSz) _bInp[0][b*inputSz + k] = X[b][k];
            foreach (i; 0..n) {
                int oS = _outSz[i];
                bool 마지막 = (i + 1 == n);
                auto inp = _bInp[i];
                // 다음 층 "입력" 폭 = 이 층 출력 폭(oS) — _inSz[i+1] 와 항상 같다 (Attn 은
                // 폭을 유지, Each 는 _outSz 를 items*outW 로 이미 그렇게 잡아둔다). 그래서
                // 마지막이 아니면 _bInp[i+1] 에, 마지막이면 _bHout 에 stride=oS 로 그대로 써도
                // Attn/Each 의 fwdBatch(count=B) 출력과 레이아웃이 정확히 맞는다.
                auto 다음 = 마지막 ? _bHout : _bInp[i+1];
                final switch (kinds[i]) {
                    case 0:
                        auto L = &lins[slot[i]];
                        auto pre = _bPre[i];
                        L.batchForward(inp, pre, B);
                        int nextW = 마지막 ? houtSz : oS;
                        _parChunk(B, (int blo, int bhi) {           // ReLU — b 를 스레드로 나눔
                            foreach (b; blo .. bhi) foreach (k; 0..oS) {
                                float v = pre[b*oS + k];
                                다음[b*nextW + k] = v < 0f ? 0f : v;
                            }
                        });
                        break;
                    case 1:
                        attns[slot[i]].fwdBatch(inp, 다음, B);
                        break;
                    case 2:
                        eachs[slot[i]].fwdBatch(inp, 다음, B);
                        break;
                    case 3:
                        logics[slot[i]].fwdBatch(inp, 다음, B);
                        break;
                    case 4:
                        // 이 층만 b 를 차례로 돈다 (메모가 이어져야 해서).
                        // 앞뒤 층은 그대로 묶여서 돈다.
                        rnns[slot[i]].fwdBatch(inp, 다음, B);
                        break;
                    case 5:
                        convs[slot[i]].fwdBatch(inp, 다음, B);
                        break;
                }
            }
        }
        foreach (hi, ref head; heads)
            head.batchForward(_bHout[0..B*houtSz], _bHd[hi], B);
    }

    // _bDHead (헤드별 [B*outSz], 호출자가 채움) -> 가중치 기울기 누적. step 은 밖에서.
    // dInput != null 이면 입력에 대한 기울기도 거기에 낸다 ([B*inputSz]) — jepa 처럼
    // 앞 신경망으로 기울기를 계속 흘려보내야 할 때 쓴다. 평소(null)엔 버린다.
    void backwardBatch(int B, float[] dInput = null) {
        _sync();
        int n = layerCount;
        int houtSz = n > 0 ? _outSz[n-1] : inputSz;
        _bDHout[0 .. B*houtSz] = 0f;
        // 여러 헤드가 같은 _bDHout 를 공유해서 더해 넣는다 (zeroDIn=false — 위에서 이미 0으로 채움)
        foreach (hi, ref head; heads)
            head.batchAccum(_bHout[0..B*houtSz], _bDHead[hi][0..B*head.outSz],
                             _bDHout[0..B*houtSz], B, true, false);
        if (n == 0) {
            if (dInput !is null) dInput[0 .. B*inputSz] = _bDHout[0 .. B*inputSz];
            return;
        }
        for (int i = n-1; i >= 0; i--) {
            int oS = _outSz[i], iS = _inSz[i];
            bool 첫층 = (i == 0);
            bool 마지막 = (i + 1 == n);
            auto inp = _bInp[i];
            // 이 층 "출력" 으로 들어오는 기울기 — 마지막이면 헤드에서 온 _bDHout, 아니면
            // 다음 층이 앞서(i+1 을 먼저 처리했으므로) _bDZ[i] 에 이미 써둔 dIn.
            auto dOut = 마지막 ? _bDHout[0..B*houtSz] : _bDZ[i][0..B*oS];
            // Linear 는 (dInput 을 안 받는) 첫층이면 dIn 계산 자체를 건너뛸 수 있지만
            // Attn/Each 는 항상 계산하므로, 그때는 버릴 스크래치(_bScratch0)를 준다.
            // dInput 이 있으면 첫층의 dIn 은 버리지 않고 거기로 내보낸다.
            auto dIn = !첫층      ? _bDZ[i-1]
                     : (dInput !is null ? dInput[0..B*iS] : _bScratch0[0..B*iS]);
            final switch (kinds[i]) {
                case 0:
                    auto L = &lins[slot[i]];
                    auto pre = _bPre[i]; auto dZ = _bDZ[i];
                    // dZ[b][k] = 들어온 기울기 * (pre>0)  — b 로 나눔
                    _parChunk(B, (int blo, int bhi) {
                        foreach (b; blo .. bhi) foreach (k; 0..oS) {
                            dZ[b*oS + k] = (pre[b*oS + k] > 0f) ? dOut[b*oS + k] : 0f;
                        }
                    });
                    bool 필요 = !첫층 || dInput !is null;
                    L.batchAccum(inp, dZ, 필요 ? dIn : null, B, 필요);
                    break;
                case 1:
                    // AttnLayer.bwdBatch 는 (per-sample bwd() 와 마찬가지로) dX 에 누적만
                    // 하고 스스로 비우지 않는다 — Linear/Each 의 batchAccum 은 내부에서
                    // zeroDIn=true 로 스스로 비우는 것과 다르다. 호출자가 먼저 비워야 한다.
                    dIn[] = 0f;
                    attns[slot[i]].bwdBatch(inp, dOut, dIn, B);
                    break;
                case 2:
                    // tmp 스크래치로 _bDZ[i] 를 재사용한다 — 마지막이 아니면 dOut 과 같은
                    // 버퍼지만, EachLayer.bwdBatch 는 tmp[c]=f(dY[c]) 형태(같은 인덱스만
                    // 읽고 쓰는 원소별 변환)라 그 자리에서 덮어써도 안전하다.
                    eachs[slot[i]].bwdBatch(inp, dOut, dIn, _bDZ[i], B);
                    break;
                case 3:
                    // LogicLayer.bwdBatch 도 (Attn 처럼) dX 에 누적만 한다
                    dIn[] = 0f;
                    logics[slot[i]].bwdBatch(inp, dOut, dIn, B);
                    break;
                case 4:
                    dIn[] = 0f;          // bwdBatch 는 누적만 한다
                    rnns[slot[i]].bwdBatch(inp, dOut, dIn, B);
                    break;
                case 5:
                    // col2im 이 더하기만 하므로 호출자가 먼저 비운다.
                    // tmp 로 _bDZ[i] 를 쓴다 (크기 B*_outSz[i] = B*items*outCh).
                    dIn[] = 0f;
                    convs[slot[i]].bwdBatch(inp, dOut, dIn, _bDZ[i], B);
                    break;
            }
        }
    }

    void zeroGrad() nothrow @nogc {
        foreach (ref h; lins)  h.zeroGrad();
        foreach (ref a; attns) a.zeroGrad();
        foreach (ref e; eachs) e.zeroGrad();
        foreach (ref g; logics) g.zeroGrad();
        foreach (ref r; rnns)  r.zeroGrad();
        foreach (ref v; convs) v.zeroGrad();
        foreach (ref h; heads) h.zeroGrad();
    }

    // nothrow 를 지키려고 여기서만 try 로 감싼다 (훅은 GPU 다운로드를 한다).
    //
    // 감쇠는 Linear 가중치 행렬에만 건다. logic 층의 게이트 선택 가중치는 softmax
    // 로짓이라 0 으로 당기면 "16가지를 똑같이 섞어라" 가 되는데, 그건 가중치 감쇠가
    // 뜻하는 것과 다른 종류의 압력이라 놀랄 일이 된다 — 그래서 뺀다.
    void step(Opt opt, float lr, float decay = 0f) nothrow {
        try { _sync(); } catch (Throwable) {}
        foreach (ref h; lins)  h.step(opt, lr, decay);
        foreach (ref a; attns) a.step(opt, lr, decay);
        foreach (ref e; eachs) e.step(opt, lr, decay);
        foreach (ref g; logics) g.step(opt, lr);
        foreach (ref r; rnns)  r.step(opt, lr, decay);
        foreach (ref v; convs) v.step(opt, lr, decay);
        foreach (ref h; heads) h.step(opt, lr, decay);
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

// GPU 학습 경로가 층 하나에 대해 들고 있는 버퍼들. 호출마다 새로 만들지 않고
// BlackBoxAI 가 캐시한다 — clCreateBuffer 는 싸지 않고, 큰 망이면 호출당 수십 개다.
private struct GLayer {
    gpucl.GpuBuf w, b, gradW, gradB, mW, vW, mB, vB, pre, act, dZ, dIn;
    int inSz, outSz;
    Linear* host;

    void free() nothrow {
        foreach (ref buf; [&w,&b,&gradW,&gradB,&mW,&vW,&mB,&vB,&pre,&act,&dZ,&dIn])
            gpucl.freeBuf(*buf);
    }
}

class BlackBoxAI {
    string     name;
    string[][] actionLists;   // 헤드별 액션 이름 (cos 헤드는 빈 배열)
    bool[]     cosModes;      // 헤드별 cos 여부
    int[]      outSizes;      // 헤드별 출력 개수
    int[]      hiddenSizes;   // Linear 층 폭 (호환용)
    ubyte[]    layKind;       // 0=Linear, 1=Attn
    int[]      layA, layB, layC;  // Linear: A=폭 / Attn: A=조각수,B=헤드수 /
                                  // Conv: A=출력채널,B=창,C=항목수
    float      lr = 0.01f;
    float      cosSigma = 1.0f;   // cos 헤드 탐험 폭
    float      entropy  = 0.01f;  // 이산 헤드 엔트로피 보너스
    float      decay    = 0.0f;   // 가중치 감쇠 (0 = 끔). Linear 가중치 행렬에만 걸린다
    float      temp     = 1.0f;   // 고르기 샘플링 온도. 낮으면 과감하지 않게, 0 이면 최선만
    Opt        opt;
    string     file;

    Network net;
    bool    ready;

    private float[]   _envBuf;
    private float[][] _probBufs;   // 헤드별 확률 스크래치
    private int[][]   _maskBufs;   // 헤드별 마스크 스크래치

    // GPU 학습 경로 캐시 — 버퍼와 호스트 평탄화 배열을 호출 간에 재사용한다.
    // 예전엔 호출마다 전부 새로 잡았는데, [512,1024,1024] 기준 GC 할당만 40MB 가
    // 넘어서 호출당 30ms 가 그냥 나갔다 (전송 자체는 2ms 남짓이다).
    private GLayer[]      _gpuLayers;
    private gpucl.GpuBuf  _gpuX;
    private int           _gpuCapB;      // 캐시된 버퍼가 감당하는 배치 크기
    private float[]       _gpuFlat;      // 가중치 평탄화용 (층 중 제일 큰 것 기준)
    private float[]       _gpuHeadBuf;   // 헤드 출력/기울기용 [B]

    // 가중치와 옵티마이저 상태의 최신본이 GPU 에 있는가. GPU 로 연달아 학습할 때
    // 이걸 매 호출 내렸다 올리는 것이 남아있던 고정비의 대부분이었다.
    // 호스트가 만지려 할 때만 회수한다 — 그 시점을 Network 의 훅이 잡아준다.
    private bool _gpuResident = false;
    private bool _gpuSyncing  = false;   // 회수 도중 훅이 다시 불리는 것 방지

    // 신경망을 거치지 않고 가중치를 직접 훑는 쪽(export_weights)이 부른다.
    void syncFromGpu() { _gpuSyncBack(); }

    // GPU 버퍼를 놓아준다. 모델을 버릴 때(파이썬 캡슐 소멸) 부른다.
    // 내장 GPU 는 이 메모리가 곧 시스템 RAM 이고, 큰 망이면 100MB 를 넘는다.
    // 예전엔 아무도 안 불러서 프로세스가 끝날 때까지 붙잡고 있었다.
    void releaseGpu() nothrow {
        try { _gpuSyncBack(); } catch (Throwable) {}   // 남은 학습 결과는 회수하고
        foreach (ref gl; _gpuLayers) gl.free();
        gpucl.freeBuf(_gpuX);
        _gpuLayers = null;
        _gpuCapB = 0;
        _gpuResident = false;
    }

    // GPU 에 남아있는 w, b, m, v 를 호스트로 회수한다. Network._sync 가 부른다.
    private void _gpuSyncBack() {
        if (!_gpuResident || _gpuSyncing) return;
        _gpuSyncing = true;
        _gpuResident = false;      // 실패하더라도 두 번 시도하지 않는다
        scope(exit) _gpuSyncing = false;
        foreach (ref gl; _gpuLayers) {
            auto L = gl.host;
            if (L is null) continue;
            size_t wn = cast(size_t) gl.outSz * gl.inSz;
            auto flat = _gpuFlat[0 .. wn];
            if (!gpucl.download(gl.w, flat)) return;
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) L.w[j][k] = flat[j*gl.inSz+k];
            if (!gpucl.download(gl.mW, flat)) return;
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) L.mW[j][k] = flat[j*gl.inSz+k];
            if (!gpucl.download(gl.vW, flat)) return;
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) L.vW[j][k] = flat[j*gl.inSz+k];
            if (!gpucl.download(gl.b, L.b) || !gpucl.download(gl.mB, L.mB) ||
                !gpucl.download(gl.vB, L.vB)) return;
        }
    }

    int nHeads() const nothrow @nogc { return cast(int) outSizes.length; }

    this(string name, int inputSz, ubyte[] lk, int[] la, int[] lb, int[] lc, int[] heads,
         string[][] actions, bool[] cos,
         Opt opt = Opt.adam, float sigma = 1.0f, float ent = 0.01f,
         float lr_ = 0.01f, float decay_ = 0.0f, float temp_ = 1.0f) {
        this.name = name; this.opt = opt;
        cosSigma = sigma; entropy = ent;
        lr = lr_; decay = decay_; temp = temp_;
        layKind = lk.dup; layA = la.dup; layB = lb.dup; layC = lc.dup;
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
                if      (layKind[i] == 0) r ~= to!string(layA[i]);
                else if (layKind[i] == 1) r ~= "attn(" ~ to!string(layA[i]) ~ "," ~ to!string(layB[i]) ~ ")";
                else if (layKind[i] == 3) r ~= "logic(" ~ to!string(layA[i]) ~ ")";
                else if (layKind[i] == 4) r ~= "memory(" ~ to!string(layA[i]) ~ ")";
                else if (layKind[i] == 5) r ~= "conv(" ~ to!string(layA[i]) ~ ","
                                             ~ to!string(layB[i]) ~ "/" ~ to!string(layC[i]) ~ ")";
                else                      r ~= "each(" ~ to!string(layB[i]) ~ ")";
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
                    if (layKind[i] != lk[i] || layA[i] != la[i] || layB[i] != lb[i]
                        || layC[i] != lc[i]) { same = false; break; }
                if (same) foreach (i, sz; outSizes) if (sz != heads[i]) { same = false; break; }
                if (!same) {
                    string 저장된 = 층설명();
                    layKind = lk.dup; layA = la.dup; layB = lb.dup; layC = lc.dup;
                    hiddenSizes = [];
                    foreach (i; 0..lk.length) if (lk[i] == 0) hiddenSizes ~= la[i];
                    알림(" [%s] 저장된 구조 %s 가 요청한 %s 와 다릅니다. 새로 만듭니다.",
                             name, 저장된, 층설명());
                    outSizes = heads.dup;
                    actionLists = actions.dup; cosModes = cos.dup;
                    net = null; ready = false;
                } else {
                    ready = true;
                    알림(" [%s] 이전 학습 데이터를 불러왔습니다.", name);
                }
            } catch (Exception e) {
                알림(" [%s] 불러오기 실패 (%s). 새로 시작합니다.", name, e.msg);
                ready = false;
                // 못 읽은 파일을 그대로 두면 다음 save() 가 덮어쓴다. 그런데
                // "읽을 수 없다" 가 "버려도 된다" 는 아니다 — 더 새 버전에서
                // 저장했거나, 예전 포맷이라 change() 가 필요한 경우일 수 있다.
                // 옆으로 치워두고 새로 시작한다.
                //
                // 구조가 달라서 새로 만드는 경우(위 !same)는 사용자가 층을 바꾼
                // 것이라 의도된 초기화다 — 거기선 백업하지 않는다 (실험할 때마다
                // .bak 이 쌓인다).
                try {
                    string bak;
                    foreach (i; 0..10) {
                        bak = file ~ ".bak" ~ (i == 0 ? "" : to!string(i + 1));
                        if (!exists(bak)) break;
                        bak = null;
                    }
                    if (bak is null)
                        알림(" [%s] 원본을 옮기지 못했습니다 (.bak 자리가 다 찼습니다)."
                             ~ " %s 를 직접 치워주세요.", name, file);
                    else {
                        rename(file, bak);
                        알림(" [%s] 원본은 %s 로 옮겨뒀습니다.", name, bak);
                    }
                } catch (Exception e2) {
                    알림(" [%s] 원본을 옮기지 못했습니다 (%s). %s 가 덮어써질 수"
                         ~ " 있으니 직접 치워주세요.", name, e2.msg, file);
                }
            }
        }
        if (!ready) {
            net = new Network(inputSz, layKind, layA, layB, layC, outSizes);
            알림(" [%s] 새로 생성되었습니다. %s->%s", name, 층설명(), outSizes);
            ready = true;
        }
        // GPU 가 가중치를 장치에 남겨두고 갈 수 있으므로, 신경망을 읽는 입구마다
        // 회수가 걸리도록 훅을 꽂는다. net 이 바뀌면(load 포함) 다시 꽂아야 한다.
        net.gpuSync = &_gpuSyncBack;
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
        _pickHere(legal, chosen, value);
    }

    private void _pickHere(string[][] legal, int[] chosen, float[] value) {
        net.forward(_envBuf);
        foreach (h; 0..nHeads) {
            if (cosModes[h]) {
                int unit = outSizes[h] > 1 ? uniform(0, outSizes[h], rng) : 0;
                chosen[h] = unit;
                value[h] = net._hd[h][unit] + cosSigma * gauss();
            } else {
                auto lg = (h < legal.length) ? legal[h] : null;
                int mlen = maskOf(h, lg);
                // 온도: 점수를 temp 로 나눠서 softmax 에 넣는다. 1 보다 작으면 차이가
                // 벌어져 제일 좋은 것만 고르게 되고, 크면 차이가 좁혀져 골고루 찔러본다.
                // 0 이하면 아예 최선만 (무작위성 없음 — 배포용).
                if (temp <= 0f) {
                    int best = 0;
                    foreach (k; 1..mlen)
                        if (net._hd[h][_maskBufs[h][k]] > net._hd[h][_maskBufs[h][best]]) best = k;
                    chosen[h] = _maskBufs[h][best];
                    value[h]  = 0f;
                    continue;
                }
                immutable float invT = 1f / temp;
                foreach (k; 0..mlen) _probBufs[h][k] = net._hd[h][_maskBufs[h][k]] * invT;
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
        _predHere(legal, chosen, value);
    }

    private void _predHere(string[][] legal, int[] chosen, float[] value) {
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
        // memory 층이 있으면 이 스텝들을 **다시** 순전파하게 되는데, 수집할 때
        // rl() 들이 메모를 이미 끝까지 밀어놨다. 그 첫 스텝이 보던 메모로
        // 되돌려놓고 재생한다 (RnnLayer.hSnap 주석 참고).
        if (net.hasMemo()) net.restoreMemo();
        // 배치 크기로 나눠 평균 기울기를 쓴다.
        // (합산만 하면 배치가 커질수록 갱신 폭이 커져 발산한다)
        immutable float inv = 1.0f / cast(float) inputs.length;

        // ── 묶음 경로 — sl() 이 쓰는 것과 같은 코어(forwardBatch/backwardBatch).
        // 기울기 계산 자체는 아래 per-sample 경로와 한 글자도 다르지 않다.
        // 읽는 곳이 _hd -> _bHd, 쓰는 곳이 _dHead -> _bDHead 로 바뀔 뿐이다.
        //
        // 1개짜리는 아래 직렬 경로로 보낸다. 배치 버퍼 준비·층별 디스패치 값이
        // 재사용할 게 없는 상태에선 그냥 손해다 (측정: [64,256,256] 에서
        // 0.94ms -> 0.85ms). 배치=1 온라인 학습이 이 라이브러리의 핵심 용도라
        // 그 경로를 제일 짧게 둔다.
        if (!_noBatch && inputs.length > 1) {
            int B = cast(int) inputs.length;
            net._allocBatch(B);
            net.forwardBatch(inputs, B);
            foreach (h; 0..nHeads) {
                int oS = outSizes[h];
                net._bDHead[h][0 .. B*oS] = 0f;
                foreach (i; 0..B) {
                    float sc = scores[i][h];
                    if (sc != sc) continue;   // NaN -> 이 헤드는 학습 안 함
                    sc *= inv;
                    auto hd  = net._bHd[h][i*oS .. i*oS + oS];
                    auto dst = net._bDHead[h][i*oS .. i*oS + oS];
                    if (cosModes[h]) {
                        int u = chosen[i][h] < oS ? chosen[i][h] : 0;
                        float mu = hd[u];
                        float gr = -sc * (values[i][h] - mu) / (cosSigma * cosSigma);
                        if (gr >  GCLIP) gr =  GCLIP;      // 발산 방지
                        if (gr < -GCLIP) gr = -GCLIP;
                        dst[u] = gr;
                    } else {
                        foreach (k; 0..oS) _probBufs[h][k] = hd[k];
                        softmaxInPlace(_probBufs[h][0..oS]);
                        float Hh = 0f;
                        foreach (k; 0..oS) {
                            float pk = _probBufs[h][k];
                            if (pk > 1e-8f) Hh -= pk * log(pk);
                        }
                        foreach (k; 0..oS) {
                            float pk = _probBufs[h][k];
                            float g  = sc * (pk - (k == chosen[i][h] ? 1f : 0f));
                            if (entropy != 0f && pk > 1e-8f)
                                g += entropy * inv * pk * (log(pk) + Hh);
                            dst[k] = g;
                        }
                    }
                }
            }
            net.backwardBatch(B);
            net.step(opt, lr, decay);
            // 재생이 끝난 메모가 다음 재생의 출발점이 된다
            if (net.hasMemo()) net.snapshotMemo();
            return;
        }

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
        net.step(opt, lr, decay);
        if (net.hasMemo()) net.snapshotMemo();
    }

    // ── 지도학습 1스텝 (헤드별 정답; 이산은 인덱스, cos 는 목표값) ──
    void slBatch(float[] input, string[][] legal, int[] ansIdx, float[] ansVal, bool[] use) {
        bool any = false;
        foreach (u; use) if (u) { any = true; break; }
        if (!any) return;
        foreach (k; 0..input.length) _envBuf[k] = input[k];
        _slHere(legal, ansIdx, ansVal, use);
    }

    private void _slHere(string[][] legal, int[] ansIdx, float[] ansVal, bool[] use) {
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
        net.step(opt, lr, decay);
        if (net.hasMemo()) net.snapshotMemo();
    }

    // GPU 로 넘길 만한 크기인지 판단. 이 라이브러리는 배치=1 온라인 학습이 핵심
    // 사용처라(README 벤치마크 참고 — [1,128,3] 이 PyTorch 보다 9µs vs 707µs 로
    // 압도적으로 빠른 이유가 정확히 "배치/디스패치 오버헤드 없음"), GPU 디스패치
    // 오버헤드(보통 수십~수백 µs)가 오히려 손해가 되는 경우가 훨씬 많다. 그래서
    // 총 FLOPs 와 배치 크기 둘 다 문턱값을 넘을 때만 시도한다.
    private bool _gpuWorthTrying(int n, int B) const {
        if (_gpuMode == "0") return false;
        if (_gpuMode == "1") return true;
        if (B < _gpuMinB) return false;
        long flops = 0;
        foreach (i; 0..n) flops += 2L * net._inSz[i] * net._outSz[i];
        flops *= B;
        return flops >= _gpuMinFlops;
    }

    // GPU 버퍼와 호스트 평탄화 배열을 준비한다. 모양이 그대로면 아무것도 안 한다
    // — 재사용이 요점이다 (호출마다 새로 잡으면 그 비용이 계산보다 크다).
    // 배치가 줄어든 경우는 그냥 쓴다 (버퍼가 넉넉하니까). 커지면 다시 잡는다.
    private bool _gpuEnsure(int n, int B) {
        bool 그대로 = (_gpuLayers.length == n + 1) && (B <= _gpuCapB);
        if (그대로) {
            // 층 폭이 달라졌으면(모델이 바뀌었으면) 다시 잡아야 한다
            int prev = net.inputSz;
            foreach (i; 0..n) {
                if (_gpuLayers[i].inSz != prev || _gpuLayers[i].outSz != net._outSz[i])
                    { 그대로 = false; break; }
                prev = net._outSz[i];
            }
            if (그대로 && (_gpuLayers[n].inSz != prev
                          || _gpuLayers[n].outSz != net.heads[0].outSz)) 그대로 = false;
        }
        if (그대로) return true;

        // 버퍼를 다시 잡기 전에 GPU 에 있던 옵티마이저 상태를 회수한다.
        // 안 그러면 지금까지의 관성이 통째로 사라진다.
        _gpuSyncBack();
        foreach (ref gl; _gpuLayers) gl.free();
        gpucl.freeBuf(_gpuX);
        _gpuLayers = new GLayer[n + 1];

        int inputSz = net.inputSz;
        _gpuX = gpucl.allocBuf(cast(size_t) B * inputSz);
        if (!_gpuX.valid) return false;

        size_t maxFlat = cast(size_t) B * inputSz;   // 입력 평탄화에도 같은 버퍼를 쓴다
        int prev = inputSz;
        foreach (i; 0..n + 1) {
            auto gl = &_gpuLayers[i];
            gl.inSz  = prev;
            gl.outSz = (i < n) ? net._outSz[i] : net.heads[0].outSz;
            size_t wsz = cast(size_t) gl.outSz * gl.inSz;
            if (wsz > maxFlat) maxFlat = wsz;
            gl.w  = gpucl.allocBuf(wsz);        gl.b  = gpucl.allocBuf(gl.outSz);
            gl.gradW = gpucl.allocBuf(wsz);     gl.gradB = gpucl.allocBuf(gl.outSz);
            gl.mW = gpucl.allocBuf(wsz);        gl.vW = gpucl.allocBuf(wsz);
            gl.mB = gpucl.allocBuf(gl.outSz);   gl.vB = gpucl.allocBuf(gl.outSz);
            gl.pre = gpucl.allocBuf(cast(size_t) B * gl.outSz);
            gl.dZ  = gpucl.allocBuf(cast(size_t) B * gl.outSz);
            gl.dIn = gpucl.allocBuf(cast(size_t) B * gl.inSz);
            // 헤드 뒤에는 ReLU 가 없어서 act 가 필요 없다
            if (i < n) gl.act = gpucl.allocBuf(cast(size_t) B * gl.outSz);
            if (!gl.w.valid||!gl.b.valid||!gl.gradW.valid||!gl.gradB.valid||!gl.mW.valid||
                !gl.vW.valid||!gl.mB.valid||!gl.vB.valid||!gl.pre.valid||!gl.dZ.valid||
                !gl.dIn.valid||(i < n && !gl.act.valid)) return false;
            prev = gl.outSz;
        }

        _gpuFlat = new float[maxFlat];       _gpuFlat[] = 0f;
        _gpuHeadBuf = new float[cast(size_t) B * net.heads[0].outSz];
        _gpuHeadBuf[] = 0f;
        _gpuCapB = B;
        return true;
    }

    // GPU 배치 학습 경로 — v1 범위: 순수 Linear 망 + 헤드 1개 + cos(실수) 출력만
    // (attn/each, 다중 헤드, pick 헤드는 아직 없음 — CPU 배치 경로로 자연히 폴백됨).
    // 매 호출마다 가중치를 업로드/다운로드한다 (가중치 상주 캐싱은 다음 단계 최적화
    // 대상으로 남겨둠 — sl() 호출마다 step() 이 돌아 매번 갱신되므로 상주시키려면
    // CPU 쪽 호출과의 동기화 규칙이 더 필요하다). 실패하면 false — 호출자가 CPU
    // 배치 경로로 그대로 이어간다.
    private bool _gpuSlMany(float[][] inputs, float[][] ansVal, bool[][] use, int B) {
        if (nHeads != 1 || !cosModes[0]) return false;
        foreach (k; net.kinds) if (k != 0) return false;

        int n = net.layerCount;
        // MYML_GPU=0 이면 여기서 완전히 끝나야 한다 — gpucl.available() 이 OpenCL.dll
        // 을 프로브하므로, 문턱값 체크보다 먼저 와야 "완전 비활성" 이 진짜로 지켜진다.
        if (!_gpuWorthTrying(n, B)) return false;
        if (!gpucl.available()) return false;

        int inputSz = net.inputSz;
        if (!_gpuEnsure(n, B)) return false;
        auto layers = _gpuLayers;
        auto xBuf = _gpuX;

        foreach (b; 0..B) foreach (k; 0..inputSz) _gpuFlat[b*inputSz+k] = inputs[b][k];
        if (!gpucl.upload(xBuf, _gpuFlat[0 .. B*inputSz])) return false;

        gpucl.GpuBuf curIn = xBuf; int curInSz = inputSz;

        // 가중치·옵티마이저 상태를 올린다. 평탄화 버퍼 하나를 돌려쓴다 (채우고 바로
        // 올리고 다시 채운다) — 예전엔 배열 셋을 층마다 새로 잡았다.
        //
        // 이미 GPU 에 남아있으면(_gpuResident) 통째로 건너뛴다. 연달아 GPU 로
        // 학습할 때 이 전송이 남아있던 고정비의 대부분이었다
        // (측정: [1024,2048,2048] 에서 190ms 중 75ms).
        bool uploadLinear(ref GLayer gl, Linear* L) {
            gl.host = L;
            if (_gpuResident) return true;
            size_t wn = cast(size_t) gl.outSz * gl.inSz;
            auto flat = _gpuFlat[0 .. wn];
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) flat[j*gl.inSz+k] = L.w[j][k];
            if (!gpucl.upload(gl.w, flat)) return false;
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) flat[j*gl.inSz+k] = L.mW[j][k];
            if (!gpucl.upload(gl.mW, flat)) return false;
            foreach (j; 0..gl.outSz) foreach (k; 0..gl.inSz) flat[j*gl.inSz+k] = L.vW[j][k];
            if (!gpucl.upload(gl.vW, flat)) return false;
            return gpucl.upload(gl.b, L.b) && gpucl.upload(gl.mB, L.mB)
                && gpucl.upload(gl.vB, L.vB);
        }

        // ── 은닉층 순전파 ──
        foreach (i; 0..n) {
            auto L = &net.lins[net.slot[i]];
            auto gl = &layers[i];
            size_t wsz = cast(size_t) gl.outSz * gl.inSz;
            // gradW/gradB 는 커널이 "+=" 로 누적하므로 매 호출 0으로 되돌려야 한다
            // (버퍼를 재사용하니 지난 호출의 값이 그대로 남아있다).
            if (!gpucl.zeroBuf(gl.gradW, wsz) || !gpucl.zeroBuf(gl.gradB, gl.outSz)) return false;
            if (!uploadLinear(*gl, L)) return false;
            if (!gpucl.linearForward(gl.w, gl.b, curIn, gl.pre, gl.inSz, gl.outSz, B)) return false;
            if (!gpucl.relu(gl.pre, gl.act, B*gl.outSz)) return false;
            curIn = gl.act; curInSz = gl.outSz;
        }

        // ── 헤드 순전파 (ReLU 없음) ──
        auto head = &net.heads[0];
        auto hl = &layers[n];
        size_t hwsz = cast(size_t) hl.outSz * hl.inSz;
        if (!gpucl.zeroBuf(hl.gradW, hwsz) || !gpucl.zeroBuf(hl.gradB, hl.outSz)) return false;
        if (!uploadLinear(*hl, head)) return false;
        if (!gpucl.linearForward(hl.w, hl.b, curIn, hl.pre, hl.inSz, hl.outSz, B)) return false;

        // ── 손실 기울기 (cos: (예측-정답)/B) — 호스트에서 계산 (B개 스칼라라 저렴) ──
        auto headBuf = _gpuHeadBuf[0 .. B*hl.outSz];
        if (!gpucl.download(hl.pre, headBuf)) return false;
        immutable float inv = 1.0f / cast(float) B;
        foreach (b; 0..B)
            headBuf[b] = use[b][0] ? (headBuf[b] - ansVal[b][0]) * inv : 0f;
        if (!gpucl.upload(hl.dZ, headBuf)) return false;

        // ── 헤드 역전파 ──
        if (!gpucl.linearBackwardGradW(curIn, hl.dZ, hl.gradW, hl.gradB, hl.inSz, hl.outSz, B))
            return false;
        if (!gpucl.linearBackwardDInput(hl.w, hl.dZ, hl.dIn, hl.inSz, hl.outSz, B))
            return false;
        auto dOut = hl.dIn; int dOutSz = hl.inSz;

        // ── 은닉층 역전파 (역순) ──
        for (int i = n-1; i >= 0; i--) {
            auto gl = &layers[i];
            if (!gpucl.reluBackward(gl.pre, dOut, gl.dZ, B*gl.outSz)) return false;
            auto inp = (i == 0) ? xBuf : layers[i-1].act;
            if (!gpucl.linearBackwardGradW(inp, gl.dZ, gl.gradW, gl.gradB, gl.inSz, gl.outSz, B))
                return false;
            if (!gpucl.linearBackwardDInput(gl.w, gl.dZ, gl.dIn, gl.inSz, gl.outSz, B))
                return false;
            dOut = gl.dIn; dOutSz = gl.inSz;
        }

        // ── Adam step (GPU 위에서) + 갱신된 가중치/옵티마이저 상태 다운로드 ──
        // t 는 전부 성공한 뒤에 올린다. 중간에 실패하면 호출자가 CPU 경로로 다시
        // 도는데, 그때 t 만 앞서 있으면 Adam 의 편향보정이 한 스텝 어긋난다.
        // (가중치 쪽은 안전하다 — 실패하면 _gpuResident 를 안 세우므로 호스트 값이
        //  그대로고, GPU 버퍼의 부분 갱신은 다음 업로드가 덮는다.)
        float bc1 = 1f, bc2 = 1f;
        foreach (ref gl; layers) {
            auto L = gl.host;
            int t1 = L.t + 1;
            if (opt == Opt.adam) { bc1 = 1f - 0.9f^^t1; bc2 = 1f - 0.999f^^t1; }
            size_t wn = cast(size_t) gl.outSz * gl.inSz;
            if (!gpucl.adamStep(gl.w, gl.gradW, gl.mW, gl.vW, lr, bc1, bc2, cast(int) opt, cast(int) wn))
                return false;
            if (!gpucl.adamStep(gl.b, gl.gradB, gl.mB, gl.vB, lr, bc1, bc2, cast(int) opt, gl.outSz))
                return false;

            // 아무것도 안 내린다 — 최신본은 GPU 에 있고, 호스트가 만지려 할 때
            // Network 의 훅이 _gpuSyncBack() 을 불러 그때 회수한다.
        }
        foreach (ref gl; layers) gl.host.t++;   // 여기까지 왔을 때만 올린다
        _gpuResident = true;   // 이제 w, b, m, v 의 최신본은 GPU 에 있다
        gpucl.runCount++;   // 여기까지 왔을 때만 센다 (중간 return false 는 CPU 폴백)
        return true;
    }

    // 여러 문제를 한 번에. 기울기를 모았다가 갱신은 한 번만 한다.
    // (문제마다 갱신하면 가중치 전체를 훑는 비용이 순전파보다 커진다)
    // 묶음 크기는 부르는 쪽이 정한다.
    void slMany(float[][] inputs, string[][] legal, int[][] ansIdx,
                float[][] ansVal, bool[][] use) {
        if (inputs.length == 0) return;
        net.zeroGrad();
        immutable float inv = 1.0f / cast(float) inputs.length;

        // ── 묶음 경로: 가중치 행을 배치 전체에 재사용 — Linear/Attn/Each 전부 지원 ──
        // MYML_NOBATCH 환경변수로 끌 수 있다 (per-sample 와 동치 검증용).
        if (!_noBatch) {
            int B = cast(int) inputs.length;
            if (_gpuSlMany(inputs, ansVal, use, B)) return;
            net._allocBatch(B);
            net.forwardBatch(inputs, B);
            foreach (h; 0..nHeads) {
                int oS = outSizes[h];
                net._bDHead[h][0 .. B*oS] = 0f;
                foreach (i; 0..B) {
                    if (!use[i][h]) continue;
                    auto slice = net._bHd[h][i*oS .. i*oS + oS];
                    auto dst   = net._bDHead[h][i*oS .. i*oS + oS];
                    if (cosModes[h]) {
                        dst[0] = (slice[0] - ansVal[i][h]) * inv;
                    } else {
                        auto lg = (h < legal.length) ? legal[h] : null;
                        int mlen = maskOf(h, lg);
                        foreach (k; 0..mlen) _probBufs[h][k] = slice[_maskBufs[h][k]];
                        softmaxInPlace(_probBufs[h][0..mlen]);
                        int tgt = -1;
                        foreach (k; 0..mlen) if (_maskBufs[h][k] == ansIdx[i][h]) { tgt = k; break; }
                        foreach (k; 0..mlen)
                            dst[_maskBufs[h][k]] = (_probBufs[h][k] - (k == tgt ? 1f : 0f)) * inv;
                    }
                }
            }
            net.backwardBatch(B);
            net.step(opt, lr, decay);
            // 재생이 끝난 메모가 다음 재생의 출발점이 된다
            if (net.hasMemo()) net.snapshotMemo();
            return;
        }

        foreach (i; 0..inputs.length) {
            bool any = false;
            foreach (u; use[i]) if (u) { any = true; break; }
            if (!any) continue;

            foreach (k; 0..inputs[i].length) _envBuf[k] = inputs[i][k];
            net.forward(_envBuf);

            foreach (h; 0..nHeads) {
                net._dHead[h][] = 0f;
                if (!use[i][h]) continue;
                if (cosModes[h]) {
                    net._dHead[h][0] = (net._hd[h][0] - ansVal[i][h]) * inv;
                } else {
                    auto lg = (h < legal.length) ? legal[h] : null;
                    int mlen = maskOf(h, lg);
                    foreach (k; 0..mlen) _probBufs[h][k] = net._hd[h][_maskBufs[h][k]];
                    softmaxInPlace(_probBufs[h][0..mlen]);
                    int tgt = -1;
                    foreach (k; 0..mlen) if (_maskBufs[h][k] == ansIdx[i][h]) { tgt = k; break; }
                    foreach (k; 0..mlen)
                        net._dHead[h][_maskBufs[h][k]] =
                            (_probBufs[h][k] - (k == tgt ? 1f : 0f)) * inv;
                }
            }
            net.backward();
        }

        net.step(opt, lr, decay);
        if (net.hasMemo()) net.snapshotMemo();
    }

    // ── jepa 용 진입점 ────────────────────────────────────────────────
    // 보통 학습(sl/save)은 "입력 -> 정답" 한 번으로 끝나지만, jepa 는 같은 신경망을
    // 두 번(x 한 번, y 한 번) 돌린 뒤 두 결과를 비교해야 한다. 그래서 순전파와
    // 역전파를 따로 부를 수 있게 열어둔다. 학습 자체는 Jepa 가 조립한다.

    // 헤드 h 의 출력 전체를 벡터로 (샘플 하나). net.forward 경로.
    void embedOne(const(float)[] x, float[] outv, int h = 0) {
        foreach (k; 0..net.inputSz) _envBuf[k] = k < x.length ? x[k] : 0f;
        net.forward(_envBuf);
        int D = outSizes[h];
        outv[0..D] = net._hd[h][0..D];
    }

    // 배치 순전파. 결과는 net._bHd[0] 에 [B*outSizes[0]] 로 남는다.
    void embedForward(float[][] X, int B) {
        net._allocBatch(B);
        net.forwardBatch(X, B);
    }

    // dOut([B*outSizes[0]]) -> 가중치 기울기 누적. step 은 밖에서 한 번만.
    // dInput != null 이면 입력에 대한 기울기도 낸다 ([B*net.inputSz]).
    void embedBackward(const(float)[] dOut, int B, float[] dInput) {
        int D = outSizes[0];
        foreach (h; 0..nHeads) net._bDHead[h][0 .. B*outSizes[h]] = 0f;
        net._bDHead[0][0 .. B*D] = dOut[0 .. B*D];
        net.backwardBatch(B, dInput);
    }

    void save() {
        if (!ready) return;
        _gpuSyncBack();          // m, v 를 파일에 쓰므로 GPU 에 있는 최신본을 먼저 내린다
        auto f = File(file, "wb");
        void wu(uint v)  { f.rawWrite((&v)[0..1]); }
        void wf(float v) { f.rawWrite((&v)[0..1]); }
        // ver 10 = 로직(kind 3), ver 11 = 기억(kind 4), ver 12 = 합성곱(kind 5).
        // 새 kind 가 든 파일을 옛 바이너리가 읽으면 Each 로 오해해서 조용히 깨지므로
        // kind 를 늘릴 때마다 올린다. ver 12 부터는 층 스펙이 두 칸에서 세 칸으로
        // 늘었다 (conv 가 출력채널·창·항목수 셋을 쓴다). 9~12 는 전부 읽는다.
        wu(0xBEEFCAFE); wu(12); wu(cast(uint)opt);
        wu(cast(uint)net.inputSz);
        wu(cast(uint)layKind.length);
        foreach (i; 0..layKind.length) {
            wu(cast(uint)layKind[i]); wu(cast(uint)layA[i]); wu(cast(uint)layB[i]);
            wu(cast(uint)layC[i]);     // ver 12 부터 (conv 가 세 번째 값을 쓴다)
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
            else if (net.kinds[i] == 1) {
                auto a = &net.attns[net.slot[i]];
                wn(a.ln); wl(a.wq); wl(a.wk); wl(a.wv); wl(a.wo);
            } else if (net.kinds[i] == 3) {
                auto g = &net.logics[net.slot[i]];
                // 입력 연결(ia, ib)은 생성할 때 무작위로 뽑은 것이라 반드시 같이
                // 저장해야 한다. 안 그러면 불러온 뒤 완전히 다른 회로가 된다.
                foreach (v; g.ia) wu(cast(uint)v);
                foreach (v; g.ib) wu(cast(uint)v);
                foreach (v; g.gw) wf(v);
                foreach (v; g.mg) wf(v);
                foreach (v; g.vg) wf(v);
                wu(cast(uint)g.t);
            } else if (net.kinds[i] == 4) {
                // 메모(h)는 저장하지 않는다 — 배운 값이 아니라 "지금 어디까지 왔나"
                // 라는 현재 상태다. 불러오면 빈 메모로 시작한다.
                auto r = &net.rnns[net.slot[i]];
                wl(r.gx); wl(r.gh); wl(r.zx); wl(r.zh);
            } else if (net.kinds[i] == 5) {
                wl(net.convs[net.slot[i]].lin);
            } else {
                wl(net.eachs[net.slot[i]].lin);
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
        if (ver < 9) throw new Exception("예전 포맷입니다. change() 로 변환하세요");
        if (ver > 12) throw new Exception("더 새 버전에서 저장한 파일입니다 (ver "
                                        ~ to!string(ver) ~ ")");
        opt = cast(Opt)ru();
        int inputSz = ru();
        int nLay = ru();
        layKind = new ubyte[nLay]; layA = new int[nLay]; layB = new int[nLay];
        layC = new int[nLay];
        hiddenSizes = [];
        foreach (i; 0..nLay) {
            layKind[i] = cast(ubyte)ru(); layA[i] = ru(); layB[i] = ru();
            // ver 12 에서 세 번째 값이 생겼다. 그 전 파일엔 없으므로 읽지 않는다
            // (읽으면 바이트 흐름이 어긋나 뒤가 전부 깨진다).
            layC[i] = (ver >= 12) ? cast(int)ru() : 0;
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
        net = new Network(inputSz, layKind, layA, layB, layC, outSizes);
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
            else if (net.kinds[i] == 1) {
                auto a = &net.attns[net.slot[i]];
                rn_(a.ln); rl_(a.wq); rl_(a.wk); rl_(a.wv); rl_(a.wo);
            } else if (net.kinds[i] == 3) {
                auto g = &net.logics[net.slot[i]];
                foreach (ref v; g.ia) v = cast(int)ru();
                foreach (ref v; g.ib) v = cast(int)ru();
                foreach (ref v; g.gw) v = rf();
                foreach (ref v; g.mg) v = rf();
                foreach (ref v; g.vg) v = rf();
                g.t = ru();
            } else if (net.kinds[i] == 4) {
                auto r = &net.rnns[net.slot[i]];
                rl_(r.gx); rl_(r.gh); rl_(r.zx); rl_(r.zh);
            } else if (net.kinds[i] == 5) {
                rl_(net.convs[net.slot[i]].lin);
            } else {
                rl_(net.eachs[net.slot[i]].lin);
            }
        }
        foreach (ref h; net.heads) rl_(h);
    }
}

// ─────────────────────────────────────────────
// Jepa — 요약기(encoder) + 예측기(predictor)
// ─────────────────────────────────────────────
// 원본을 그대로 맞추는 대신 "요약"만 맞춘다 (LeCun 의 JEPA).
//   sx = 요약기(x),  sy = 요약기(y),  p = 예측기(sx (+행동))
//   손실 = 평균제곱오차(p, sy) + collapse 벌점(sx) + collapse 벌점(sy)
// x 는 지금, y 는 다음 — 이렇게 쓰면 월드모델. x/y 를 같은 것의 두 조각으로 주면
// 그냥 자기지도학습. 라이브러리 입장에선 둘이 똑같고, 뭘 넣을지는 쓰는 쪽 마음이다.
//
// 요약기는 x 와 y 에 같은 가중치를 쓴다 (한 덩어리를 두 번 돌린다). 그래서 기울기도
// 양쪽에서 와서 합쳐진다.
class Jepa {
    BlackBoxAI enc, pred;
    VicReg vic;
    int D;        // 요약 크기 (= 요약기 헤드 0 의 출력 개수)
    int A;        // 행동 입력 개수 (= 예측기 입력폭 - D).  0 이면 행동 없는 형태

    private int       _cap;
    private float[]   _sx, _sy, _dP, _dSx, _dSy, _dPin;
    private float[][] _pin;      // 예측기 입력 [B][D+A]
    private float[]   _pinFlat;  // 위의 실제 저장소
    private float[]   _one;      // 샘플 하나짜리 스크래치 [D+A]

    this(BlackBoxAI e, BlackBoxAI p) {
        enc = e; pred = p;
        if (e.nHeads < 1 || p.nHeads < 1)
            throw new Exception("jepa: 요약기와 예측기 둘 다 출력이 있어야 합니다");
        if (!e.cosModes[0] || !p.cosModes[0])
            throw new Exception("jepa: 요약기와 예측기의 출력은 숫자(vec)여야 합니다");
        D = e.outSizes[0];
        if (p.outSizes[0] != D)
            throw new Exception("jepa: 예측기 출력 " ~ to!string(p.outSizes[0])
                              ~ " 가 요약 크기 " ~ to!string(D) ~ " 와 다릅니다");
        A = p.net.inputSz - D;
        if (A < 0)
            throw new Exception("jepa: 예측기 입력 " ~ to!string(p.net.inputSz)
                              ~ " 이 요약 크기 " ~ to!string(D) ~ " 보다 작습니다");
        _one = new float[D + A]; _one[] = 0f;
    }

    private void _alloc(int B) {
        if (B <= _cap) return;
        _sx  = new float[B*D];        _sx[]  = 0f;
        _sy  = new float[B*D];        _sy[]  = 0f;
        _dP  = new float[B*D];        _dP[]  = 0f;
        _dSx = new float[B*D];        _dSx[] = 0f;
        _dSy = new float[B*D];        _dSy[] = 0f;
        _dPin    = new float[B*(D+A)]; _dPin[]    = 0f;
        _pinFlat = new float[B*(D+A)]; _pinFlat[] = 0f;
        _pin = new float[][B];
        foreach (b; 0..B) _pin[b] = _pinFlat[b*(D+A) .. (b+1)*(D+A)];
        _cap = B;
    }

    // xs/ys: 각 [B][입력수].  acts: 행동이 있으면 [B][A], 없으면 null.
    // 돌려주는 값은 손실 (예측 오차 + collapse 벌점).
    float train(float[][] xs, float[][] ys, float[][] acts) {
        int B = cast(int) xs.length;
        if (B == 0) return 0f;
        if (ys.length != xs.length)
            throw new Exception("jepa: x 와 y 의 개수가 다릅니다");
        if (A > 0 && (acts is null || acts.length != xs.length))
            throw new Exception("jepa: 행동을 " ~ to!string(A) ~ "개씩 x 개수만큼 주세요");
        _alloc(B);
        enc.net.zeroGrad();
        pred.net.zeroGrad();

        // 1) 같은 요약기에 x, y 를 각각 통과시킨다.
        //    두 번째 호출이 첫 번째의 중간값 버퍼를 덮어쓰므로 결과를 따로 빼둔다.
        enc.embedForward(xs, B);
        _sx[0..B*D] = enc.net._bHd[0][0..B*D];
        enc.embedForward(ys, B);      // 이제 요약기 버퍼는 y 패스를 담고 있다
        _sy[0..B*D] = enc.net._bHd[0][0..B*D];

        // 2) 예측기: sx (+행동) -> sy 예측
        foreach (b; 0..B) {
            _pin[b][0..D] = _sx[b*D .. b*D + D];
            if (A > 0) foreach (k; 0..A) _pin[b][D+k] = acts[b][k];
        }
        pred.embedForward(_pin, B);

        // 3) 예측 손실 — 원본이 아니라 "요약" 공간에서 잰다. 이게 JEPA 의 핵심.
        float loss = 0f;
        immutable float g = 2f / cast(float)(B * D);
        foreach (i; 0..B*D) {
            float d = pred.net._bHd[0][i] - _sy[i];
            loss += d * d;
            _dP[i] = d * g;
        }
        loss /= cast(float)(B * D);

        // 4) 예측기 역전파. 입력 기울기를 받아서 앞의 요약기(x 쪽)로 넘긴다.
        pred.embedBackward(_dP[0..B*D], B, _dPin[0..B*(D+A)]);
        foreach (b; 0..B) _dSx[b*D .. b*D + D] = _dPin[b*(D+A) .. b*(D+A) + D];

        // 5) y 쪽 요약기 기울기 — 예측의 목표라서 부호가 반대다.
        foreach (i; 0..B*D) _dSy[i] = -_dP[i];

        // 6) collapse 벌점을 양쪽 요약에 더한다. 이것만 배치 전체를 봐야 계산된다.
        loss += vic.grad(_sy[0..B*D], _dSy[0..B*D], B, D);
        loss += vic.grad(_sx[0..B*D], _dSx[0..B*D], B, D);

        // 7) 요약기 역전파 2번 — 지금 버퍼가 y 패스라 y 를 먼저 하고,
        //    x 는 중간값을 다시 만들어야 하므로 순전파를 한 번 더 돌린다.
        enc.embedBackward(_dSy[0..B*D], B, null);
        enc.embedForward(xs, B);
        enc.embedBackward(_dSx[0..B*D], B, null);

        // 8) 모아둔 기울기로 한 번씩만 갱신
        // step() 이 옵티마이저 상태를 읽는다 — GPU 에 최신본이 있으면 먼저 내린다
        // (요약기가 전에 sl() 로 GPU 경로를 탔을 수 있다)
        enc._gpuSyncBack();  enc.net.step(enc.opt, enc.lr, enc.decay);
        pred._gpuSyncBack(); pred.net.step(pred.opt, pred.lr, pred.decay);
        return loss;
    }

    // x -> 요약
    void encode(const(float)[] x, float[] outv) { enc.embedOne(x, outv); }

    // x (+행동) -> 다음 요약 예측. "미리 상상해보기".
    void imagine(const(float)[] x, const(float)[] act, float[] outv) {
        enc.embedOne(x, _one[0..D]);
        if (A > 0) foreach (k; 0..A) _one[D+k] = k < act.length ? act[k] : 0f;
        pred.embedOne(_one, outv);
    }

    void save() { enc.save(); pred.save(); }
}

void resset(string modelName) {
    bool deleted = false;
    foreach (suffix; ["_ml_memory.pth", "_sl_memory.pth", "_auto_memory.pth"]) {
        string path = modelName ~ suffix;
        if (exists(path)) {
            try { remove(path); 알림(" [%s] 초기화: %s", modelName, path); deleted = true; }
            catch (Exception e) { 알림("오류: %s 삭제 실패 (%s)", path, e.msg); }
        }
    }
    if (!deleted) 알림(" [%s] 모델 파일이 존재하지 않습니다.", modelName);
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
    int[]      kK, kA, kB;      // 층 스펙 (종류, A, B)
    int[]      outSizes;
    bool[]     cosModes;
    string[][] als;
    ubyte[]    blob;

    {
        auto f = File(path, "rb");
        uint ru() { uint v; f.rawRead((&v)[0..1]); return v; }
        if (ru() != 0xBEEFCAFE) throw new Exception("my_ml 포맷이 아닙니다");
        uint ver = ru();
        if (ver >= 9) return "already";
        if (ver >= 2) o = cast(Opt) ru();

        bool cos3 = false; int out3 = 0;
        if (ver == 3) { cos3 = ru() != 0; out3 = ru(); }
        bool ver4 = (ver >= 4);
        bool ver5 = (ver == 5);
        bool ver7up = (ver >= 7);
        if (ver == 8) { ru(); ru(); ru(); ru(); }   // 옛 토큰 설정 — 버린다

        inputSz = ru();
        nH = ru(); hid = new int[nH];
        kA = new int[nH]; kB = new int[nH]; kK = new int[nH];
        foreach (i; 0..nH) {
            if (ver7up) { kK[i] = ru(); kA[i] = ru(); kB[i] = ru(); hid[i] = kA[i]; }
            else        { kK[i] = 0;    kA[i] = ru(); kB[i] = 0;    hid[i] = kA[i]; }
        }
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
    wu(0xBEEFCAFE); wu(11); wu(cast(uint) o);
    wu(cast(uint) inputSz);
    wu(cast(uint) nH);
    foreach (i; 0..nH) { wu(cast(uint) kK[i]); wu(cast(uint) kA[i]); wu(cast(uint) kB[i]); }
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
    auto n = PyList_Size(lst);
    if (n <= 0) return null;
    auto r = new string[n];
    foreach (i; 0..n) r[i] = fromStringz(PyUnicode_AsUTF8(PyList_GetItem(lst, i))).idup;
    return r;
}
string[][] pyLals(PyObject* lals) {
    auto n = PyList_Size(lals);
    if (n <= 0) return null;
    auto r = new string[][n];
    foreach (i; 0..n) r[i] = pyStrList(PyList_GetItem(lals, i));
    return r;
}
// 길이를 먼저 물어보고 한 번에 잡는다. 예전엔 `~=` 로 늘려가서 입력 길이에
// 로그 비례로 재할당이 났다 — 배치 1 호출마다 지나가는 자리라 그게 다 비용이었다.
float[] pyFloatList(PyObject* lst) {
    auto n = PyList_Size(lst);
    if (n <= 0) return null;
    auto r = new float[n];
    foreach (i; 0..n) r[i] = cast(float) PyFloat_AsDouble(PyList_GetItem(lst, i));
    return r;
}
int[] pyIntList(PyObject* lst) {
    auto n = PyList_Size(lst);
    if (n <= 0) return null;
    auto r = new int[n];
    foreach (i; 0..n) r[i] = cast(int) PyLong_AsLong(PyList_GetItem(lst, i));
    return r;
}
PyObject* toPyList(string[] strs) {
    auto lst = PyList_New(strs.length);
    foreach (i, s; strs) PyList_SetItem(lst, i, PyUnicode_FromString(toStringz(s)));
    return lst;
}

extern(C) void bbai_dtor(PyObject* cap) nothrow @trusted {
    auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
    if (ai) try {
        ai.releaseGpu();               // 루트를 떼기 전에 — 아직 살아있을 때 놓아준다
        GC.removeRoot(cast(void*) ai);
    } catch (Throwable) {}
}

// 예외를 파이썬 쪽으로 넘길 때 실제 내용을 담는다.
//
// 예전엔 전부 "my_ml: exception in _ml_xxx" 한 줄로만 보고해서, 무슨 일이
// 일어났는지 알 방법이 아예 없었다 (재현 안 되는 실패를 만났을 때 정확히 이것
// 때문에 아무것도 알아내지 못했다). 클래스 이름은 항상 붙인다 —
// OutOfMemoryError 처럼 msg 가 비어있는 Error 도 이름만으로 구분이 된다.
private void setPyError(string where, Throwable t) nothrow {
    try {
        string m = where ~ ": " ~ typeid(t).name;
        if (t.msg.length) m ~= " — " ~ t.msg;
        PyErr_SetString(_pyRuntimeError, toStringz(m));
    } catch (Throwable) {
        // 메모리 부족을 처리하는 중이면 위 할당도 실패할 수 있다. 리터럴은
        // 널 종료라 할당 없이 그대로 넘길 수 있다.
        PyErr_SetString(_pyRuntimeError, "my_ml: 예외 (내용을 담지 못했습니다)");
    }
}

extern(C) void jepa_dtor(PyObject* cap) nothrow @trusted {
    auto j = cast(Jepa) PyCapsule_GetPointer(cap, "Jepa");
    if (j) try { GC.removeRoot(cast(void*) j); } catch (Throwable) {}
}

// [[a,b],[c,d]] -> float[][]
float[][] pyFloatMat(PyObject* lst) {
    Py_ssize_t n = PyList_Size(lst);
    auto r = new float[][n];
    foreach (i; 0..n) r[i] = pyFloatList(PyList_GetItem(lst, i));
    return r;
}

PyObject* toPyFloats(const(float)[] v) {
    auto lst = PyList_New(v.length);
    foreach (i, x; v) PyList_SetItem(lst, i, PyFloat_FromDouble(x));
    return lst;
}

// ─────────────────────────────────────────────
// Python extension functions
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
extern(C) nothrow @trusted:

PyObject* py_ml_make(PyObject* self, PyObject* args) {
    try {
        PyObject* nm; int inputSz; PyObject* hid; PyObject* heads;
        PyObject* acts; PyObject* coss; PyObject* opt;
        PyObject* lk; PyObject* lb2; PyObject* lc2;
        double sigma, ent, lr, decay, temp;
        if (!PyArg_ParseTuple(args, "OiOOOOOOOOddddd", &nm, &inputSz, &lk, &hid, &lb2,
                              &lc2, &heads, &acts, &coss, &opt, &sigma, &ent,
                              &lr, &decay, &temp))
            return null;
        string name   = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        string optStr = fromStringz(PyUnicode_AsUTF8(opt)).idup;
        int[] kindsI  = pyIntList(lk);
        int[] la      = pyIntList(hid);
        int[] lbv     = pyIntList(lb2);
        int[] lcv     = pyIntList(lc2);
        auto  lkb     = new ubyte[kindsI.length];
        foreach (i, v; kindsI) lkb[i] = cast(ubyte) v;
        int[] hsz     = pyIntList(heads);
        string[][] al = pyLals(acts);
        int[] cosI    = pyIntList(coss);
        auto cosB = new bool[cosI.length];
        foreach (i, v; cosI) cosB[i] = v != 0;
        auto ai = new BlackBoxAI(name, inputSz, lkb, la, lbv, lcv, hsz, al, cosB, parseOpt(optStr),
                                 cast(float)sigma, cast(float)ent,
                                 cast(float)lr, cast(float)decay, cast(float)temp);
        GC.addRoot(cast(void*) ai);
        return PyCapsule_New(cast(void*) ai, "BlackBoxAI", &bbai_dtor);
    } catch (Throwable t) { setPyError("my_ml: _ml_make", t); return null; }
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
    } catch (Throwable t) { setPyError("my_ml: _ml_pick", t); return null; }
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
    } catch (Throwable t) { setPyError("my_ml: _ml_predict", t); return null; }
}

// 헤드 h 의 출력 전체를 벡터로 (vec 출력용). predict 는 한 칸짜리라 이걸 쓴다.
PyObject* py_ml_embed(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inp; int h;
        if (!PyArg_ParseTuple(args, "OOi", &cap, &inp, &h)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        if (h < 0 || h >= ai.nHeads) {
            PyErr_SetString(_pyRuntimeError, "my_ml: embed() 의 출력 번호가 범위를 벗어났습니다");
            return null;
        }
        auto v = new float[ai.outSizes[h]];
        ai.embedOne(pyFloatList(inp), v, h);
        return toPyFloats(v);
    } catch (Throwable t) { setPyError("my_ml: _ml_embed", t); return null; }
}

// GPU 를 실제로 쓰고 있는지 확인용. "켜 뒀는데 사실 CPU 로 돌고 있었다" 를
// 알아챌 방법이 달리 없다.
PyObject* py_ml_gpu_info(PyObject* self, PyObject* args) {
    try {
        auto d = PyDict_New();
        void put(string k, PyObject* v) { PyDict_SetItemString(d, toStringz(k), v); Py_DecRef(v); }
        // MYML_GPU=0 이면 프로브 자체를 하지 않는다 (완전 비활성 약속을 지킨다)
        bool avail = (_gpuMode != "0") && gpucl.available();
        put("available", PyLong_FromLong(avail ? 1 : 0));
        put("device", PyUnicode_FromString(toStringz(avail ? gpucl.deviceName() : "")));
        put("mode", PyUnicode_FromString(toStringz(_gpuMode)));
        put("runs", PyLong_FromLong(cast(long) gpucl.runCount));
        put("min_flops", PyLong_FromLong(_gpuMinFlops));
        put("min_batch", PyLong_FromLong(_gpuMinB));
        return d;
    } catch (Throwable t) { setPyError("my_ml: _ml_gpu_info", t); return null; }
}

// 어떤 SIMD 수준으로 돌고 있는지. avx512 경로는 실하드웨어에서 검증한 적이
// 없으므로 (개발 기계가 Zen 3+ 라 지원을 안 한다) 쓰는 쪽이 확인할 수단을 둔다.
PyObject* py_ml_cpu_info(PyObject* self, PyObject* args) {
    try {
        import cpu = core.cpuid;
        auto d = PyDict_New();
        void put(string k, PyObject* v) { PyDict_SetItemString(d, toStringz(k), v); Py_DecRef(v); }
        put("simd",    PyUnicode_FromString(toStringz(_simd)));
        put("threads", PyLong_FromLong(_nThreads));
        put("avx2",    PyLong_FromLong(cpu.avx2 ? 1 : 0));
        put("avx512",  PyLong_FromLong((cpu.avx2 && cpu.avx512f) ? 1 : 0));
        put("cores",   PyLong_FromLong(cast(long) totalCPUs));
        return d;
    } catch (Throwable t) { setPyError("my_ml: _ml_cpu_info", t); return null; }
}

// 기억(rnn) 층의 메모를 비운다. 에피소드 경계에서 부른다.
PyObject* py_ml_forget(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        if (ai && ai.ready) ai.net.forget();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _ml_forget", t); return null; }
}

// 학습 중에 바꿀 수 있어야 하는 값들 (학습률 스케줄, 탐험 식히기 등).
// 하나에 몰아넣는다 — 값마다 함수를 만들면 추가할 때마다 메서드 표가 늘어난다.
PyObject* py_ml_tune(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* nameObj; PyObject* valObj;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &nameObj, &valObj)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        string k = fromStringz(PyUnicode_AsUTF8(nameObj)).idup;

        // valObj 가 None 이면 읽기, 아니면 쓰기
        bool 쓰기 = (valObj !is _pyNone);
        float v = 쓰기 ? cast(float) PyFloat_AsDouble(valObj) : 0f;

        float cur;
        switch (k) {
            case "lr":      if (쓰기) ai.lr = v;       cur = ai.lr;       break;
            case "decay":   if (쓰기) ai.decay = v;    cur = ai.decay;    break;
            case "temp":    if (쓰기) ai.temp = v;     cur = ai.temp;     break;
            case "sigma":   if (쓰기) ai.cosSigma = v; cur = ai.cosSigma; break;
            case "entropy": if (쓰기) ai.entropy = v;  cur = ai.entropy;  break;
            default:
                PyErr_SetString(_pyRuntimeError,
                    "my_ml: 모르는 설정값입니다 (lr, decay, temp, sigma, entropy)");
                return null;
        }
        return PyFloat_FromDouble(cur);
    } catch (Throwable t) { setPyError("my_ml: _ml_tune", t); return null; }
}

// 로직 층이 배운 회로를 글로. 층별 리스트의 리스트를 돌려준다.
PyObject* py_ml_logic_rules(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        if (!ai || !ai.ready) return PyList_New(0);
        ai.syncFromGpu();   // 신경망을 안 거치고 직접 훑는다
        auto outer = PyList_New(ai.net.logics.length);
        foreach (i; 0..ai.net.logics.length)
            PyList_SetItem(outer, i, toPyList(ai.net.logics[i].rules()));
        return outer;
    } catch (Throwable t) { setPyError("my_ml: _ml_logic_rules", t); return null; }
}

PyObject* py_jepa_make(PyObject* self, PyObject* args) {
    try {
        PyObject* ec; PyObject* pc; double vw, cw, gm;
        if (!PyArg_ParseTuple(args, "OOddd", &ec, &pc, &vw, &cw, &gm)) return null;
        auto e = cast(BlackBoxAI) PyCapsule_GetPointer(ec, "BlackBoxAI");
        auto p = cast(BlackBoxAI) PyCapsule_GetPointer(pc, "BlackBoxAI");
        auto j = new Jepa(e, p);
        j.vic.varW = cast(float)vw; j.vic.covW = cast(float)cw; j.vic.gamma = cast(float)gm;
        GC.addRoot(cast(void*) j);
        return PyCapsule_New(cast(void*) j, "Jepa", &jepa_dtor);
    } catch (Exception e) {
        PyErr_SetString(_pyRuntimeError, toStringz("my_ml: " ~ e.msg)); return null;
    } catch (Throwable t) { setPyError("my_ml: _jepa_make", t); return null; }
}

PyObject* py_jepa_train(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* xs; PyObject* ys; PyObject* acts;
        if (!PyArg_ParseTuple(args, "OOOO", &cap, &xs, &ys, &acts)) return null;
        auto j = cast(Jepa) PyCapsule_GetPointer(cap, "Jepa");
        float[][] am = (acts is _pyNone) ? null : pyFloatMat(acts);
        float loss = j.train(pyFloatMat(xs), pyFloatMat(ys), am);
        return PyFloat_FromDouble(loss);
    } catch (Exception e) {
        PyErr_SetString(_pyRuntimeError, toStringz("my_ml: " ~ e.msg)); return null;
    } catch (Throwable t) { setPyError("my_ml: _jepa_train", t); return null; }
}

PyObject* py_jepa_encode(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inp;
        if (!PyArg_ParseTuple(args, "OO", &cap, &inp)) return null;
        auto j = cast(Jepa) PyCapsule_GetPointer(cap, "Jepa");
        auto v = new float[j.D];
        j.encode(pyFloatList(inp), v);
        return toPyFloats(v);
    } catch (Throwable t) { setPyError("my_ml: _jepa_encode", t); return null; }
}

PyObject* py_jepa_imagine(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* inp; PyObject* act;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &inp, &act)) return null;
        auto j = cast(Jepa) PyCapsule_GetPointer(cap, "Jepa");
        auto v = new float[j.D];
        float[] a = (act is _pyNone) ? null : pyFloatList(act);
        j.imagine(pyFloatList(inp), a, v);
        return toPyFloats(v);
    } catch (Throwable t) { setPyError("my_ml: _jepa_imagine", t); return null; }
}

PyObject* py_jepa_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(Jepa) PyCapsule_GetPointer(cap, "Jepa")).save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _jepa_save", t); return null; }
}

PyObject* py_jepa_meta(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto j = cast(Jepa) PyCapsule_GetPointer(cap, "Jepa");
        auto lst = PyList_New(2);
        PyList_SetItem(lst, 0, PyLong_FromLong(j.D));
        PyList_SetItem(lst, 1, PyLong_FromLong(j.A));
        return lst;
    } catch (Throwable t) { setPyError("my_ml: _jepa_meta", t); return null; }
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
        // 파일로 쓸지는 파이썬 쪽이 정한다 (autosave). 온라인 RL 은 매 스텝
        // 여기를 지나가는데, 그때마다 가중치 파일을 쓰면 학습 시간의 3분의 2가
        // 디스크에 나간다 (측정: [64,256,256] 에서 14.4ms 중 9.7ms).
        ai.learnBatch(inputs, chosen, values, score);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _ml_learn", t); return null; }
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
        auto lals  = pyLals(legal);
        auto chosen = new int[ai.nHeads];
        auto value  = new float[ai.nHeads];
        auto input = pyFloatList(inp);
        ai.slBatch(input, lals, pyIntList(ansI), pyFloatList(ansV), use);
        ai.predictAll(lals, input, chosen, value);
        return packPick(ai, chosen, value);
    } catch (Throwable t) { setPyError("my_ml: _ml_sl", t); return null; }
}

// 여러 문제를 한 번에 넘긴다 (경계 넘나드는 비용을 줄이려는 것)
PyObject* py_ml_sl_many(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* legal; PyObject* inps;
        PyObject* ansI; PyObject* ansV; PyObject* useL;
        if (!PyArg_ParseTuple(args, "OOOOOO", &cap, &legal, &inps, &ansI, &ansV, &useL))
            return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        Py_ssize_t n = PyList_Size(inps);
        auto inputs = new float[][n];
        auto ai_    = new int[][n];
        auto av_    = new float[][n];
        auto us_    = new bool[][n];
        foreach (i; 0..n) {
            inputs[i] = pyFloatList(PyList_GetItem(inps, i));
            ai_[i]    = pyIntList(PyList_GetItem(ansI, i));
            av_[i]    = pyFloatList(PyList_GetItem(ansV, i));
            auto ui   = pyIntList(PyList_GetItem(useL, i));
            us_[i]    = new bool[ui.length];
            foreach (j, v; ui) us_[i][j] = v != 0;
        }
        ai.slMany(inputs, pyLals(legal), ai_, av_, us_);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _ml_sl_many", t); return null; }
}

PyObject* py_ml_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _ml_save", t); return null; }
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
    } catch (Throwable t) { setPyError("my_ml: _ml_change", t); return null; }
}

PyObject* py_ml_resset(PyObject* self, PyObject* args) {
    try {
        PyObject* nm;
        if (!PyArg_ParseTuple(args, "O", &nm)) return null;
        resset(fromStringz(PyUnicode_AsUTF8(nm)).idup);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable t) { setPyError("my_ml: _ml_resset", t); return null; }
}

PyObject* py_ml_gc_disable(PyObject* self, PyObject* args) {
    try { GC.disable(); Py_IncRef(_pyNone); return _pyNone; }
    catch (Throwable t) { setPyError("my_ml: _ml_gc_disable", t); return null; }
}

PyObject* py_ml_gc_collect(PyObject* self, PyObject* args) {
    try { GC.enable(); GC.collect(); Py_IncRef(_pyNone); return _pyNone; }
    catch (Throwable t) { setPyError("my_ml: _ml_gc_collect", t); return null; }
}

PyObject* py_ml_export_weights(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto d  = PyDict_New();
        if (!ai || !ai.ready) return d;
        ai.syncFromGpu();   // 가중치를 직접 훑는다 — Network 훅을 안 거치므로 직접 부른다
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
    } catch (Throwable t) { setPyError("my_ml: _ml_export_weights", t); return null; }
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
    } catch (Throwable t) { setPyError("my_ml: _ml_get_meta", t); return null; }
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


class _Each:
    """항목마다 따로 도는 층. 같은 가중치를 항목 수만큼 돌려쓴다.

        each(64)       -> 항목 하나를 64칸으로 (항목 수는 앞 층에서 물려받는다)
        each(64, 8)    -> 항목 8개로 직접 지정 (맨 앞에 쓸 때)

    일반 층은 전체를 한 덩어리로 섞어서 항목 구분이 사라진다.
    attn/conv 사이에 이걸 끼우면 항목이 끝까지 유지된다.
    """
    __slots__ = ("width", "items")

    def __init__(self, width=0, items=0):
        self.width, self.items = int(width), int(items)

    def __call__(self, width, items=0):
        if width < 1: raise ValueError("each(폭) 은 1 이상이어야 합니다")
        return _Each(width, items)

    def __repr__(self):
        return f"each({self.width}" + (f", {self.items})" if self.items else ")")

each = _Each()


class _Logic:
    """논리 게이트를 배우는 층 (뉴로심볼릭).

        logic(32)   -> 게이트 32개

    유닛 하나하나가 논리 게이트 한 개다. 들어온 값 중 두 칸을 보고 그 둘로 뭘
    할지(AND, OR, XOR, NAND, "그대로", "항상 참" ... 16가지)를 학습한다.
    학습이 끝나면 rules() 로 배운 식을 글로 읽을 수 있다 — 일반 층은 못 하는 것.

    들어온 값은 0~1 로 눌러서 참/거짓처럼 쓴다 (로직 층을 연달아 쌓을 때는
    이미 0~1 이므로 다시 누르지 않는다). 나오는 값도 0~1 이다.
    """
    __slots__ = ("gates",)

    def __init__(self, gates=0):
        self.gates = int(gates)

    def __call__(self, gates):
        if gates < 1: raise ValueError("logic(개수) 는 1 이상이어야 합니다")
        return _Logic(gates)

    def __repr__(self):
        return f"logic({self.gates})"

logic = _Logic()


class _Memory:
    """기억을 들고 다니는 층 (순환신경망).

        memory(32)   -> 숫자 32개짜리 메모장

    신경망 안에 메모장을 둔다. 부를 때마다 "이번 입력과 지금 메모를 보고 메모를
    어떻게 고칠지" 를 계산한다. 그래서 입력을 늘리지 않고도 과거를 볼 수 있고,
    몇 스텝 전까지 볼지 미리 정할 필요가 없다.

    에피소드가 바뀌면 ai.forget() 으로 메모를 비운다.

    알아둘 것 두 가지:
      - 기울기가 1스텝만 거슬러 간다. 메모에 든 걸 **읽어 쓰는 법**은 배우지만,
        "지금 적어두면 20스텝 뒤에 득이 된다" 는 잘 못 배운다.
      - 묶음(배치) 학습을 쓸 수 없다. 메모는 순서대로 이어져야 하는데 배치는
        샘플을 독립으로 처리해서 뜻이 안 맞는다. 자동으로 하나씩 처리한다.
    """
    __slots__ = ("cells",)

    def __init__(self, cells=0):
        self.cells = int(cells)

    def __call__(self, cells):
        if cells < 1: raise ValueError("memory(개수) 는 1 이상이어야 합니다")
        return _Memory(cells)

    def __repr__(self):
        return f"memory({self.cells})"

memory = _Memory()


class _Conv:
    """창을 미끄러뜨리며 같은 무늬를 찾는 층 (1차원 합성곱).

        conv(16, 3)        -> 출력채널 16, 창 3 (항목 수는 앞 층에서 물려받는다)
        conv(16, 3, 20)    -> 항목 20개로 직접 지정 (맨 앞 층일 때)

    입력을 "항목 n개 × 채널 c개" 로 보고, 창 하나를 항목 축으로 쭉 미끄러뜨린다.
    "왼쪽 끝의 무늬" 와 "오른쪽 끝의 무늬" 를 따로 배우지 않고 무늬 찾는 법 하나를
    배워서 전체에 적용한다. 그래서 가중치가 훨씬 적고 위치가 바뀌어도 통한다.

    길이는 유지한다 (출력도 항목 n개). 양끝은 0 으로 메운다.
    맨 앞에 쓸 때는 항목 수를 직접 줘야 한다 — 입력폭만 보고는 "항목 20 × 채널 3"
    인지 "항목 3 × 채널 20" 인지 알 수 없다.
    """
    __slots__ = ("ch", "win", "items")

    def __init__(self, ch=0, win=0, items=0):
        self.ch, self.win, self.items = int(ch), int(win), int(items)

    def __call__(self, ch, win, items=0):
        if ch < 1:  raise ValueError("conv(채널, ...) 의 채널은 1 이상이어야 합니다")
        if win < 1: raise ValueError("conv(..., 창) 의 창은 1 이상이어야 합니다")
        return _Conv(ch, win, items)

    def __repr__(self):
        return f"conv({self.ch}, {self.win}" + (f", {self.items})" if self.items else ")")

conv = _Conv()


class _Vec:
    """숫자 여러 개를 한 덩어리로 내는 출력.

        vec(64)   -> 숫자 64개짜리 출력 하나

    cos 를 64개 나열하는 것과 값은 같지만, 이건 "한 덩어리"라서 jepa 의
    요약(summary)처럼 전체를 통째로 다뤄야 할 때 쓴다. 값을 꺼낼 땐
    predict() 대신 embed() 를 쓴다 (predict 는 첫 칸만 준다).
    """
    __slots__ = ("size",)

    def __init__(self, size=0):
        self.size = int(size)

    def __call__(self, size):
        if size < 1: raise ValueError("vec(개수) 는 1 이상이어야 합니다")
        return _Vec(size)

    def __repr__(self):
        return f"vec({self.size})"

vec = _Vec()


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


class _판:
    """with ai.round(): 의 양쪽 끝에서 forget() 을 부른다.

    들여쓰기가 시작되는 곳이 판의 시작, 끝나는 곳이 판의 끝이다. 판의 경계가
    코드 모양 그 자체가 되어서, 부르는 걸 잊을 자리가 없어진다.

    나갈 때도 비우는 이유: 들어올 때만 비우면 블록을 나간 뒤에 메모가 남아서,
    판 밖에서 부르는 predict() 같은 것이 지난 판의 기억을 들고 돈다.
    예외로 빠져나가도 비운다 — 중간에 터진 판이 다음 판에 섞이면 더 나쁘다.
    """
    __slots__ = ("_ai",)
    def __init__(self, ai):            self._ai = ai
    def __enter__(self):               self._ai.forget(); return self._ai
    def __exit__(self, *_):            self._ai.forget(); return False


class BlackBoxAI:
    def __init__(self, h, name, heads, vecs=None, autosave=1, has_memory=False):
        self._h     = h
        self._name  = name
        self._heads = heads          # [(actions or None, cos:bool), ...]
        self._n     = len(heads)
        self._vecs  = list(vecs) if vecs else [False]*len(heads)
        self._autosave = int(autosave)
        self._since    = 0           # 마지막으로 파일에 쓴 뒤 학습한 횟수
        # memory 층을 썼는데 forget() 을 한 번도 안 부르면 판이 전부 이어져버린다.
        # 오류도 안 나고 결과만 조용히 나빠지는 종류라 한 번 알려준다.
        self._has_memory = bool(has_memory)
        self._forgot     = False
        self._warned     = False

    def _메모경고(self):
        if self._has_memory and not self._forgot and not self._warned:
            self._warned = True
            import warnings
            warnings.warn(
                f"[{self._name}] memory() 층이 있는데 forget() 을 한 번도 안 불렀습니다."
                " 판(에피소드)이 바뀔 때 ai.forget() 을 부르지 않으면 지난 판의 기억을"
                " 들고 새 판을 시작합니다 (오류는 안 나고 결과만 나빠집니다).",
                RuntimeWarning, stacklevel=3)

    @property
    def autosave(self):     return self._autosave

    @autosave.setter
    def autosave(self, n):  self._autosave = int(n)

    # ── 학습 중에 바꿀 수 있는 값들 ──────────────────────────────────
    # 학습률 스케줄("처음엔 크게, 나중엔 작게")이나 탐험 식히기에 쓴다.
    #     ai.lr = 0.001
    #     ai.temp = max(0.1, 1.0 - 진행도)
    @property
    def lr(self):           return _ml_tune(self._h, "lr", None)
    @lr.setter
    def lr(self, v):        _ml_tune(self._h, "lr", float(v))

    @property
    def decay(self):        return _ml_tune(self._h, "decay", None)
    @decay.setter
    def decay(self, v):     _ml_tune(self._h, "decay", float(v))

    @property
    def temp(self):         return _ml_tune(self._h, "temp", None)
    @temp.setter
    def temp(self, v):      _ml_tune(self._h, "temp", float(v))

    @property
    def sigma(self):        return _ml_tune(self._h, "sigma", None)
    @sigma.setter
    def sigma(self, v):     _ml_tune(self._h, "sigma", float(v))

    @property
    def entropy(self):      return _ml_tune(self._h, "entropy", None)
    @entropy.setter
    def entropy(self, v):   _ml_tune(self._h, "entropy", float(v))

    def _파일에쓸까(self):
        """save(scored) 가 학습한 뒤 파일까지 쓸지."""
        self._since += 1
        if self._autosave > 0 and self._since >= self._autosave:
            self._since = 0
            return True
        return False

    def forget(self):
        """memory() 층의 메모를 비운다. 판(에피소드)이 바뀔 때 부른다.

        안 부르면 지난 판의 기억을 들고 새 판을 시작한다.
        memory() 층이 없으면 아무 일도 하지 않는다.
        """
        self._forgot = True
        _ml_forget(self._h)

    def round(self):
        """판(에피소드) 하나를 감싼다. 양쪽 끝에서 memory() 의 메모를 비운다.

            for 판 in range(1000):
                with ai.round():        # <- 여기가 판의 시작
                    while 안끝났으면:
                        step = ai.rl(관측)
                        ...
                    ai.save(점수들)
                                        # <- 여기가 판의 끝 (여기서도 비운다)

        들여쓰기가 곧 판의 범위다. forget() 을 직접 불러도 되지만, 그쪽은
        "어디서 부를지" 를 매번 정해야 하고 이쪽은 정할 게 없다.
        memory() 층이 없으면 아무 일도 하지 않는다.
        """
        return _판(self)

    def rules(self):
        """로직 층이 배운 논리식을 글로 읽는다. 층이 여러 개면 리스트의 리스트.

            ai = make("M", [8, 16, logic(6)], [["예","아니오"]])
            ai.sl(...)
            for 층 in ai.rules():
                for i, 식 in enumerate(층):
                    print(i, 식)      # 예) 3 x5∧¬x2   (91%)

        괄호 안 숫자는 확신도다 (16가지 중 그 게이트에 몰린 정도).
        낮으면 아직 안 굳은 것이니 더 학습시키거나 그 유닛을 무시한다.
        """
        return _ml_logic_rules(self._h)

    def embed(self, input_list, head=0):
        """그 출력의 값 전체를 리스트로. vec 출력을 꺼낼 때 쓴다.
        출력이 여러 개면 head 로 몇 번째인지 고른다 (기본 0번)."""
        return _ml_embed(self._h, [float(x) for x in input_list], int(head))

    # 헤드별 원시값 → 사람이 쓰는 출력.
    # vec 출력 자리는 None 으로 비운다 — 숫자가 여러 개라 한 칸에 안 들어간다.
    # 그 값은 embed(입력, 번호) 로 따로 꺼낸다.
    def _decode(self, flat):
        out, units, raw = [], [], []
        for i, (acts, cos) in enumerate(self._heads):
            idx, val = int(flat[i*2]), float(flat[i*2+1])
            units.append(idx)
            raw.append(val)
            if self._vecs[i]:   out.append(None)
            else:               out.append(val if cos else acts[idx])
        return out, units, raw

    def _legal_arg(self, legal):
        """legal 은 출력 개수만큼의 리스트. 안 거는 자리는 None.
        출력이 하나면 액션 목록을 그냥 줘도 받는다 (legal=["왼쪽","정지"])."""
        if legal is None:
            return [[] for _ in range(self._n)]
        # 문자열만 든 리스트 = 액션 목록 하나
        if self._n == 1 and isinstance(legal, (list, tuple)) \
                and legal and all(isinstance(x, str) for x in legal):
            legal = [legal]
        if len(legal) != self._n:
            raise ValueError(f"legal 은 출력 개수({self._n})만큼 주세요. 안 걸 자리는 None")
        return [list(x) if x else [] for x in legal]

    # ── 순수: (입력, 출력) 데이터만 ──
    def rl(self, input_list, legal=None):
        inp  = [float(x) for x in input_list]
        flat = _ml_pick(self._h, self._legal_arg(legal), inp)
        out, units, raw = self._decode(flat)
        return Step(inp, out, units, raw)

    def _펼치기(self, v, 뭐):
        """출력이 하나면 값 하나를 그냥 받는다. 여러 개면 개수만큼의 리스트여야 한다."""
        if isinstance(v, (list, tuple)):
            return list(v)
        if self._n == 1:
            return [v]
        raise ValueError(
            f"{뭐}는 출력 개수({self._n})만큼의 리스트로 주세요 "
            f"(출력이 하나일 때만 값 하나로 줄 수 있습니다)")

    # ── 순수: 보상 붙이기 ──
    def reward(self, data, point):
        """point 는 출력 개수만큼의 리스트 (하나면 값 하나도 된다).
        None 이면 그 출력은 학습에서 빠진다."""
        point = self._펼치기(point, "점수")
        if len(point) != self._n:
            raise ValueError(f"점수 개수({len(point)})가 출력 개수({self._n})와 다릅니다")
        for i, pt in enumerate(point):
            if self._vecs[i] and pt is not None:
                raise ValueError(
                    f"{i}번째 출력은 vec 이라 점수를 줄 수 없습니다. "
                    "None 으로 비우고 jepa() 로 학습하세요")
        return Scored(data.input, data.output, list(point), data._units, data._raw)

    # ── 여기서만 학습 + 저장 ──
    def save(self, scored=None):
        """scored 를 주면 학습한다. 인자 없이 부르면 파일에만 쓴다.

        파일 쓰기는 망이 커지면 학습 자체보다 비싸다 ([64,512,512] 에서 학습
        19ms / 쓰기 34ms). 매 스텝 학습하는 온라인 RL 이면 make(..., autosave=N)
        으로 N 번에 한 번만 쓰게 하고, 끝날 때 ai.save() 로 마무리하면 된다.
        """
        if scored is None:
            _ml_save(self._h); self._since = 0; return 0
        batch = [scored] if isinstance(scored, Scored) else list(scored)
        if not batch:
            return 0
        inputs, chosen, values, points = [], [], [], []
        for s in batch:
            inputs.append([float(x) for x in s.input])
            chosen.append([int(u) for u in s._units])
            values.append([float(v) if c else 0.0
                           for v, (_, c) in zip(s._raw, self._heads)])
            points.append([_NAN if x is None else float(x) for x in s.point])
        self._메모경고()
        _ml_learn(self._h, inputs, chosen, values, points)
        if self._파일에쓸까():
            _ml_save(self._h)
        return len(batch)

    # ── 지도학습 ──
    def _정답풀기(self, answer):
        ansI = [0] * self._n
        ansV = [0.0] * self._n
        use  = [0] * self._n
        if answer is not None:
            answer = self._펼치기(answer, "정답")
            if len(answer) != self._n:
                raise ValueError(f"정답 개수({len(answer)})가 출력 개수({self._n})와 다릅니다")
            for i, a in enumerate(answer):
                if a is None: continue
                if self._vecs[i]:
                    raise ValueError(
                        f"{i}번째 출력은 vec 이라 sl() 로 정답을 줄 수 없습니다. "
                        "None 으로 비우고 jepa() 로 학습하세요")
                acts, cos = self._heads[i]
                use[i] = 1
                if cos: ansV[i] = float(a)
                else:   ansI[i] = acts.index(a)
        return ansI, ansV, use

    def sl(self, input_list, answer=None, legal=None):
        """정답을 주면 배우고 예측을 반환한다.

        입력을 여러 개 겹쳐 주면 한 번에 처리한다 (훨씬 빠르다).
            ai.sl(입력, 정답)                 # 하나
            ai.sl([입력1, 입력2, ...], [정답1, 정답2, ...])   # 묶음
        """
        self._메모경고()
        묶음 = bool(input_list) and isinstance(input_list[0], (list, tuple))
        if 묶음:
            if answer is None or len(answer) != len(input_list):
                raise ValueError("묶음으로 줄 때는 정답도 같은 개수만큼 주세요")
            lals = self._legal_arg(legal)
            inps, AI, AV, US = [], [], [], []
            for x, a in zip(input_list, answer):
                inps.append([float(v) for v in x])
                i_, v_, u_ = self._정답풀기(a)
                AI.append(i_); AV.append(v_); US.append(u_)
            _ml_sl_many(self._h, lals, inps, AI, AV, US)
            return None

        inp  = [float(x) for x in input_list]
        lals = self._legal_arg(legal)
        ansI, ansV, use = self._정답풀기(answer)
        flat = _ml_sl(self._h, lals, inp, ansI, ansV, use)
        out, _, _ = self._decode(flat)
        return out

    # ── 예측(샘플링 없음) ──
    def predict(self, input_list, legal=None):
        inp  = [float(x) for x in input_list]
        flat = _ml_predict(self._h, self._legal_arg(legal), inp)
        out, _, _ = self._decode(flat)
        return out

    # ── 스텝 묶음에 같은 보상 (순수) ──
    def episode(self, steps, point):
        return [self.reward(s, point) for s in steps]

    @property
    def heads(self):    return self._n
    @property
    def actions(self):  return [a for a, _ in self._heads]


def _헤드해석(spec):
    """스펙 하나 → (액션목록 또는 None, cos여부, 출력개수)"""
    if isinstance(spec, _Vec):
        if spec.size < 1: raise ValueError("vec(개수) 는 1 이상이어야 합니다")
        return None, True, spec.size
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


def _출력하나인가(x):
    """x 가 "출력 스펙 하나" 인가, 아니면 "스펙들의 리스트" 인가.

    리스트면 애매하다 — ["A","B"] 는 액션 목록 하나지만 [cos, "..."] 는 스펙 둘이다.
    문자열만 들었으면 액션 목록 하나로 본다. 그게 아니면 스펙들의 리스트다.
    (여기를 "리스트면 전부 스펙 하나" 로 뭉개면 [cos] 가 액션 목록으로 오해된다.)
    """
    if isinstance(x, (_Cos, _Vec)):
        return True
    if isinstance(x, str):
        return x.lower() == "cos"
    if isinstance(x, (list, tuple)):
        return bool(x) and all(isinstance(e, str) for e in x)
    return False


def make(model_name, layers, outputs, optimizer='adam', *,
         lr=0.01, decay=0.0, temp=1.0, sigma=1.0, entropy=0.01, autosave=1):
    """
    외울 것은 앞의 셋뿐이다. 나머지는 키워드로만 받으므로 순서를 몰라도 되고,
    만들 때 안 줘도 된다 — 전부 나중에 ai.lr = 0.02 처럼 바꿀 수 있다.

    (주의: 이 파이썬 코드는 D 의 백틱 문자열 안에 들어있다. 백틱을 쓰면 문자열이
     거기서 끝나버려서 빌드가 깨진다 — 주석에도 쓰지 말 것.)

    model_name : 모델 이름 (가중치 파일명)
    layers     : [입력수, 은닉...]   출력은 outputs 에서 정해진다
                 은닉 자리에 넣을 수 있는 것: 정수(일반 층), attn, each, conv,
                 memory, logic.  예) [38, 128, attn(8), each(32), 128]
    outputs    : 출력 하나면 그냥 줘도 되고 리스트로 감싸도 된다.
                   ["A","B"]           고르기 하나 (= [["A","B"]])
                   cos                 숫자 하나 (= [cos])
                   [["A","B"], cos]    두 개
                 출력이 여러 개면 반환·보상·정답·legal 도 그 개수만큼의 리스트다
                 (하나일 때는 그냥 값으로 줘도 받는다).
    sigma      : cos 가 값을 얼마나 넓게 탐험할지 (기본 1.0)
    entropy    : 고르는 쪽이 한 답으로 굳는 것을 막는 힘 (기본 0.01)
    autosave   : save(scored) 가 몇 번에 한 번 파일까지 쓸지 (기본 1 = 매번)
                 파일 쓰기는 망이 커지면 학습보다 비싸다. 매 스텝 학습하는
                 온라인 RL 이면 100 정도로 두고, 끝낼 때 ai.save() 로 마무리한다.
                 0 이면 자동으로 안 쓴다 (ai.save() 를 직접 불러야 한다).
    lr         : 학습률 — 틀렸을 때 얼마나 크게 고칠지 (기본 0.01)
                 안 배워지면 제일 먼저 만져볼 값이다. 크면 요동치고, 작으면 느리다.
    decay      : 가중치 감쇠 — 외우기(과적합)를 막는 힘 (기본 0 = 끔)
                 가중치를 0 쪽으로 살살 당겨서 꼭 필요한 것만 남게 한다.
                 1e-4 ~ 1e-2 정도부터 시작한다. 편향과 logic 게이트에는 안 걸린다.
    temp       : 고르기 샘플링 온도 (기본 1.0)
                 작으면 제일 좋아 보이는 것만, 크면 골고루 찔러본다.
                 0 이면 무작위성 없이 최선만 (배포용). predict() 에는 영향 없다.

    lr / decay / temp / sigma / entropy 는 만든 뒤에도 바꿀 수 있다 (ai.lr = 0.001).
    """
    if len(layers) < 1:
        raise ValueError("layers 는 [입력수, 은닉...] 형태입니다")
    inputSz = int(layers[0])
    lay_kind, lay_a, lay_b, lay_c = [], [], [], []
    폭 = inputSz
    항목수 = 0        # attn 이나 conv(..., 항목수) 를 만나야 항목 구조가 생긴다
    for i, L in enumerate(layers[1:], 1):
        if isinstance(L, _Attn):
            if 폭 % L.items:
                raise ValueError(
                    f"{i}번째 층 attn({L.items}): 폭 {폭} 이 조각 {L.items} 로 나뉘지 않습니다")
            조각폭 = 폭 // L.items
            if 조각폭 % L.heads:
                raise ValueError(
                    f"{i}번째 층 attn: 조각폭 {조각폭} 이 헤드 {L.heads} 로 나뉘지 않습니다")
            lay_kind.append(1); lay_a.append(L.items); lay_b.append(L.heads); lay_c.append(0)
            항목수 = L.items
            # 폭 그대로
        elif isinstance(L, _Conv):
            it = L.items if L.items else 항목수
            if it < 1:
                raise ValueError(
                    f"{i}번째 층 conv: 항목 수를 알 수 없습니다. 맨 앞에 쓸 때는 "
                    f"conv(채널, 창, 항목수) 처럼 직접 주세요 "
                    f"(입력폭 {폭} 만 보고는 '항목 n × 채널 c' 를 알 수 없습니다)")
            if 폭 % it:
                raise ValueError(
                    f"{i}번째 층 conv: 폭 {폭} 이 항목 {it} 로 나뉘지 않습니다")
            if L.win > it:
                raise ValueError(
                    f"{i}번째 층 conv: 창 {L.win} 이 항목 수 {it} 보다 큽니다")
            lay_kind.append(5); lay_a.append(L.ch); lay_b.append(L.win); lay_c.append(it)
            폭 = it * L.ch
            항목수 = it                      # 길이를 유지하므로 항목 구조도 유지된다
        elif isinstance(L, _Memory):
            lay_kind.append(4); lay_a.append(L.cells); lay_b.append(0); lay_c.append(0)
            폭 = L.cells
            항목수 = 0                       # 메모는 항목 구분을 유지하지 않는다
        elif isinstance(L, _Logic):
            lay_kind.append(3); lay_a.append(L.gates); lay_b.append(0); lay_c.append(0)
            폭 = L.gates
            항목수 = 0                       # 게이트는 항목 구분을 유지하지 않는다
        elif isinstance(L, _Each):
            it = L.items if L.items else 항목수
            if it < 1:
                raise ValueError(
                    f"{i}번째 층 each: 항목 수를 알 수 없습니다. 앞에 attn/conv 가 "
                    f"없으면 each(폭, 항목수) 처럼 직접 주세요 "
                    f"(입력폭 {폭} 만 보고는 항목이 몇 개인지 알 수 없습니다)")
            if 폭 % it:
                raise ValueError(
                    f"{i}번째 층 each: 폭 {폭} 이 항목 {it} 로 나뉘지 않습니다")
            lay_kind.append(2); lay_a.append(it); lay_b.append(L.width); lay_c.append(0)
            폭 = it * L.width
            항목수 = it                      # 항목 구조를 유지한다
        else:
            폭 = int(L)
            항목수 = 0                       # 일반 층은 항목 구분을 없앤다
            lay_kind.append(0); lay_a.append(폭); lay_b.append(0); lay_c.append(0)

    # 출력이 하나면 감싸지 않아도 받는다 (cos / vec(n) / ["A","B"]).
    if _출력하나인가(outputs):
        specs = [outputs]
    elif isinstance(outputs, (list, tuple)) and outputs:
        specs = list(outputs)
    else:
        raise ValueError(
            'outputs 가 비었습니다. cos / ["A","B"] / vec(n) 중 하나거나,'
            ' 여러 개면 그것들의 리스트로 주세요')

    heads, al_arg, cos_arg, sizes, vecs = [], [], [], [], []
    for spec in specs:
        names, is_cos, n = _헤드해석(spec)
        heads.append((names, is_cos))
        al_arg.append(names or [])
        cos_arg.append(1 if is_cos else 0)
        sizes.append(n)
        vecs.append(isinstance(spec, _Vec))

    h = _ml_make(model_name, inputSz, lay_kind, lay_a, lay_b, lay_c,
                 sizes, al_arg, cos_arg,
                 optimizer, float(sigma), float(entropy),
                 float(lr), float(decay), float(temp))
    return BlackBoxAI(h, model_name, heads, vecs, autosave, 4 in lay_kind)


class Jepa:
    """요약기 + 예측기 한 쌍.

    원본을 통째로 맞추는 대신 "요약"만 맞춘다.
      train(x들, y들, 행동들) : x 의 요약으로 y 의 요약을 맞추도록 학습. 손실 반환.
      encode(x)              : x -> 요약 (숫자 리스트)
      imagine(x, 행동)       : x (+행동) -> 다음 요약 예측 (한 걸음)
      roll(요약, 행동)       : 요약 -> 다음 요약 (여러 걸음 이어 붙일 때)
      save()                 : 요약기와 예측기 둘 다 저장
      encoder / predictor    : 안에 든 모델. 일반 학습을 같이 시킬 때 쓴다.
    """
    __slots__ = ("_h", "_enc", "_pred", "_d", "_a")

    def __init__(self, h, enc, pred):
        self._h, self._enc, self._pred = h, enc, pred
        self._d, self._a = _jepa_meta(h)

    @property
    def summary(self):  return self._d      # 요약 크기
    @property
    def actions(self):  return self._a      # 행동 입력 개수 (0 이면 없음)
    @property
    def encoder(self):  return self._enc
    @property
    def predictor(self): return self._pred

    def train(self, xs, ys, actions=None):
        """묶음으로 준다. xs[i] 와 ys[i] 가 한 쌍.

        묶음이 커야 배운다 — "요약들이 서로 다른가" 를 묶음 안에서 재기 때문에,
        2개 미만이면 붕괴 방지가 아예 꺼진다. 32개 이상을 권한다.
        """
        if len(xs) != len(ys):
            raise ValueError(f"x {len(xs)}개, y {len(ys)}개 — 개수가 같아야 합니다")
        if self._a > 0:
            if actions is None:
                raise ValueError(f"이 예측기는 행동 {self._a}개를 같이 받습니다")
            if len(actions) != len(xs):
                raise ValueError("행동도 x 와 같은 개수만큼 주세요")
        X = [[float(v) for v in r] for r in xs]
        Y = [[float(v) for v in r] for r in ys]
        A = None if actions is None else [[float(v) for v in r] for r in actions]
        return _jepa_train(self._h, X, Y, A)

    def encode(self, x):
        return _jepa_encode(self._h, [float(v) for v in x])

    def imagine(self, x, action=None):
        """관측 -> 다음 요약 예측. encode 한 뒤 roll 한 것과 같다."""
        a = None if action is None else [float(v) for v in action]
        return _jepa_imagine(self._h, [float(v) for v in x], a)

    def roll(self, summary, action=None):
        """요약 -> 다음 요약. 상상을 여러 걸음 이어 붙일 때 쓴다.

            s = w.encode(관측)
            for 행동 in 계획:
                s = w.roll(s, 행동)      # 머릿속으로만 굴린다

        imagine() 은 관측을 받으므로 한 걸음밖에 못 간다. 두 걸음부터는
        나온 요약을 그대로 다시 넣어야 하는데 그게 이것이다.
        """
        if len(summary) != self._d:
            raise ValueError(f"요약은 숫자 {self._d}개입니다 (받은 것: {len(summary)}개)")
        x = [float(v) for v in summary]
        if self._a > 0:
            if action is None:
                raise ValueError(f"이 예측기는 행동 {self._a}개를 같이 받습니다")
            if len(action) != self._a:
                raise ValueError(f"행동은 숫자 {self._a}개입니다 (받은 것: {len(action)}개)")
            x += [float(v) for v in action]
        elif action:
            raise ValueError("이 예측기는 행동을 받지 않습니다")
        return self._pred.embed(x, 0)

    def save(self):
        _jepa_save(self._h)

    def __repr__(self):
        return f"Jepa(summary={self._d}, actions={self._a})"


def jepa(name_or_encoder, inputs_or_predictor=None, *,
         summary=32, actions=0, hidden=128, outputs=None,
         var=25.0, cov=1.0, target=1.0, **knobs):
    """jepa 학습기를 만든다. 부르는 법이 두 가지다.

    ① 이름과 입력수만 준다 — 요약기·예측기를 알아서 만든다. 보통 이쪽.

        w = jepa("world", 24)                       # 행동 없음 (표현학습)
        w = jepa("world", 24, actions=2)            # 행동 2개 (월드모델)
        w = jepa("world", 24, summary=16, hidden=256, actions=2)

    ② 모델 두 개를 직접 만들어 묶는다 — 요약기에 attn/conv/memory 를 넣거나
       출력을 더 붙이고 싶을 때.

        enc  = make("enc",  [24, 128], [vec(32)])
        pred = make("pred", [32 + 2, 128], [vec(32)])
        w = jepa(enc, pred)

    어느 쪽이든 쓰는 법은 같다.

        w.train(지금들, 다음들, 행동들)   # 학습. 손실을 돌려준다
        w.encode(관측)                    # 관측 -> 요약
        w.imagine(관측, 행동)             # 관측 -> 다음 요약
        w.roll(요약, 행동)                # 요약 -> 다음 요약 (여러 걸음 이어 붙이기)
        w.save()                          # 둘 다 저장

    outputs 로 요약기에 출력을 더 붙이면 jepa 와 일반 학습이 몸통을 같이 쓴다.

        w = jepa("world", 24, actions=2, outputs=[["왼쪽","오른쪽"]])
        w.train(지금들, 다음들, 행동들)                 # 요약을 다듬는다 (라벨 없음)
        w.encoder.sl(관측들, [[None, "왼쪽"], ...])     # 행동을 가르친다

    요약 자리는 sl()/reward() 에서 None 으로 비우고 predict() 에서도 None 이
    나온다 — 그 값은 w.encode() 나 encoder.embed(입력, 0) 으로 꺼낸다.

    var / cov : 요약이 "다 같은 값으로 뭉개지는 것"을 막는 힘.
                var 는 값이 실제로 변하게, cov 는 칸끼리 딴 정보를 담게 민다.
                0 으로 두면 꺼지는데, 그러면 거의 확실히 뭉개진다.
    target    : 각 칸이 목표로 하는 값의 퍼짐 정도 (표준편차).
    그 밖의 키워드(lr, decay, autosave ...)는 두 모델에 그대로 넘어간다.
    """
    if isinstance(name_or_encoder, str):
        # ① 이름 + 입력수 -> 두 모델을 만든다
        name, body = name_or_encoder, inputs_or_predictor
        if body is None:
            raise ValueError('jepa("이름", 입력수) — 입력수를 같이 주세요')
        요약, 행동, 은닉 = int(summary), int(actions), int(hidden)
        if 요약 < 1:  raise ValueError("summary 는 1 이상이어야 합니다")
        if 행동 < 0:  raise ValueError("actions 는 0 이상이어야 합니다")
        if isinstance(body, (list, tuple)):
            enc_layers = list(body)        # [입력수, 은닉...] 를 직접 준 경우
        else:
            enc_layers = [int(body), 은닉]
        enc_outs = [vec(요약)] + list(outputs or [])
        encoder   = make(f"{name}_enc",  enc_layers, enc_outs, **knobs)
        predictor = make(f"{name}_pred", [요약 + 행동, 은닉], [vec(요약)], **knobs)
    else:
        encoder, predictor = name_or_encoder, inputs_or_predictor
        if outputs is not None:
            raise ValueError("outputs 는 ① 쪽에서만 씁니다 — ② 는 make() 에 직접 주세요")
        if not isinstance(encoder, BlackBoxAI) or not isinstance(predictor, BlackBoxAI):
            raise ValueError('jepa 는 jepa("이름", 입력수) 또는 jepa(요약기, 예측기) 로 부릅니다')
        for 이름, m in (("요약기", encoder), ("예측기", predictor)):
            if not m._vecs[0]:
                raise ValueError(f"{이름} 의 첫 출력이 vec 이어야 합니다 (지금: {m._heads[0][0] or 'cos'})")
    h = _jepa_make(encoder._h, predictor._h, float(var), float(cov), float(target))
    return Jepa(h, encoder, predictor)


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


def cpu_info():
    """어떤 SIMD 수준으로 돌고 있는지.

        {'simd': 'avx2',      # 실제로 고른 것: base | sse | avx2 | avx512
         'avx2': 1,           # 이 CPU 가 쓸 수 있는지
         'avx512': 0,
         'threads': 12,       # MYML_THREADS (기본: 전체 코어)
         'cores': 12}

    MYML_SIMD 로 **낮춰서** 쓸 수 있습니다 (base|sse|avx2|avx512). 없는 걸 올려서
    강제하는 건 막혀 있습니다 — 하드웨어가 없는데 쓰면 그 자리에서 죽습니다.

    avx512 경로는 만들어는 뒀지만 **실제 AVX-512 하드웨어에서 돌려본 적이
    없습니다** (개발 기계가 Zen 3+ 라 지원하지 않습니다). 해당 CPU 를 가지고
    계시면 "python tests/simd.py" 로 결과가 같은지 확인해주세요.
    """
    return _ml_cpu_info()


def gpu_info():
    """GPU 를 실제로 쓰고 있는지 확인한다.

        {'available': 1,            # OpenCL GPU 를 잡았나
         'device': 'gfx1035',       # 잡은 장치 이름
         'mode': 'auto',            # MYML_GPU 설정
         'runs': 12,                # GPU 경로가 실제로 끝까지 돈 횟수
         'min_flops': 50000000,     # auto 모드 문턱값
         'min_batch': 64}

    available 이 1이어도 runs 가 0이면 문턱값을 못 넘어 CPU 로만 돌았다는 뜻이다.
    """
    return _ml_gpu_info()


def gc_disable(): _ml_gc_disable()
def gc_collect(): _ml_gc_collect()
def resset(model_name): _ml_resset(model_name)

import builtins
builtins.resset     = resset
builtins.change     = change
builtins.gc_disable = gc_disable
builtins.gc_collect = gc_collect
builtins.gpu_info   = gpu_info
`;

// ─────────────────────────────────────────────
// Module init
// ─────────────────────────────────────────────
private __gshared PyMethodDef[26] _methods;
private __gshared PyModuleDef     _moddef;

extern(C) export PyObject* PyInit_ml() nothrow @trusted {
    try {
        _methods[0] = PyMethodDef("_ml_make",            &py_ml_make,            METH_VARARGS, null);
        _methods[1] = PyMethodDef("_ml_pick",            &py_ml_pick,            METH_VARARGS, null);
        _methods[2] = PyMethodDef("_ml_predict",         &py_ml_predict,         METH_VARARGS, null);
        _methods[3] = PyMethodDef("_ml_learn",           &py_ml_learn,           METH_VARARGS, null);
        _methods[4] = PyMethodDef("_ml_sl",              &py_ml_sl,              METH_VARARGS, null);
        _methods[5] = PyMethodDef("_ml_save",            &py_ml_save,            METH_VARARGS, null);
        _methods[6] = PyMethodDef("_ml_resset",          &py_ml_resset,          METH_VARARGS, null);
        _methods[7] = PyMethodDef("_ml_gc_disable",      &py_ml_gc_disable,      METH_VARARGS, null);
        _methods[8] = PyMethodDef("_ml_gc_collect",      &py_ml_gc_collect,      METH_VARARGS, null);
        _methods[9] = PyMethodDef("_ml_export_weights",  &py_ml_export_weights,  METH_VARARGS, null);
        _methods[10] = PyMethodDef("_ml_get_meta",        &py_ml_get_meta,        METH_VARARGS, null);
        _methods[11] = PyMethodDef("_ml_change",          &py_ml_change,          METH_VARARGS, null);
        _methods[12] = PyMethodDef("_ml_sl_many",        &py_ml_sl_many,         METH_VARARGS, null);
        _methods[13] = PyMethodDef("_ml_embed",          &py_ml_embed,           METH_VARARGS, null);
        _methods[14] = PyMethodDef("_jepa_make",         &py_jepa_make,          METH_VARARGS, null);
        _methods[15] = PyMethodDef("_jepa_train",        &py_jepa_train,         METH_VARARGS, null);
        _methods[16] = PyMethodDef("_jepa_encode",       &py_jepa_encode,        METH_VARARGS, null);
        _methods[17] = PyMethodDef("_jepa_imagine",      &py_jepa_imagine,       METH_VARARGS, null);
        _methods[18] = PyMethodDef("_jepa_save",         &py_jepa_save,          METH_VARARGS, null);
        _methods[19] = PyMethodDef("_jepa_meta",         &py_jepa_meta,          METH_VARARGS, null);
        _methods[20] = PyMethodDef("_ml_gpu_info",       &py_ml_gpu_info,        METH_VARARGS, null);
        _methods[21] = PyMethodDef("_ml_logic_rules",    &py_ml_logic_rules,     METH_VARARGS, null);
        _methods[22] = PyMethodDef("_ml_tune",           &py_ml_tune,            METH_VARARGS, null);
        _methods[23] = PyMethodDef("_ml_forget",         &py_ml_forget,          METH_VARARGS, null);
        _methods[24] = PyMethodDef("_ml_cpu_info",       &py_ml_cpu_info,        METH_VARARGS, null);
        _methods[25] = PyMethodDef(null, null, 0, null);

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
