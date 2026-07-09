// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 rnjswldbf2014-hash
module my_ml;

import std.string : fromStringz, toStringz;
import core.memory : GC;

version(Windows) {
    import core.sys.windows.windows;
    import core.runtime : rt_init, rt_term;
    import core.thread : thread_attachThis, thread_detachThis;

    extern(Windows) BOOL DllMain(HINSTANCE h, ULONG reason, LPVOID) nothrow {
        try {
            switch (reason) {
                case DLL_PROCESS_ATTACH: rt_init(); break;
                case DLL_PROCESS_DETACH: rt_term(); break;
                case DLL_THREAD_ATTACH:  thread_attachThis(); break;
                case DLL_THREAD_DETACH:  thread_detachThis(); break;
                default:
            }
        } catch (Throwable) {}
        return TRUE;
    }
}

import std.stdio  : writefln, File;
import std.math   : exp, sqrt, pow;
import std.random : Random, uniform, uniform01, unpredictableSeed;
import std.file   : exists, remove;
import std.algorithm : map, countUntil, min;
import std.array  : array;
import std.conv   : to;

private Random rng;
static this() { rng = Random(unpredictableSeed); }

// ── CPU Dispatch ──────────────────────────────────────────────────────────
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
        b     = new float[o];
        mB    = new float[o];
        vB    = new float[o];
        gradB = new float[o];
        foreach (j; 0..o) {
            b[j] = mB[j] = vB[j] = gradB[j] = 0f;
            foreach (k; 0..i) mW[j][k] = vW[j][k] = gradW[j][k] = 0f;
        }
        float s = sqrt(2.0f / (i + o));
        foreach (j; 0..o) foreach (k; 0..i) w[j][k] = uniform(-s, s, rng);
    }

    float[] forward(float[] x) {
        auto y = new float[outSz];
        foreach (j; 0..outSz)
            y[j] = b[j] + _dot(w[j], x);
        return y;
    }

    float[] accum(float[] x, float[] dOut) {
        auto dIn = new float[inSz];
        dIn[] = 0f;
        foreach (j; 0..outSz) {
            _saxpy(dIn,      w[j], dOut[j]);
            _saxpy(gradW[j], x,    dOut[j]);
            gradB[j] += dOut[j];
        }
        return dIn;
    }

    void zeroGrad() {
        foreach (j; 0..outSz) gradW[j][] = 0f;
        gradB[] = 0f;
    }

    void step(Opt opt, float lr) {
        enum float B1=0.9f, B2=0.999f, EPS=1e-8f, RHO=0.99f;
        final switch (opt) {
            case Opt.adam:
                t++;
                float bc1 = 1f - B1^^t;
                float bc2 = 1f - B2^^t;
                foreach (j; 0..outSz) {
                    foreach (k; 0..inSz) {
                        float g = gradW[j][k];
                        mW[j][k] = B1*mW[j][k] + (1-B1)*g;
                        vW[j][k] = B2*vW[j][k] + (1-B2)*g*g;
                        w[j][k] -= lr * (mW[j][k]/bc1) / (sqrt(vW[j][k]/bc2) + EPS);
                    }
                    float gb = gradB[j];
                    mB[j] = B1*mB[j] + (1-B1)*gb;
                    vB[j] = B2*vB[j] + (1-B2)*gb*gb;
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
// Neural network
// ─────────────────────────────────────────────
private class Network {
    Linear[] hidden;
    Linear[] heads;
    int      inputSz;

    this(int inputSz, int[] hiddenSizes, int[] headSizes) {
        this.inputSz = inputSz;
        int prev = inputSz;
        foreach (sz; hiddenSizes) { hidden ~= Linear(prev, sz); prev = sz; }
        foreach (sz; headSizes)   { heads  ~= Linear(prev, sz); }
    }

    struct Fwd {
        float[][] layerIn;
        float[][] preReLU;
        float[]   hiddenOut;
        float[][] headOut;
    }

    Fwd forward(float[] x) {
        Fwd f;
        f.layerIn = new float[][hidden.length];
        f.preReLU = new float[][hidden.length];
        f.headOut = new float[][heads.length];
        auto cur = x;
        foreach (i, ref h; hidden) {
            f.layerIn[i] = cur.dup;
            auto z = h.forward(cur);
            f.preReLU[i] = z.dup;
            foreach (ref v; z) if (v < 0f) v = 0f;
            cur = z;
        }
        f.hiddenOut = cur;
        foreach (i, ref head; heads) f.headOut[i] = head.forward(cur);
        return f;
    }

    void backward(ref Fwd f, float[][] dHead) {
        auto dH = new float[f.hiddenOut.length];
        dH[] = 0f;
        foreach (i, ref head; heads) {
            if (dHead[i] is null) continue;
            auto d = head.accum(f.hiddenOut, dHead[i]);
            foreach (k; 0..dH.length) dH[k] += d[k];
        }
        for (int i = cast(int)hidden.length-1; i >= 0; i--) {
            foreach (k; 0..dH.length) if (f.preReLU[i][k] <= 0f) dH[k] = 0f;
            dH = hidden[i].accum(f.layerIn[i], dH);
        }
    }

    void zeroGrad() {
        foreach (ref h; hidden) h.zeroGrad();
        foreach (ref h; heads)  h.zeroGrad();
    }

    void step(Opt opt, float lr) {
        foreach (ref h; hidden) h.step(opt, lr);
        foreach (ref h; heads)  h.step(opt, lr);
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
private float[] softmax(float[] x) {
    float mx = x[0]; foreach (v; x) if (v > mx) mx = v;
    auto e = x.map!(v => exp(v - mx)).array;
    float s = 0f; foreach (v; e) s += v;
    foreach (ref v; e) v /= s;
    return e;
}

private int[] buildMask(string[] all, string[] legal) {
    int[] idx;
    foreach (a; legal)
        foreach (j, ac; all)
            if (ac == a) { idx ~= cast(int)j; break; }
    return idx;
}

// ─────────────────────────────────────────────
// RL step cache
// ─────────────────────────────────────────────
private struct RLStep {
    float[]   input;
    int[][]   mask;
    int[]     chosen;
    float[][] probs;
}

// ─────────────────────────────────────────────
// BlackBoxAI
// ─────────────────────────────────────────────
class BlackBoxAI {
    string     name;
    string[][] actionLists;
    int[]      hiddenSizes;
    float      lr = 0.01f;
    Opt        opt;
    string     file;

    private Network  net;
    private bool     ready;
    private RLStep   lastStep;
    private bool     hasLast;
    private RLStep[] episodeBuf;
    private bool     inEpisode;

    this(string name, string[][] lists, int[] hidden, Opt opt = Opt.adam, bool reset = false) {
        this.name   = name;
        actionLists = lists;
        hiddenSizes = hidden;
        this.opt    = opt;
        file = name ~ "_ml_memory.pth";
        if (reset && exists(file)) {
            remove(file);
            writefln(" [%s] 기존 학습 데이터를 삭제하고 초기화했습니다.", name);
        }
    }

    private void ensureNet(int inputSz) {
        if (ready) return;
        int[] headSz;
        foreach (al; actionLists) headSz ~= cast(int)al.length;

        if (exists(file)) {
            try   { load(); writefln(" [%s] 이전 학습 데이터를 성공적으로 불러왔습니다!", name); }
            catch (Exception e) {
                writefln(" [%s] 데이터 불러오기 실패 (%s). 새로 시작합니다.", name, e.msg);
                if (!isTorchFile(file)) remove(file);
                net = new Network(inputSz, hiddenSizes, headSz);
                writefln(" [%s] 새로 생성되었습니다.", name);
            }
        } else {
            net = new Network(inputSz, hiddenSizes, headSz);
            writefln(" [%s] 새로 생성되었습니다.", name);
        }
        ready = true;
    }

    string[] rl(string[][] legal, float[] env) {
        ensureNet(cast(int)env.length);
        auto f = net.forward(env);
        int[][] mask; int[] chosen; float[][] probs; string[] result;

        foreach (i; 0..actionLists.length) {
            auto lal = (i < legal.length) ? legal[i] : legal[0];
            int[] idx = buildMask(actionLists[i], lal);
            auto ml = idx.map!(k => f.headOut[i][k]).array;
            auto p  = softmax(ml);
            mask  ~= idx; probs ~= p;

            double r = uniform01!double(rng);
            double cum = 0.0;
            int c = cast(int)(p.length - 1);
            foreach (k, pv; p) { cum += pv; if (r < cum) { c = cast(int)k; break; } }
            chosen ~= c; result ~= lal[c];
        }

        lastStep = RLStep(env.dup, mask, chosen, probs);
        hasLast  = true;
        if (inEpisode) episodeBuf ~= lastStep;
        return result;
    }

    private float[][] rlGrads(ref RLStep s, float score) {
        auto dH = new float[][actionLists.length];
        foreach (i; 0..actionLists.length) {
            dH[i] = new float[actionLists[i].length];
            dH[i][] = 0f;
            foreach (k, idx; s.mask[i])
                dH[i][idx] = score * (s.probs[i][k] - (k == s.chosen[i] ? 1f : 0f));
        }
        return dH;
    }

    private void applyStep(ref RLStep s, float score) {
        auto f  = net.forward(s.input);
        auto dH = rlGrads(s, score);
        net.backward(f, dH);
    }

    void reward(float score) {
        if (!ready || !hasLast) return;
        net.zeroGrad();
        applyStep(lastStep, score);
        net.step(opt, lr);
        if (inEpisode && episodeBuf.length > 0) episodeBuf = episodeBuf[0..$-1];
        hasLast = false;
    }

    void beginEpisode() { episodeBuf = []; inEpisode = true; }
    void endEpisode()   { inEpisode = false; }

    void lastReward(float score) {
        if (episodeBuf.length == 0) {
            writefln(" [%s] 에피소드 버퍼가 비어 있습니다.", name); return;
        }
        net.zeroGrad();
        foreach (ref s; episodeBuf) applyStep(s, score);
        net.step(opt, lr);
        episodeBuf = [];
    }

    string[] sl(string[][] legal, float[] env, string[] answers = null) {
        ensureNet(cast(int)env.length);

        if (answers && answers.length > 0) {
            net.zeroGrad();
            auto f  = net.forward(env);
            auto dH = new float[][actionLists.length];
            foreach (i; 0..min(answers.length, actionLists.length)) {
                auto lal = (i < legal.length) ? legal[i] : legal[0];
                int[] idx = buildMask(actionLists[i], lal);
                auto ml = idx.map!(k => f.headOut[i][k]).array;
                auto p  = softmax(ml);
                int  tgt = cast(int)lal.countUntil(answers[i]);
                dH[i] = new float[actionLists[i].length];
                dH[i][] = 0f;
                foreach (k, midx; idx)
                    dH[i][midx] = p[k] - (k == tgt ? 1f : 0f);
            }
            net.backward(f, dH);
            net.step(opt, lr);
        }

        auto f2 = net.forward(env);
        string[] result;
        foreach (i; 0..actionLists.length) {
            auto lal = (i < legal.length) ? legal[i] : legal[0];
            int[] idx = buildMask(actionLists[i], lal);
            auto ml = idx.map!(k => f2.headOut[i][k]).array;
            int best = 0;
            foreach (k; 1..ml.length) if (ml[k] > ml[best]) best = cast(int)k;
            result ~= lal[best];
        }
        return result;
    }

    void save() {
        if (!ready) return;
        auto f = File(file, "wb");
        void wu(uint v)  { f.rawWrite((&v)[0..1]); }
        void wf(float v) { f.rawWrite((&v)[0..1]); }

        wu(0xBEEFCAFE); wu(2);
        wu(cast(uint)opt);
        wu(cast(uint)net.inputSz);
        wu(cast(uint)hiddenSizes.length);
        foreach (sz; hiddenSizes) wu(cast(uint)sz);
        wu(cast(uint)actionLists.length);
        foreach (al; actionLists) {
            wu(cast(uint)al.length);
            foreach (a; al) {
                auto bytes = cast(ubyte[])a;
                wu(cast(uint)bytes.length);
                f.rawWrite(bytes);
            }
        }

        void wl(ref Linear l) {
            foreach (row; l.w)  foreach (v; row)  wf(v);
            foreach (v;  l.b)   wf(v);
            foreach (row; l.mW) foreach (v; row)  wf(v);
            foreach (row; l.vW) foreach (v; row)  wf(v);
            foreach (v;  l.mB)  wf(v);
            foreach (v;  l.vB)  wf(v);
            wu(cast(uint)l.t);
        }
        foreach (ref h; net.hidden) wl(h);
        foreach (ref h; net.heads)  wl(h);
        writefln(" [%s] 저장 완료.", name);
    }

    private void load() {
        auto f = File(file, "rb");
        uint ru()  { uint v;  f.rawRead((&v)[0..1]); return v; }
        float rf() { float v; f.rawRead((&v)[0..1]); return v; }

        if (ru() != 0xBEEFCAFE) throw new Exception("magic mismatch");
        uint ver = ru();
        if (ver >= 2) opt = cast(Opt)ru();
        int inputSz = ru();
        int nH = ru();
        hiddenSizes = new int[nH];
        foreach (i; 0..nH) hiddenSizes[i] = ru();
        int nL = ru();
        actionLists = new string[][nL];
        foreach (i; 0..nL) {
            int nA = ru();
            actionLists[i] = new string[nA];
            foreach (j; 0..nA) {
                auto buf = new ubyte[ru()];
                f.rawRead(buf);
                actionLists[i][j] = cast(string)buf.dup;
            }
        }
        int[] headSz;
        foreach (al; actionLists) headSz ~= cast(int)al.length;
        net = new Network(inputSz, hiddenSizes, headSz);

        void rl_(ref Linear l) {
            foreach (ref row; l.w)  foreach (ref v; row)  v = rf();
            foreach (ref v;  l.b)   v = rf();
            foreach (ref row; l.mW) foreach (ref v; row)  v = rf();
            foreach (ref row; l.vW) foreach (ref v; row)  v = rf();
            foreach (ref v;  l.mB)  v = rf();
            foreach (ref v;  l.vB)  v = rf();
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

private string optToStr(Opt o) pure nothrow {
    final switch (o) {
        case Opt.adam:    return "adam";
        case Opt.sgd:     return "sgd";
        case Opt.rmsprop: return "rmsprop";
        case Opt.adagrad: return "adagrad";
    }
}

private bool isTorchFile(string path) nothrow {
    try {
        auto f = File(path, "rb");
        ubyte[2] magic;
        f.rawRead(magic[]);
        return magic[0] == 0x50 && magic[1] == 0x4B; // ZIP magic: PK
    } catch (Exception) { return false; }
}

// ─────────────────────────────────────────────
// Python C API declarations
// ─────────────────────────────────────────────
private:

alias Py_ssize_t = long;

struct PyObject { Py_ssize_t ob_refcnt; void* ob_type; }

struct PyModuleDef_Base {
    PyObject   ob_base;
    void*      m_init;
    Py_ssize_t m_index;
    void*      m_copy;
}

struct PyModuleDef {
    PyModuleDef_Base m_base;
    const(char)*     m_name;
    const(char)*     m_doc;
    Py_ssize_t       m_size;
    PyMethodDef*     m_methods;
    void*            m_slots;
    void*            m_traverse;
    void*            m_clear;
    void*            m_free;
}

alias PyCFunction = extern(C) PyObject* function(PyObject*, PyObject*) nothrow;
alias PyCapsuleDestructor = extern(C) void function(PyObject*) nothrow;

struct PyMethodDef {
    const(char)* ml_name;
    PyCFunction  ml_meth;
    int          ml_flags;
    const(char)* ml_doc;
}

enum METH_VARARGS = 0x0001;
enum PYTHON_API_VERSION = 1013;
enum Py_file_input  = 257;
enum Py_eval_input  = 258;

// Initialized in PyInit_my_ml to avoid DLL data-import issues on Windows
private __gshared PyObject* _pyNone;
private __gshared PyObject* _pyRuntimeError;

private extern(C) nothrow @nogc {
    void        Py_IncRef(PyObject*);
    void        Py_DecRef(PyObject*);
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

// ─────────────────────────────────────────────
// Python <-> D type helpers
// ─────────────────────────────────────────────
private @trusted:

string[] pyStrList(PyObject* lst) {
    string[] r;
    foreach (i; 0..PyList_Size(lst))
        r ~= fromStringz(PyUnicode_AsUTF8(PyList_GetItem(lst, i))).idup;
    return r;
}

string[][] pyLals(PyObject* lals) {
    string[][] r;
    foreach (i; 0..PyList_Size(lals))
        r ~= pyStrList(PyList_GetItem(lals, i));
    return r;
}

float[] pyFloatList(PyObject* lst) {
    float[] r;
    foreach (i; 0..PyList_Size(lst))
        r ~= cast(float) PyFloat_AsDouble(PyList_GetItem(lst, i));
    return r;
}

int[] pyIntList(PyObject* lst) {
    int[] r;
    foreach (i; 0..PyList_Size(lst))
        r ~= cast(int) PyLong_AsLong(PyList_GetItem(lst, i));
    return r;
}

PyObject* toPyList(string[] strs) {
    auto lst = PyList_New(strs.length);
    foreach (i, s; strs)
        PyList_SetItem(lst, i, PyUnicode_FromString(toStringz(s)));
    return lst;
}

// capsule destructor: called when Python GCs the capsule
extern(C) void bbai_dtor(PyObject* cap) nothrow @trusted {
    auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
    if (ai) try { GC.removeRoot(cast(void*) ai); } catch (Throwable) {}
}

// ─────────────────────────────────────────────
// Python extension functions (_ml_* → Python)
// ─────────────────────────────────────────────
extern(C) nothrow @trusted:

// _ml_make(name, all_lists, hidden, optimizer, reset) → capsule
PyObject* py_ml_make(PyObject* self, PyObject* args) {
    try {
        PyObject* nm;  PyObject* lals; PyObject* hid; PyObject* opt;
        int reset;
        if (!PyArg_ParseTuple(args, "OOOOi", &nm, &lals, &hid, &opt, &reset)) return null;

        string   name   = fromStringz(PyUnicode_AsUTF8(nm)).idup;
        string   optStr = fromStringz(PyUnicode_AsUTF8(opt)).idup;
        string[][] al   = pyLals(lals);
        int[]    hidden = pyIntList(hid);
        if (hidden.length == 0) hidden = [128];

        auto ai = new BlackBoxAI(name, al, hidden, parseOpt(optStr), reset != 0);
        GC.addRoot(cast(void*) ai);
        return PyCapsule_New(cast(void*) ai, "BlackBoxAI", &bbai_dtor);
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_make");
        return null;
    }
}

// _ml_rl(handle, lals, env) → list[str]
PyObject* py_ml_rl(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* lals; PyObject* env;
        if (!PyArg_ParseTuple(args, "OOO", &cap, &lals, &env)) return null;
        auto ai  = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto res = ai.rl(pyLals(lals), pyFloatList(env));
        return toPyList(res);
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_rl");
        return null;
    }
}

// _ml_sl(handle, lals, env, answers) → list[str]
PyObject* py_ml_sl(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; PyObject* lals; PyObject* env; PyObject* ans;
        if (!PyArg_ParseTuple(args, "OOOO", &cap, &lals, &env, &ans)) return null;
        auto ai      = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        string[] answers = PyList_Size(ans) > 0 ? pyStrList(ans) : null;
        auto res     = ai.sl(pyLals(lals), pyFloatList(env), answers);
        return toPyList(res);
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_sl");
        return null;
    }
}

// _ml_reward(handle, score) → None
PyObject* py_ml_reward(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; double score;
        if (!PyArg_ParseTuple(args, "Od", &cap, &score)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).reward(cast(float)score);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_reward");
        return null;
    }
}

// _ml_begin_episode / _ml_end_episode / _ml_save (handle only)
PyObject* py_ml_begin_episode(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).beginEpisode();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_begin_episode");
        return null;
    }
}

PyObject* py_ml_end_episode(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).endEpisode();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_end_episode");
        return null;
    }
}

// _ml_last_reward(handle, score) → None
PyObject* py_ml_last_reward(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; double score;
        if (!PyArg_ParseTuple(args, "Od", &cap, &score)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).lastReward(cast(float)score);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_last_reward");
        return null;
    }
}

PyObject* py_ml_save(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        (cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI")).save();
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_save");
        return null;
    }
}

// _ml_resset(name) → None
PyObject* py_ml_resset(PyObject* self, PyObject* args) {
    try {
        PyObject* nm;
        if (!PyArg_ParseTuple(args, "O", &nm)) return null;
        resset(fromStringz(PyUnicode_AsUTF8(nm)).idup);
        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError,"my_ml: exception in _ml_resset");
        return null;
    }
}

// _ml_export_weights(handle) → dict of flat float lists (PyTorch state_dict keys)
PyObject* py_ml_export_weights(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai  = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto d   = PyDict_New();
        if (!ai || !ai.ready) return d;
        auto net = ai.net;

        foreach (i, ref h; net.hidden) {
            auto wflat = PyList_New(h.outSz * h.inSz);
            Py_ssize_t idx = 0;
            foreach (row; h.w) foreach (v; row)
                PyList_SetItem(wflat, idx++, PyFloat_FromDouble(v));
            string wkey = "hidden." ~ to!string(2*i) ~ ".weight";
            PyDict_SetItemString(d, toStringz(wkey), wflat);
            Py_DecRef(wflat);

            auto wshape = PyList_New(2);
            PyList_SetItem(wshape, 0, PyLong_FromLong(h.outSz));
            PyList_SetItem(wshape, 1, PyLong_FromLong(h.inSz));
            PyDict_SetItemString(d, toStringz(wkey ~ ".shape"), wshape);
            Py_DecRef(wshape);

            auto bflat = PyList_New(h.outSz);
            foreach (j; 0..h.outSz)
                PyList_SetItem(bflat, j, PyFloat_FromDouble(h.b[j]));
            PyDict_SetItemString(d, toStringz("hidden." ~ to!string(2*i) ~ ".bias"), bflat);
            Py_DecRef(bflat);
        }
        foreach (i, ref h; net.heads) {
            auto wflat = PyList_New(h.outSz * h.inSz);
            Py_ssize_t idx = 0;
            foreach (row; h.w) foreach (v; row)
                PyList_SetItem(wflat, idx++, PyFloat_FromDouble(v));
            string wkey = "output_layers." ~ to!string(i) ~ ".weight";
            PyDict_SetItemString(d, toStringz(wkey), wflat);
            Py_DecRef(wflat);

            auto wshape = PyList_New(2);
            PyList_SetItem(wshape, 0, PyLong_FromLong(h.outSz));
            PyList_SetItem(wshape, 1, PyLong_FromLong(h.inSz));
            PyDict_SetItemString(d, toStringz(wkey ~ ".shape"), wshape);
            Py_DecRef(wshape);

            auto bflat = PyList_New(h.outSz);
            foreach (j; 0..h.outSz)
                PyList_SetItem(bflat, j, PyFloat_FromDouble(h.b[j]));
            PyDict_SetItemString(d, toStringz("output_layers." ~ to!string(i) ~ ".bias"), bflat);
            Py_DecRef(bflat);
        }

        auto isz  = PyLong_FromLong(net.inputSz);
        PyDict_SetItemString(d, "input_size", isz);
        Py_DecRef(isz);

        auto opts = PyUnicode_FromString(toStringz(optToStr(ai.opt)));
        PyDict_SetItemString(d, "optimizer_name", opts);
        Py_DecRef(opts);

        return d;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_export_weights");
        return null;
    }
}

// _ml_import_weights(handle, input_size, flat_dict) — load PyTorch-format weights into D network
PyObject* py_ml_import_weights(PyObject* self, PyObject* args) {
    try {
        PyObject* cap; int inputSz; PyObject* wdict;
        if (!PyArg_ParseTuple(args, "OiO", &cap, &inputSz, &wdict)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");

        if (!ai.ready) {
            int[] headSz;
            foreach (al; ai.actionLists) headSz ~= cast(int)al.length;
            ai.net   = new Network(inputSz, ai.hiddenSizes, headSz);
            ai.ready = true;
        }

        auto net = ai.net;
        foreach (i, ref h; net.hidden) {
            auto wkey  = toStringz("hidden." ~ to!string(2*i) ~ ".weight");
            auto wlist = PyDict_GetItemString(wdict, wkey);
            if (wlist) {
                Py_ssize_t idx = 0;
                foreach (j; 0..h.outSz) foreach (k; 0..h.inSz)
                    h.w[j][k] = cast(float) PyFloat_AsDouble(PyList_GetItem(wlist, idx++));
            }
            auto blist = PyDict_GetItemString(wdict, toStringz("hidden." ~ to!string(2*i) ~ ".bias"));
            if (blist)
                foreach (j; 0..h.outSz)
                    h.b[j] = cast(float) PyFloat_AsDouble(PyList_GetItem(blist, j));
        }
        foreach (i, ref h; net.heads) {
            auto wkey  = toStringz("output_layers." ~ to!string(i) ~ ".weight");
            auto wlist = PyDict_GetItemString(wdict, wkey);
            if (wlist) {
                Py_ssize_t idx = 0;
                foreach (j; 0..h.outSz) foreach (k; 0..h.inSz)
                    h.w[j][k] = cast(float) PyFloat_AsDouble(PyList_GetItem(wlist, idx++));
            }
            auto blist = PyDict_GetItemString(wdict, toStringz("output_layers." ~ to!string(i) ~ ".bias"));
            if (blist)
                foreach (j; 0..h.outSz)
                    h.b[j] = cast(float) PyFloat_AsDouble(PyList_GetItem(blist, j));
        }

        Py_IncRef(_pyNone); return _pyNone;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_import_weights");
        return null;
    }
}

// _ml_get_meta(handle) → dict with model_name, action_lists, hidden_layers, optimizer_name
PyObject* py_ml_get_meta(PyObject* self, PyObject* args) {
    try {
        PyObject* cap;
        if (!PyArg_ParseTuple(args, "O", &cap)) return null;
        auto ai = cast(BlackBoxAI) PyCapsule_GetPointer(cap, "BlackBoxAI");
        auto d  = PyDict_New();

        auto nm = PyUnicode_FromString(toStringz(ai.name));
        PyDict_SetItemString(d, "model_name", nm);
        Py_DecRef(nm);

        auto al_outer = PyList_New(ai.actionLists.length);
        foreach (i, al; ai.actionLists) {
            auto al_inner = toPyList(al);
            PyList_SetItem(al_outer, i, al_inner);
        }
        PyDict_SetItemString(d, "action_lists", al_outer);
        Py_DecRef(al_outer);

        auto hl = PyList_New(ai.hiddenSizes.length);
        foreach (i, sz; ai.hiddenSizes)
            PyList_SetItem(hl, i, PyLong_FromLong(sz));
        PyDict_SetItemString(d, "hidden_layers", hl);
        Py_DecRef(hl);

        auto opts = PyUnicode_FromString(toStringz(optToStr(ai.opt)));
        PyDict_SetItemString(d, "optimizer_name", opts);
        Py_DecRef(opts);

        return d;
    } catch (Throwable) {
        PyErr_SetString(_pyRuntimeError, "my_ml: exception in _ml_get_meta");
        return null;
    }
}

// ─────────────────────────────────────────────
// Python class/API — embedded as a string,
// executed in the module dict at import time
// ─────────────────────────────────────────────
private enum string PY_CLASS_CODE = import("my_ml_class.py");

// ─────────────────────────────────────────────
// Module definition & PyInit
// ─────────────────────────────────────────────
private __gshared PyMethodDef[13] _methods;
private __gshared PyModuleDef     _moddef;

extern(C) PyObject* PyInit_my_ml() nothrow @trusted {
    try {
        _methods[0] = PyMethodDef("_ml_make",          &py_ml_make,          METH_VARARGS, null);
        _methods[1] = PyMethodDef("_ml_rl",            &py_ml_rl,            METH_VARARGS, null);
        _methods[2] = PyMethodDef("_ml_sl",            &py_ml_sl,            METH_VARARGS, null);
        _methods[3] = PyMethodDef("_ml_reward",        &py_ml_reward,        METH_VARARGS, null);
        _methods[4] = PyMethodDef("_ml_begin_episode", &py_ml_begin_episode, METH_VARARGS, null);
        _methods[5] = PyMethodDef("_ml_end_episode",   &py_ml_end_episode,   METH_VARARGS, null);
        _methods[6] = PyMethodDef("_ml_last_reward",   &py_ml_last_reward,   METH_VARARGS, null);
        _methods[7] = PyMethodDef("_ml_save",          &py_ml_save,          METH_VARARGS, null);
        _methods[8]  = PyMethodDef("_ml_resset",         &py_ml_resset,         METH_VARARGS, null);
        _methods[9]  = PyMethodDef("_ml_export_weights", &py_ml_export_weights, METH_VARARGS, null);
        _methods[10] = PyMethodDef("_ml_import_weights", &py_ml_import_weights, METH_VARARGS, null);
        _methods[11] = PyMethodDef("_ml_get_meta",       &py_ml_get_meta,       METH_VARARGS, null);
        _methods[12] = PyMethodDef(null, null, 0, null);

        _moddef.m_base.ob_base.ob_refcnt = 1;
        _moddef.m_name    = "my_ml";
        _moddef.m_doc     = null;
        _moddef.m_size    = -1;
        _moddef.m_methods = _methods.ptr;

        auto mod = PyModule_Create2(&_moddef, PYTHON_API_VERSION);
        if (!mod) return null;

        // Bootstrap: get None and RuntimeError without linking DLL data symbols
        auto globals = PyModule_GetDict(mod);
        auto none_obj = PyRun_String("None", Py_eval_input, globals, globals);
        if (none_obj) { _pyNone = none_obj; Py_IncRef(_pyNone); Py_DecRef(none_obj); }
        auto err_obj  = PyRun_String("RuntimeError", Py_eval_input, globals, globals);
        if (err_obj)  { _pyRuntimeError = err_obj; Py_IncRef(_pyRuntimeError); Py_DecRef(err_obj); }

        auto res = PyRun_String(PY_CLASS_CODE.ptr, Py_file_input, globals, globals);
        if (!res) return null;
        Py_DecRef(res);

        return mod;
    } catch (Throwable) {
        return null;
    }
}
