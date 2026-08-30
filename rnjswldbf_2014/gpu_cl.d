// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 rnjswldbf2014-hash
//
// OpenCL GPU 백엔드 — NVIDIA/AMD 둘 다 OpenCL ICD 하나로 커버한다 (CUDA 전용 경로는
// 따로 안 만든다: 이 라이브러리가 이미 배치=1 온라인 RL에 최적화돼 있어서 GPU 는 큰
// sl() 묶음에만 조건부로 붙는다 — ml.d 의 _gpuShouldUse 참고).
//
// OpenCL.dll 을 런타임에 동적 로드한다 (링크 의존성 아님 — OpenCL SDK 없이도
// 빌드/실행된다. GPU/드라이버가 없으면 그냥 조용히 비활성 상태로 남는다).
module gpucl;

version(Windows) {
    import core.sys.windows.windows : HMODULE, LoadLibraryA, GetProcAddress, FreeLibrary;
}

// ── OpenCL C API 타입/상수 (필요한 만큼만 손으로 선언) ──────────────────────
alias cl_int = int;
alias cl_uint = uint;
alias cl_platform_id = void*;
alias cl_device_id = void*;
alias cl_context = void*;
alias cl_command_queue = void*;
alias cl_mem = void*;
alias cl_program = void*;
alias cl_kernel = void*;
alias cl_device_type = ulong;
alias cl_mem_flags = ulong;
alias size_t_ = size_t;

enum cl_int CL_SUCCESS = 0;
enum cl_device_type CL_DEVICE_TYPE_GPU = 1 << 2;
enum cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
enum cl_mem_flags CL_MEM_READ_WRITE = 1 << 0;
enum cl_mem_flags CL_MEM_READ_ONLY  = 1 << 2;
enum cl_int CL_PROGRAM_BUILD_LOG = 0x1183;
enum cl_int CL_DEVICE_NAME = 0x102B;

extern(System) {
    alias FnGetPlatformIDs = cl_int function(cl_uint, cl_platform_id*, cl_uint*) nothrow;
    alias FnGetDeviceIDs = cl_int function(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) nothrow;
    alias FnGetDeviceInfo = cl_int function(cl_device_id, cl_uint, size_t_, void*, size_t_*) nothrow;
    alias FnCreateContext = cl_context function(const(void)*, cl_uint, const(cl_device_id)*,
        void*, void*, cl_int*) nothrow;
    alias FnCreateCommandQueue = cl_command_queue function(cl_context, cl_device_id, ulong, cl_int*) nothrow;
    alias FnCreateBuffer = cl_mem function(cl_context, cl_mem_flags, size_t_, void*, cl_int*) nothrow;
    alias FnEnqueueWriteBuffer = cl_int function(cl_command_queue, cl_mem, uint, size_t_, size_t_,
        const(void)*, cl_uint, const(void)*, void*) nothrow;
    alias FnEnqueueReadBuffer = cl_int function(cl_command_queue, cl_mem, uint, size_t_, size_t_,
        void*, cl_uint, const(void)*, void*) nothrow;
    alias FnCreateProgramWithSource = cl_program function(cl_context, cl_uint, const(char*)*,
        const(size_t_)*, cl_int*) nothrow;
    alias FnBuildProgram = cl_int function(cl_program, cl_uint, const(cl_device_id)*, const(char)*,
        void*, void*) nothrow;
    alias FnGetProgramBuildInfo = cl_int function(cl_program, cl_device_id, cl_uint, size_t_, void*, size_t_*) nothrow;
    alias FnCreateKernel = cl_kernel function(cl_program, const(char)*, cl_int*) nothrow;
    alias FnSetKernelArg = cl_int function(cl_kernel, cl_uint, size_t_, const(void)*) nothrow;
    alias FnEnqueueNDRangeKernel = cl_int function(cl_command_queue, cl_kernel, cl_uint,
        const(size_t_)*, const(size_t_)*, const(size_t_)*, cl_uint, const(void)*, void*) nothrow;
    alias FnFinish = cl_int function(cl_command_queue) nothrow;
    alias FnReleaseMemObject = cl_int function(cl_mem) nothrow;
    alias FnReleaseKernel = cl_int function(cl_kernel) nothrow;
    alias FnReleaseProgram = cl_int function(cl_program) nothrow;
    alias FnReleaseCommandQueue = cl_int function(cl_command_queue) nothrow;
    alias FnReleaseContext = cl_int function(cl_context) nothrow;
}

private __gshared FnGetPlatformIDs clGetPlatformIDs;
private __gshared FnGetDeviceIDs clGetDeviceIDs;
private __gshared FnGetDeviceInfo clGetDeviceInfo;
private __gshared FnCreateContext clCreateContext;
private __gshared FnCreateCommandQueue clCreateCommandQueue;
private __gshared FnCreateBuffer clCreateBuffer;
private __gshared FnEnqueueWriteBuffer clEnqueueWriteBuffer;
private __gshared FnEnqueueReadBuffer clEnqueueReadBuffer;
private __gshared FnCreateProgramWithSource clCreateProgramWithSource;
private __gshared FnBuildProgram clBuildProgram;
private __gshared FnGetProgramBuildInfo clGetProgramBuildInfo;
private __gshared FnCreateKernel clCreateKernel;
private __gshared FnSetKernelArg clSetKernelArg;
private __gshared FnEnqueueNDRangeKernel clEnqueueNDRangeKernel;
private __gshared FnFinish clFinish;
private __gshared FnReleaseMemObject clReleaseMemObject;
private __gshared FnReleaseKernel clReleaseKernel;
private __gshared FnReleaseProgram clReleaseProgram;
private __gshared FnReleaseCommandQueue clReleaseCommandQueue;
private __gshared FnReleaseContext clReleaseContext;

// 한 번 실패하면 그 프로세스에서는 다시 시도하지 않는다 (매 호출마다 DLL 을
// 다시 찾아보는 비용을 피한다).
private __gshared bool _triedInit = false;
private __gshared bool _available = false;
private __gshared cl_context _ctx;
private __gshared cl_command_queue _queue;
private __gshared cl_device_id _device;
private __gshared cl_program _prog;
private __gshared cl_kernel _kLinearFwd, _kRelu, _kReluBwd, _kGradW, _kDInput, _kAdamStep;

// ── 커널 소스 ────────────────────────────────────────────────────────────
private enum string KERNEL_SRC = `
__kernel void k_linear_forward(__global const float* w, __global const float* b,
                                __global const float* xs, __global float* pre,
                                int inSz, int outSz, int count) {
    int j = get_global_id(0);
    int c = get_global_id(1);
    if (j >= outSz || c >= count) return;
    __global const float* wj = w + (long)j * inSz;
    __global const float* xc = xs + (long)c * inSz;
    float acc = b[j];
    for (int k = 0; k < inSz; k++) acc += wj[k] * xc[k];
    pre[(long)c * outSz + j] = acc;
}

// in-place 아님 — pre 는 backward 의 ReLU 도함수 마스킹에 그대로 남아있어야 한다.
__kernel void k_relu(__global const float* pre, __global float* act, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = pre[i];
    act[i] = v < 0.0f ? 0.0f : v;
}

// dZ[i] = pre[i] > 0 ? dOut[i] : 0  (ReLU 도함수 마스킹)
__kernel void k_relu_backward(__global const float* pre, __global const float* dOut,
                               __global float* dZ, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    dZ[i] = pre[i] > 0.0f ? dOut[i] : 0.0f;
}

__kernel void k_linear_backward_gradW(__global const float* xs, __global const float* dZ,
                                       __global float* gradW, __global float* gradB,
                                       int inSz, int outSz, int count) {
    int j = get_global_id(0);
    if (j >= outSz) return;
    __global float* gj = gradW + (long)j * inSz;
    float gb = 0.0f;
    for (int c = 0; c < count; c++) {
        float d = dZ[(long)c * outSz + j];
        if (d == 0.0f) continue;
        __global const float* xc = xs + (long)c * inSz;
        for (int k = 0; k < inSz; k++) gj[k] += d * xc[k];
        gb += d;
    }
    gradB[j] += gb;
}

__kernel void k_linear_backward_dInput(__global const float* w, __global const float* dZ,
                                        __global float* dIn, int inSz, int outSz, int count) {
    int c = get_global_id(0);
    if (c >= count) return;
    __global float* o = dIn + (long)c * inSz;
    for (int k = 0; k < inSz; k++) o[k] = 0.0f;
    for (int j = 0; j < outSz; j++) {
        float d = dZ[(long)c * outSz + j];
        if (d == 0.0f) continue;
        __global const float* wj = w + (long)j * inSz;
        for (int k = 0; k < inSz; k++) o[k] += d * wj[k];
    }
}

// optKind: 0=adam 1=sgd 2=rmsprop 3=adagrad
__kernel void k_adam_step(__global float* w, __global const float* grad,
                           __global float* m, __global float* v,
                           float lr, float bc1, float bc2, int optKind, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    const float B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f, RHO = 0.99f;
    float g = grad[i];
    if (optKind == 0) {
        m[i] = B1*m[i] + (1.0f-B1)*g;
        v[i] = B2*v[i] + (1.0f-B2)*g*g;
        w[i] -= lr * (m[i]/bc1) / (sqrt(v[i]/bc2) + EPS);
    } else if (optKind == 1) {
        w[i] -= lr * g;
    } else if (optKind == 2) {
        v[i] = RHO*v[i] + (1.0f-RHO)*g*g;
        w[i] -= lr * g / (sqrt(v[i]) + EPS);
    } else {
        v[i] += g*g;
        w[i] -= lr * g / (sqrt(v[i]) + EPS);
    }
}
`;

private void* loadProc(void* dll, const(char)* name) nothrow {
    version(Windows) return GetProcAddress(cast(HMODULE) dll, name);
    else return null;
}

// 실패하면 false 반환 — 예외를 던지지 않는다 (nothrow 컨텍스트에서도 호출할 수
// 있도록: GPU 가 없는 게 정상적인 경우이지 오류가 아니다).
bool ensureInit() nothrow {
    if (_triedInit) return _available;
    _triedInit = true;
    try {
        version(Windows) {
            auto dll = LoadLibraryA("OpenCL.dll");
            if (dll is null) return false;

            clGetPlatformIDs = cast(FnGetPlatformIDs) loadProc(dll, "clGetPlatformIDs");
            clGetDeviceIDs = cast(FnGetDeviceIDs) loadProc(dll, "clGetDeviceIDs");
            clGetDeviceInfo = cast(FnGetDeviceInfo) loadProc(dll, "clGetDeviceInfo");
            clCreateContext = cast(FnCreateContext) loadProc(dll, "clCreateContext");
            clCreateCommandQueue = cast(FnCreateCommandQueue) loadProc(dll, "clCreateCommandQueue");
            clCreateBuffer = cast(FnCreateBuffer) loadProc(dll, "clCreateBuffer");
            clEnqueueWriteBuffer = cast(FnEnqueueWriteBuffer) loadProc(dll, "clEnqueueWriteBuffer");
            clEnqueueReadBuffer = cast(FnEnqueueReadBuffer) loadProc(dll, "clEnqueueReadBuffer");
            clCreateProgramWithSource = cast(FnCreateProgramWithSource) loadProc(dll, "clCreateProgramWithSource");
            clBuildProgram = cast(FnBuildProgram) loadProc(dll, "clBuildProgram");
            clGetProgramBuildInfo = cast(FnGetProgramBuildInfo) loadProc(dll, "clGetProgramBuildInfo");
            clCreateKernel = cast(FnCreateKernel) loadProc(dll, "clCreateKernel");
            clSetKernelArg = cast(FnSetKernelArg) loadProc(dll, "clSetKernelArg");
            clEnqueueNDRangeKernel = cast(FnEnqueueNDRangeKernel) loadProc(dll, "clEnqueueNDRangeKernel");
            clFinish = cast(FnFinish) loadProc(dll, "clFinish");
            clReleaseMemObject = cast(FnReleaseMemObject) loadProc(dll, "clReleaseMemObject");
            clReleaseKernel = cast(FnReleaseKernel) loadProc(dll, "clReleaseKernel");
            clReleaseProgram = cast(FnReleaseProgram) loadProc(dll, "clReleaseProgram");
            clReleaseCommandQueue = cast(FnReleaseCommandQueue) loadProc(dll, "clReleaseCommandQueue");
            clReleaseContext = cast(FnReleaseContext) loadProc(dll, "clReleaseContext");
        } else {
            return false; // Windows 외 플랫폼: 아직 미구현, 조용히 비활성
        }

        if (clGetPlatformIDs is null || clCreateContext is null) return false;

        cl_uint nPlat;
        if (clGetPlatformIDs(0, null, &nPlat) != CL_SUCCESS || nPlat == 0) return false;
        auto plats = new cl_platform_id[nPlat];
        clGetPlatformIDs(nPlat, plats.ptr, null);

        // GPU 타입 디바이스를 우선으로 찾는다 — 첫 플랫폼에서 못 찾으면 다음 플랫폼.
        foreach (p; plats) {
            cl_uint nDev;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, null, &nDev) == CL_SUCCESS && nDev > 0) {
                auto devs = new cl_device_id[nDev];
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nDev, devs.ptr, null);
                _device = devs[0];
                break;
            }
        }
        if (_device is null) return false;

        cl_int err;
        _ctx = clCreateContext(null, 1, &_device, null, null, &err);
        if (err != CL_SUCCESS || _ctx is null) return false;
        _queue = clCreateCommandQueue(_ctx, _device, 0, &err);
        if (err != CL_SUCCESS || _queue is null) return false;

        const(char)* src = KERNEL_SRC.ptr;
        size_t srcLen = KERNEL_SRC.length;
        _prog = clCreateProgramWithSource(_ctx, 1, &src, &srcLen, &err);
        if (err != CL_SUCCESS || _prog is null) return false;
        if (clBuildProgram(_prog, 1, &_device, null, null, null) != CL_SUCCESS) return false;

        _kLinearFwd = clCreateKernel(_prog, "k_linear_forward", &err);
        if (err != CL_SUCCESS) return false;
        _kRelu = clCreateKernel(_prog, "k_relu", &err);
        if (err != CL_SUCCESS) return false;
        _kReluBwd = clCreateKernel(_prog, "k_relu_backward", &err);
        if (err != CL_SUCCESS) return false;
        _kGradW = clCreateKernel(_prog, "k_linear_backward_gradW", &err);
        if (err != CL_SUCCESS) return false;
        _kDInput = clCreateKernel(_prog, "k_linear_backward_dInput", &err);
        if (err != CL_SUCCESS) return false;
        _kAdamStep = clCreateKernel(_prog, "k_adam_step", &err);
        if (err != CL_SUCCESS) return false;

        _available = true;
        return true;
    } catch (Throwable) {
        _available = false;
        return false;
    }
}

bool available() nothrow {
    return ensureInit();
}

// ── 버퍼 래퍼 ────────────────────────────────────────────────────────────
struct GpuBuf {
    cl_mem handle;
    size_t bytes;

    bool valid() const nothrow @nogc { return handle !is null; }
}

GpuBuf allocBuf(size_t nFloats) nothrow {
    GpuBuf r;
    if (!available) return r;
    try {
        cl_int err;
        r.bytes = nFloats * float.sizeof;
        r.handle = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, r.bytes, null, &err);
        if (err != CL_SUCCESS) r.handle = null;
    } catch (Throwable) { r.handle = null; }
    return r;
}

void freeBuf(ref GpuBuf b) nothrow {
    if (b.handle !is null) {
        try { clReleaseMemObject(b.handle); } catch (Throwable) {}
        b.handle = null;
    }
}

bool upload(GpuBuf b, const(float)[] src) nothrow {
    if (!b.valid) return false;
    try {
        return clEnqueueWriteBuffer(_queue, b.handle, 1, 0, src.length*float.sizeof,
            src.ptr, 0, null, null) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

// OpenCL 은 clCreateBuffer 로 새로 만든 버퍼의 내용을 0으로 보장하지 않는다 —
// gradW/gradB 처럼 커널이 "+=" 로 누적하는 버퍼는 반드시 먼저 이걸로 채워야 한다.
bool zeroBuf(GpuBuf b, size_t nFloats) nothrow {
    if (!b.valid) return false;
    try {
        auto zeros = new float[nFloats];
        zeros[] = 0f;   // D 는 new float[] 를 NaN 으로 채운다 — 반드시 직접 0 을 넣어야 한다.
        return upload(b, zeros);
    } catch (Throwable) { return false; }
}

bool download(GpuBuf b, float[] dst) nothrow {
    if (!b.valid) return false;
    try {
        return clEnqueueReadBuffer(_queue, b.handle, 1, 0, dst.length*float.sizeof,
            dst.ptr, 0, null, null) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

// ── 커널 실행 ────────────────────────────────────────────────────────────
private bool setArgs(cl_kernel k, void*[] args, size_t[] sizes) nothrow {
    foreach (i, a; args)
        if (clSetKernelArg(k, cast(cl_uint) i, sizes[i], a) != CL_SUCCESS) return false;
    return true;
}

bool linearForward(GpuBuf w, GpuBuf b, GpuBuf xs, GpuBuf pre, int inSz, int outSz, int count) nothrow {
    if (!available) return false;
    try {
        auto wa=w.handle; auto ba=b.handle; auto xa=xs.handle; auto pa=pre.handle;
        void*[7] args = [cast(void*)&wa, cast(void*)&ba, cast(void*)&xa, cast(void*)&pa,
                          cast(void*)&inSz, cast(void*)&outSz, cast(void*)&count];
        size_t[7] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,(void*).sizeof,
                            int.sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kLinearFwd, args[], sizes[])) return false;
        size_t[2] gws = [cast(size_t) outSz, cast(size_t) count];
        if (clEnqueueNDRangeKernel(_queue, _kLinearFwd, 2, null, gws.ptr, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

bool relu(GpuBuf pre, GpuBuf act, int n) nothrow {
    if (!available) return false;
    try {
        auto pa = pre.handle; auto aa = act.handle;
        void*[3] args = [cast(void*)&pa, cast(void*)&aa, cast(void*)&n];
        size_t[3] sizes = [(void*).sizeof, (void*).sizeof, int.sizeof];
        if (!setArgs(_kRelu, args[], sizes[])) return false;
        size_t gws = cast(size_t) n;
        if (clEnqueueNDRangeKernel(_queue, _kRelu, 1, null, &gws, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

bool reluBackward(GpuBuf pre, GpuBuf dOut, GpuBuf dZ, int n) nothrow {
    if (!available) return false;
    try {
        auto pa=pre.handle; auto doa=dOut.handle; auto dza=dZ.handle;
        void*[4] args = [cast(void*)&pa, cast(void*)&doa, cast(void*)&dza, cast(void*)&n];
        size_t[4] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,int.sizeof];
        if (!setArgs(_kReluBwd, args[], sizes[])) return false;
        size_t gws = cast(size_t) n;
        if (clEnqueueNDRangeKernel(_queue, _kReluBwd, 1, null, &gws, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

bool linearBackwardGradW(GpuBuf xs, GpuBuf dZ, GpuBuf gradW, GpuBuf gradB,
                          int inSz, int outSz, int count) nothrow {
    if (!available) return false;
    try {
        auto xa=xs.handle; auto da=dZ.handle; auto gwa=gradW.handle; auto gba=gradB.handle;
        void*[7] args = [cast(void*)&xa, cast(void*)&da, cast(void*)&gwa, cast(void*)&gba,
                          cast(void*)&inSz, cast(void*)&outSz, cast(void*)&count];
        size_t[7] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,(void*).sizeof,
                            int.sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kGradW, args[], sizes[])) return false;
        size_t gws = cast(size_t) outSz;
        if (clEnqueueNDRangeKernel(_queue, _kGradW, 1, null, &gws, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

bool linearBackwardDInput(GpuBuf w, GpuBuf dZ, GpuBuf dIn, int inSz, int outSz, int count) nothrow {
    if (!available) return false;
    try {
        auto wa=w.handle; auto da=dZ.handle; auto dia=dIn.handle;
        void*[6] args = [cast(void*)&wa, cast(void*)&da, cast(void*)&dia,
                          cast(void*)&inSz, cast(void*)&outSz, cast(void*)&count];
        size_t[6] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,
                            int.sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kDInput, args[], sizes[])) return false;
        size_t gws = cast(size_t) count;
        if (clEnqueueNDRangeKernel(_queue, _kDInput, 1, null, &gws, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

bool adamStep(GpuBuf w, GpuBuf grad, GpuBuf m, GpuBuf v, float lr, float bc1, float bc2,
              int optKind, int n) nothrow {
    if (!available) return false;
    try {
        auto wa=w.handle; auto ga=grad.handle; auto ma=m.handle; auto va=v.handle;
        void*[9] args = [cast(void*)&wa, cast(void*)&ga, cast(void*)&ma, cast(void*)&va,
                          cast(void*)&lr, cast(void*)&bc1, cast(void*)&bc2,
                          cast(void*)&optKind, cast(void*)&n];
        size_t[9] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,(void*).sizeof,
                            float.sizeof,float.sizeof,float.sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kAdamStep, args[], sizes[])) return false;
        size_t gws = cast(size_t) n;
        if (clEnqueueNDRangeKernel(_queue, _kAdamStep, 1, null, &gws, null, 0, null, null) != CL_SUCCESS)
            return false;
        return clFinish(_queue) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}
