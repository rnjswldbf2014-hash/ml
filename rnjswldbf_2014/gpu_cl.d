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
private __gshared string _deviceName = "";
private __gshared cl_context _ctx;
private __gshared cl_command_queue _queue;
private __gshared cl_device_id _device;
private __gshared cl_program _prog;
private __gshared cl_kernel _kLinearFwd, _kRelu, _kReluBwd, _kGradW, _kGradB, _kDInput,
                             _kAdamStep, _kFillZero;

// ── 커널 소스 ────────────────────────────────────────────────────────────
private enum string KERNEL_SRC = `
// 세 개의 행렬곱 커널은 전부 같은 구조다. work-group 하나가 TS x TS 출력 타일을
// 맡고, 줄어드는 축을 TS 씩 끊어 로컬 메모리로 올린 뒤 곱한다. 다만 스레드를
// TS x TS 개 띄우지 않고 TS x RTS 개만 띄워서, 스레드 하나가 WPT 개 출력을 맡는다
// (레지스터 블로킹). 로컬 메모리에서 읽은 값 하나를 WPT 번 재사용하므로 로컬
// 메모리 대역폭이 병목에서 빠진다.
//
// 지켜야 하는 세 가지:
//  ① 전역 메모리 읽기는 get_local_id(0) 이 "붙어있는" 축을 훑게 한다. 이게 어긋나면
//     lane 마다 다른 캐시라인을 건드려서 대역폭이 수십분의 1로 떨어진다.
//     (맨 처음 커널이 정확히 이랬다 — 이론 성능의 0.2% 밖에 못 냈다.)
//     dZ 는 [count][outSz] 라 j 가, xs/w 는 k 가 붙어있다. 그래서 커널마다 어느
//     인덱스를 lane 에 태우는지가 다르다 — 베껴 쓸 때 제일 틀리기 쉬운 부분.
//  ② 로컬 배열은 [TS][TS+1] 로 한 칸 패딩한다. 패딩이 없으면 열 방향 접근이
//     전부 같은 뱅크로 몰린다.
//  ③ 누적 루프의 로컬 읽기 중 하나는 lane 간 broadcast, 다른 하나는 연속이어야
//     한다. 둘 다 흩어지면 패딩을 해도 느리다.
#define TS  32
#define WPT 4
#define RTS (TS/WPT)

// pre[c][j] = b[j] + sum_k wt[j][k] * xs[c][k]
__kernel void k_linear_forward(__global const float* wt, __global const float* bs,
                                __global const float* xs, __global float* pre,
                                int inSz, int outSz, int count) {
    int lj = get_local_id(0);          // 0..TS-1   -> j (출력에서 붙어있는 축)
    int lc = get_local_id(1);          // 0..RTS-1  -> c (WPT 개를 맡는다)
    int j0 = get_group_id(0) * TS, c0 = get_group_id(1) * TS;
    int j = j0 + lj;

    __local float Ax[TS][TS+1];   // Ax[a][b] = xs[c0+a][k0+b]
    __local float Bw[TS][TS+1];   // Bw[a][b] = wt[j0+a][k0+b]

    float acc[WPT];
    for (int q = 0; q < WPT; q++) acc[q] = 0.0f;

    for (int k0 = 0; k0 < inSz; k0 += TS) {
        int k = k0 + lj;                              // lj 가 k 를 훑는다 -> 병합
        for (int q = 0; q < WPT; q++) {
            int r = lc + q*RTS;
            Ax[r][lj] = (c0+r < count && k < inSz) ? xs[(long)(c0+r) * inSz + k] : 0.0f;
            Bw[r][lj] = (j0+r < outSz && k < inSz) ? wt[(long)(j0+r) * inSz + k] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TS; t++) {
            float bv = Bw[lj][t];                     // lane 마다 다름 (연속)
            for (int q = 0; q < WPT; q++)
                acc[q] += Ax[lc + q*RTS][t] * bv;     // lane 간 broadcast
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j >= outSz) return;
    float bias = bs[j];
    for (int q = 0; q < WPT; q++) {
        int c = c0 + lc + q*RTS;
        if (c < count) pre[(long)c * outSz + j] = bias + acc[q];
    }
}

__kernel void k_fill_zero(__global float* dst, int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] = 0.0f;
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

// gradW[j][k] += sum_c dZ[c][j] * xs[c][k]
// 예전엔 work-item 을 outSz 개만 띄우고 그 안에서 c 와 k 를 전부 돌았다. 출력이
// outSz*inSz 개인데 일꾼이 outSz 명이었던 셈이다.
__kernel void k_linear_backward_gradW(__global const float* xs, __global const float* dZ,
                                       __global float* gradW,
                                       int inSz, int outSz, int count) {
    int lk = get_local_id(0);          // 0..TS-1   -> k (출력에서 붙어있는 축)
    int lj = get_local_id(1);          // 0..RTS-1  -> j (WPT 개를 맡는다)
    int k0 = get_group_id(0) * TS, j0 = get_group_id(1) * TS;
    int k = k0 + lk;

    __local float Ad[TS][TS+1];   // Ad[a][b] = dZ[c0+b][j0+a]   (j 가 앞)
    __local float Bx[TS][TS+1];   // Bx[a][b] = xs[c0+a][k0+b]

    float acc[WPT];
    for (int q = 0; q < WPT; q++) acc[q] = 0.0f;

    for (int c0 = 0; c0 < count; c0 += TS) {
        // 타일을 채울 때 lj 가 맡는 축은 c 다 (출력에서 맡는 j 와는 다른 축이다 —
        // 두 축 모두 RTS 간격으로 TS 개를 덮으므로 식만 같아 보인다. 헷갈리기 쉬움).
        for (int q = 0; q < WPT; q++) {
            int cl = lj + q*RTS;                      // 타일 안에서의 c
            int c  = c0 + cl;
            // dZ 는 [count][outSz] 라 j 가 붙어있다 -> lk 로 j 를 훑어야 병합된다
            Ad[lk][cl] = (c < count && j0+lk < outSz) ? dZ[(long)c * outSz + j0+lk] : 0.0f;
            Bx[cl][lk] = (c < count && k      < inSz) ? xs[(long)c * inSz  + k    ] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TS; t++) {               // t = 타일 안에서의 c
            float bv = Bx[t][lk];                     // lane 마다 다름 (연속)
            for (int q = 0; q < WPT; q++)
                acc[q] += Ad[lj + q*RTS][t] * bv;     // lane 간 broadcast (여기선 j)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (k >= inSz) return;
    for (int q = 0; q < WPT; q++) {
        int j = j0 + lj + q*RTS;
        if (j < outSz) gradW[(long)j * inSz + k] += acc[q];
    }
}

// gradB[j] += sum_c dZ[c][j].  일이 outSz*count 뿐이라 타일링할 게 없다.
__kernel void k_linear_backward_gradB(__global const float* dZ, __global float* gradB,
                                       int outSz, int count) {
    int j = get_global_id(0);
    if (j >= outSz) return;
    float gb = 0.0f;
    for (int c = 0; c < count; c++) gb += dZ[(long)c * outSz + j];
    gradB[j] += gb;
}

// dIn[c][k] = sum_j dZ[c][j] * w[j][k]
// dZ 도 w 도 줄어드는 축(j)이 각각 뒤/앞에 자연스럽게 놓여 있어 셋 중 제일 단순하다.
__kernel void k_linear_backward_dInput(__global const float* wt, __global const float* dZ,
                                        __global float* dIn, int inSz, int outSz, int count) {
    int lk = get_local_id(0);          // 0..TS-1   -> k (출력에서 붙어있는 축)
    int lc = get_local_id(1);          // 0..RTS-1  -> c (WPT 개를 맡는다)
    int k0 = get_group_id(0) * TS, c0 = get_group_id(1) * TS;
    int k = k0 + lk;

    __local float Ad[TS][TS+1];   // Ad[a][b] = dZ[c0+a][j0+b]
    __local float Bw[TS][TS+1];   // Bw[a][b] = wt[j0+a][k0+b]

    float acc[WPT];
    for (int q = 0; q < WPT; q++) acc[q] = 0.0f;

    for (int j0 = 0; j0 < outSz; j0 += TS) {
        int j = j0 + lk;                              // Ad 를 채울 땐 lk 가 j 를 훑는다
        for (int q = 0; q < WPT; q++) {
            int r = lc + q*RTS;                       // Ad 에선 c, Bw 에선 j
            Ad[r][lk] = (c0+r < count && j    < outSz) ? dZ[(long)(c0+r) * outSz + j] : 0.0f;
            Bw[r][lk] = (j0+r < outSz && k    < inSz)  ? wt[(long)(j0+r) * inSz  + k] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TS; t++) {               // t = 타일 안에서의 j
            float bv = Bw[t][lk];                     // lane 마다 다름 (연속)
            for (int q = 0; q < WPT; q++)
                acc[q] += Ad[lc + q*RTS][t] * bv;     // lane 간 broadcast
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // 예전 커널은 0으로 채운 뒤 누적했다 — 결국 대입과 같다.
    if (k >= inSz) return;
    for (int q = 0; q < WPT; q++) {
        int c = c0 + lc + q*RTS;
        if (c < count) dIn[(long)c * inSz + k] = acc[q];
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
        _kGradB = clCreateKernel(_prog, "k_linear_backward_gradB", &err);
        if (err != CL_SUCCESS) return false;
        _kDInput = clCreateKernel(_prog, "k_linear_backward_dInput", &err);
        if (err != CL_SUCCESS) return false;
        _kAdamStep = clCreateKernel(_prog, "k_adam_step", &err);
        if (err != CL_SUCCESS) return false;
        _kFillZero = clCreateKernel(_prog, "k_fill_zero", &err);
        if (err != CL_SUCCESS) return false;

        // 어느 장치를 잡았는지 기록해둔다 — GPU 가 실제로 쓰이고 있는지 밖에서
        // 확인할 방법이 없으면 "켜져 있다고 믿는데 사실은 CPU" 를 못 잡아낸다.
        try {
            char[256] nameBuf;
            size_t got;
            if (clGetDeviceInfo(_device, CL_DEVICE_NAME, nameBuf.length,
                                nameBuf.ptr, &got) == CL_SUCCESS && got > 0) {
                if (got > nameBuf.length) got = nameBuf.length;
                while (got > 0 && nameBuf[got-1] == '\0') got--;
                _deviceName = nameBuf[0..got].idup;
            }
        } catch (Throwable) {}

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

// 잡은 GPU 이름. 아직 프로브 안 했거나 GPU 가 없으면 빈 문자열.
string deviceName() nothrow {
    return _deviceName;
}

// GPU 경로가 실제로 끝까지 돈 횟수. 중간에 실패해서 CPU 로 폴백한 건 안 센다.
__gshared long runCount = 0;

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
//
// GPU 에서 직접 채운다. 예전엔 호스트에 0 배열을 만들어 업로드했는데, 큰 층이면
// 호출마다 수 MB 를 할당하고 전송하는 셈이라 그 자체가 병목이었다.
bool zeroBuf(GpuBuf b, size_t nFloats) nothrow {
    if (!b.valid || !available) return false;
    try {
        int n = cast(int) nFloats;
        auto da = b.handle;
        void*[2] args = [cast(void*)&da, cast(void*)&n];
        size_t[2] sizes = [(void*).sizeof, int.sizeof];
        if (!setArgs(_kFillZero, args[], sizes[])) return false;
        size_t gws = nFloats;
        return clEnqueueNDRangeKernel(_queue, _kFillZero, 1, null, &gws, null,
                                      0, null, null) == CL_SUCCESS;
    } catch (Throwable) { return false; }
}

// 큐에 쌓인 작업이 다 끝날 때까지 기다린다. 블로킹 download 가 이미 같은 일을
// 하므로 보통은 쓸 일이 없다 — 커널 오류를 그 자리에서 보고 싶을 때만.
bool finish() nothrow {
    if (!available) return false;
    try { return clFinish(_queue) == CL_SUCCESS; } catch (Throwable) { return false; }
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

// 타일링 커널 공통 실행. 이 세 값은 커널 소스의 TS/WPT/RTS 와 반드시 같아야 한다
// — 로컬 배열 크기와 스레드 배치가 그 전제로 짜여 있어서, 어긋나면 오류도 없이
// 조용히 틀린 값이 나온다.
private enum int TS  = 32;
private enum int WPT = 4;
private enum int RTS = TS / WPT;

// 출력 타일은 TS x TS 지만 스레드는 TS x RTS 개만 띄운다 (스레드 하나가 WPT 개
// 출력을 맡는다). 그래서 dim1 쪽 global size 는 타일 수 x RTS 다.
// 남는 work-item 은 커널 안의 범위 검사에서 걸러지고, 로컬 타일의 빈 자리에는
// 0 이 들어가므로 합에 영향이 없다.
private bool runTiled(cl_kernel k, int dim0, int dim1) nothrow {
    try {
        size_t[2] lws = [TS, RTS];
        size_t[2] gws = [cast(size_t)((dim0 + TS - 1) / TS) * TS,
                          cast(size_t)((dim1 + TS - 1) / TS) * RTS];
        return clEnqueueNDRangeKernel(_queue, k, 2, null, gws.ptr, lws.ptr,
                                      0, null, null) == CL_SUCCESS;
    } catch (Throwable) { return false; }
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
        return runTiled(_kLinearFwd, outSz, count);
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
        return true;   // clFinish 안 한다 — 큐가 in-order 라 다음 작업이 알아서 뒤에 선다
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
        return true;   // clFinish 안 한다 — 큐가 in-order 라 다음 작업이 알아서 뒤에 선다
    } catch (Throwable) { return false; }
}

bool linearBackwardGradW(GpuBuf xs, GpuBuf dZ, GpuBuf gradW, GpuBuf gradB,
                          int inSz, int outSz, int count) nothrow {
    if (!available) return false;
    try {
        auto xa=xs.handle; auto da=dZ.handle; auto gwa=gradW.handle; auto gba=gradB.handle;
        void*[6] args = [cast(void*)&xa, cast(void*)&da, cast(void*)&gwa,
                          cast(void*)&inSz, cast(void*)&outSz, cast(void*)&count];
        size_t[6] sizes = [(void*).sizeof,(void*).sizeof,(void*).sizeof,
                            int.sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kGradW, args[], sizes[])) return false;
        if (!runTiled(_kGradW, inSz, outSz)) return false;

        // gradB 는 따로 — 일이 outSz*count 뿐이라 타일링할 게 없다
        void*[4] bargs = [cast(void*)&da, cast(void*)&gba,
                           cast(void*)&outSz, cast(void*)&count];
        size_t[4] bsizes = [(void*).sizeof,(void*).sizeof,int.sizeof,int.sizeof];
        if (!setArgs(_kGradB, bargs[], bsizes[])) return false;
        size_t gws = cast(size_t) outSz;
        return clEnqueueNDRangeKernel(_queue, _kGradB, 1, null, &gws, null,
                                      0, null, null) == CL_SUCCESS;
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
        return runTiled(_kDInput, inSz, count);
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
        return true;   // clFinish 안 한다 — 큐가 in-order 라 다음 작업이 알아서 뒤에 선다
    } catch (Throwable) { return false; }
}
