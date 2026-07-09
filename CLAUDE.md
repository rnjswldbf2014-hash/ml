# dplayg

D언어로 구현한 강화학습/지도학습 라이브러리. Python extension module (.pyd)로 빌드해 `import my_ml`만으로 직접 사용 가능.
GitHub 원본(PyTorch 버전)과 동일한 API를 유지하면서 D로 성능 최적화.

## 파일 구조

```
my_ml.d      D 소스 (신경망 + Python C API 통합)
my_ml.pyd    Python extension module (정적 링크, 외부 DLL 불필요)
ldc2/        LDC 컴파일러 (ldc2-1.42.0-windows-x64)
CLAUDE.md    이 문서
```

## 빌드

```powershell
$ldc   = "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"

Start-Process -FilePath $ldc -Wait -NoNewWindow -ArgumentList @(
    "my_ml.d", $pylib,
    "--O3", "--release", "--shared", "--link-defaultlib-shared=false",
    "-of=my_ml.pyd"
)
```

- `--O3` : 최고 최적화
- `--release` : bounds check 제거
- `--link-defaultlib-shared=false` : D 런타임 정적 링크 → 외부 DLL 불필요

## CPU 디스패치

런타임에 CPU 기능을 감지해 최적 코드 경로를 선택한다.

| 대상 | 조건 |
|------|------|
| `dot_avx2` / `saxpy_avx2` | AVX2 + FMA 지원 CPU |
| `dot_sse41` / `saxpy_sse41` | SSE4.1 지원 CPU |
| `dot_base` / `saxpy_base` | 베이스라인 x86-64 |

`shared static this()`에서 `core.cpuid`로 감지 → 함수 포인터 교체.
핫 경로: `Linear.forward` (dot product), `Linear.accum` (SAXPY).

## Python API

GitHub 원본(PyTorch 버전)과 완전히 동일.

```python
import my_ml

# 생성
ai = my_ml.make("ModelName", ["a","b","c"], hidden_layers=[128,64], optimizer='adam')
# optimizer: 'adam' | 'sgd' | 'rmsprop' | 'adagrad'

# 다중 헤드
ai = my_ml.make("ModelName", [["a","b"], ["x","y"]])

# 강화학습
action = ai.rl(["a","b"], *env)              # 단일 헤드 → str
actions = ai.rl([["a","b"],["x","y"]], *env) # 다중 헤드 → list
ai.reward(1.0)

# 에피소드
with ai.episode() as e:
    e.rl(["a","b"], *env)
ai.last_reward(1.0)

# 지도학습
ai.sl(["a","b"], *env, "a")                     # 단일 헤드
ai.sl([["a","b"],["x","y"]], *env, ["a","x"])   # 다중 헤드

# 저장/초기화
ai.save()
resset("ModelName")   # 전역 빌트인, my_ml. 없이 호출 가능
reset("ModelName")    # resset과 동일
```

## 저장 포맷

`{name}_ml_memory.pth` (바이너리)

- magic: `0xBEEFCAFE`
- version: `2` (v1은 optimizer 필드 없음, 로드 시 adam으로 처리)
- optimizer, inputSz, hiddenSizes, actionLists, 가중치 순서로 저장

## 구조 개요

```
my_ml.d
├── DllMain            Windows DLL 진입점 (D 런타임 초기화)
├── CPU dispatch       dot/saxpy: AVX2+FMA / SSE4.1 / baseline
├── Opt enum           adam | sgd | rmsprop | adagrad
├── Linear struct      FC 레이어 (forward, backward, 4옵티마이저 step)
├── Network class      hidden layers + multiple output heads
├── BlackBoxAI class   RL/SL 인터페이스, 저장/로드
├── Python C API       extern(C) 선언 (python313.lib 링크)
├── py_* functions     Python ↔ D 변환 + BlackBoxAI 호출
├── PY_CLASS_CODE      모듈 import 시 실행되는 embedded Python 문자열
└── PyInit_my_ml       Python extension 진입점
```
