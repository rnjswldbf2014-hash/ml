# dplayg

D언어로 구현한 RL/SL 라이브러리. `my_ml.d` → `my_ml.pyd` (.pyd = Python extension, `import my_ml`로 사용).
가중치는 `이름_ml_memory.pth` 로 저장한다 (자체 포맷, 현재 ver 6).

## 빌드

```powershell
$ldc   = "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"
& $ldc rnjswldbf_2014\ml.d rnjswldbf_2014\gpu_cl.d $pylib --O3 --release --shared --link-defaultlib-shared=false "-of=rnjswldbf_2014\ml.pyd"
Remove-Item rnjswldbf_2014\ml.obj, rnjswldbf_2014\ml.lib, rnjswldbf_2014\ml.exp, rnjswldbf_2014\gpu_cl.obj -ErrorAction SilentlyContinue
```

(`tests\build.ps1 -OutDir <dir>` 가 이 커맨드를 감싸놓은 스크립트 — 테스트 빌드는 그걸 쓰면 됨.)

`"-of=..."` 의 따옴표 필수. 빼면 PowerShell 이 인자를 쪼개서
`Error: unrecognized file extension pyd` 로 빌드가 실패한다.

## 파일

- `rnjswldbf_2014/ml.d` — D 소스 (신경망 + Python C API + Python 클래스 코드 인라인)
- `rnjswldbf_2014/gpu_cl.d` — OpenCL GPU 백엔드 (런타임에 OpenCL.dll 동적 로드, 없으면 자동으로 CPU 로 폴백)
- `rnjswldbf_2014/ml.pyd` — 빌드 결과물 (gitignore)
- `main.py` — 빈 파일. 사용자가 여기에 작성한다.
- `tests/` — 결정성 회귀 하네스 (`python tests/regression.py`)

## 환경변수

- `MYML_THREADS` — CPU 스레드 수 (기본: 전체 코어)
- `MYML_NOBATCH=1` — 배치 경로 끄고 per-sample 직렬 경로로 (동치 검증용)
- `MYML_GPU` — `0`=완전 비활성, `1`=강제, 미설정="auto"(문턱값 넘는 큰 배치만)
- `MYML_GPU_MIN_FLOPS`, `MYML_GPU_MIN_B` — auto 모드 문턱값 (기본 5e7, 64)

GPU 경로는 현재 순수 Linear 망 + 헤드 1개 + `cos` 출력만 지원한다 (attn/each,
다중 헤드, pick 헤드는 CPU 배치 경로로 자동 폴백). NVIDIA/AMD 둘 다 OpenCL 하나로
커버— CUDA 전용 경로는 없음 (배치=1 온라인 학습이 핵심 사용처라 GPU 는 큰 sl()
묶음에만 조건부로 개입한다).

## API

```python
from my_ml import make, cos, attn, each

# layers 는 [입력수, 은닉...] 만. 출력 개수는 outputs 에서 정해진다.
# outputs 는 항상 리스트. 하나여도 감싼다. 반환·보상·정답·legal 도 전부 리스트.
ai = make("이름", [입력수, 은닉...], [["액션A", "액션B"]])   # 고르기 1개
ai = make("이름", [입력수, 은닉...], [cos])                  # 숫자 1개
ai = make("이름", [입력수, 은닉...], [cos, ["a","b"]])       # 출력 2개

step   = ai.rl([입력...])          # 순수 → Step(input, output)
scored = ai.reward(step, 점수)     # 순수 → Scored(input, output, point)
ai.save(scored)                    # 역전파 + 파일 자동 저장 (여기서만 모델이 바뀜)
ai.save([scored, ...])             # 묶음도 가능

ai.predict([입력...])              # 샘플링/학습 없이 최선값 (리스트)
ai.sl([입력...], ["정답"])         # 지도학습 1스텝 후 예측
ai.episode(steps, 점수)            # [Step...] → [Scored...] 일괄 보상 (순수)

change("이름")                     # 예전 포맷 가중치 → 현재 포맷 (.bak 백업)

# attn: 항목끼리 서로 참조 (폭은 그대로).  each: 항목마다 따로 가공 (가중치 공유)
# 일반 층을 사이에 끼우면 항목 구분이 사라진다 -> attn 과 each 를 번갈아 쓴다
ai = make("이름", [12, attn(6), each(24), attn(6), each(24), 128], 출력)

# 묶음 학습: 기울기를 모았다가 갱신 1번. 하나씩 부르는 것보다 10배 이상 빠르다.
ai.sl([입력1, 입력2, ...], [정답1, 정답2, ...])
```

`output`, `reward` 의 점수, `sl` 의 정답, `legal` 은 **항상** 출력 개수만큼의 리스트다
(출력이 하나여도). 점수/정답에 `None` 을 주면 그 출력은 학습에서 빠진다.

`rl()`/`reward()`/`episode()` 는 모델을 건드리지 않는다. 학습은 `save()` 와 `sl()` 뿐이다.

`make()` 의 `sigma`(cos 탐험 폭, 기본 1.0), `entropy`(고르기가 한 답으로 굳는 것을
막는 힘, 기본 0.01) 로 학습 성향을 조절한다.

## 주의

- `cos` 값은 0 근처에서 시작한다. 원하는 범위가 있으면 쓰는 쪽에서 펼쳐 쓴다.
  예) `파워 = 20 + 출력 * 10`
- 번호(글자, 종류 같은 것)를 입력으로 줄 때는 one-hot 으로 바꿔서 넣는다.
  그대로 넣으면 신경망이 크기로 해석한다 (55% vs 100% 사례).
- attn 만 넣으면 효과가 거의 없다. each 와 같이 써야 한다 (87% -> 99% 사례).
- `sl()` 을 하나씩 부르면 문제마다 가중치 전체를 갱신해서 매우 느리다. 묶음으로 준다.
- 드문 행동을 지도학습시킬 때는 여러 번 반복해야 한다.
  안 그러면 흔한 행동만 답하는 쪽으로 굳는다.
  (전체의 4% 인 행동은 "안 한다" 고만 답해도 96점이라 그쪽으로 수렴한다)
