# dplayg

D언어로 구현한 RL/SL 라이브러리. `ml.d` → `ml.pyd` (.pyd = Python extension, `import ml`로 사용).
(모듈 진입점이 `PyInit_ml` 이라 파일명·import 이름 모두 `ml` 이다. `import my_ml` 은 안 된다 —
`__name__` 만 "my_ml" 로 나오는데 PyModuleDef 의 m_name 이 그렇게 남아있어서다.)
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
- `main.py` — jepa 월드모델 예제 (`python main.py` 로 바로 돌아간다).
  잡음 섞인 관측만 보고 위치를 배우고, 예측기를 여러 번 이어 붙여 머릿속으로
  굴려보며 계획을 세운다.
- `tests/` — 결정성 회귀 하네스 (`python tests/regression.py`)
  와 jepa 검증 (`python tests/jepa.py` — 학습 여부·붕괴 방지·스레드 결정성)

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
from ml import make, cos, attn, each

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

### jepa (요약 예측 / 월드모델)

원본을 통째로 맞추는 대신 **요약**만 맞춘다. 관측에 예측 불가능한 잡음이 섞여 있어도
요약 단계에서 버려지므로 거기에 힘을 안 뺀다 (LeCun 이 제안한 JEPA).

```python
from ml import make, vec, jepa

enc  = make("enc",  [입력수, 128], [vec(32)])        # 요약기
pred = make("pred", [32 + 행동수, 128], [vec(32)])   # 예측기
w = jepa(enc, pred)

손실 = w.train(지금들, 다음들, 행동들)   # 묶음으로 준다. 32개 이상 권장
w.encode(관측)                           # 관측 -> 요약 (숫자 리스트)
w.imagine(관측, 행동)                    # 다음 요약 예측 = "미리 상상해보기"
w.save()                                 # 요약기·예측기 둘 다 저장
```

- `vec(n)` = 숫자 n개를 한 덩어리로 내는 출력. 값은 `embed(입력, 번호)` 로 꺼낸다
  (`predict()`/`rl()` 은 한 칸짜리라 vec 자리에는 `None` 이 나온다).
- 예측기 입력수 − 요약 크기 = **행동 입력 개수**로 자동 계산된다.
  같게 두면 행동 없는 형태(그냥 자기지도학습)가 된다.
- `x`=지금 / `y`=다음 으로 주면 월드모델, 같은 것의 두 조각으로 주면 표현학습.
  라이브러리 입장에선 똑같고 뭘 넣을지는 쓰는 쪽 마음이다.
- `jepa(..., var=25, cov=1)` 이 **붕괴(collapse) 방지** 힘이다. 이게 없으면
  요약기가 "입력이 뭐든 늘 같은 값" 을 뱉어서 손실 0 으로 만점을 받아버린다
  (실측: 끄면 요약 퍼짐 0.96 → 0.0006, 손실 0.000005). 0 으로 두지 말 것.
- 이 붕괴 방지 계산만 **배치 전체를 한꺼번에** 봐야 한다 ("이 묶음 안에서 서로
  다른가" 가 질문이라서). 그래서 묶음이 2개 미만이면 아예 꺼진다.

**기존 rl/sl 과 섞어 쓰기.** 요약기에 출력을 더 붙이면 신경망 몸통 하나를
jepa 와 일반 학습이 같이 쓴다. 첫 출력만 `vec` 이면 된다.

```python
enc = make("enc", [입력수, 128], [vec(32), ["왼쪽","오른쪽"]])
w = jepa(enc, make("pred", [32 + 2, 128], [vec(32)]))

w.train(지금들, 다음들, 행동들)             # 요약을 다듬는다 (라벨 필요 없음)
enc.sl(관측들, [[None, "왼쪽"], ...])       # 행동을 가르친다
enc.predict(관측)   # -> [None, "왼쪽"]     vec 자리는 None
enc.embed(관측, 0)  # -> 숫자 32개          요약은 여기서 꺼낸다
```

vec 자리는 `sl()` 정답과 `reward()` 점수에서 `None` 으로 비워야 한다 (숫자 하나로는
다룰 수 없어서 값을 주면 막힌다). 또는 그냥 `w.encode()` 결과를 별개 모델의 입력으로
넘겨도 된다 — 둘 다 된다.

효과는 문제에 따라 다르다. 자체 실험(6칸 고리 세계, 잡음 큰 관측)에서는 라벨이 아주
적을 때만 조금 나았고(라벨 24개: 74% → 79%), 라벨이 충분하면 차이가 없거나 오히려
살짝 손해였다. 켜면 무조건 좋아지는 게 아니라 재봐야 하는 것으로 취급할 것.

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
- `jepa` 의 요약값은 그 자체로는 뜻이 없다 — 학습할 때마다 다른 좌표계가 나온다.
  "요약끼리의 거리" 로만 쓴다. 요약 하나만 보고 뭔가를 읽어내려 하면 안 된다.
