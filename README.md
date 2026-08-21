# dplayg

D 언어로 구현한 강화학습 / 지도학습 라이브러리입니다.
`my_ml.d` 를 빌드하면 `my_ml.pyd` 가 나오고, 파이썬에서 `import my_ml` 로 씁니다.

---

## 설계 원칙

이 라이브러리의 함수는 **데이터를 만드는 함수** 와 **모델을 바꾸는 함수** 로 나뉩니다.

| 함수 | 순수? | 하는 일 |
|---|---|---|
| `rl()` | 순수 | `(입력, 출력)` 데이터만 만듭니다. 가중치를 건드리지 않습니다. |
| `reward()` | 순수 | `(입력, 출력, 보상)` 데이터만 만듭니다. |
| `episode()` | 순수 | 여러 스텝에 같은 보상을 매깁니다. |
| `predict()` | 순수 | 샘플링 없이 최선의 값만 봅니다. |
| **`save()`** | **아님** | **역전파 + 파일 저장. 모델이 바뀌는 유일한 곳입니다.** |

`rl()` 을 몇 번을 부르든 모델은 그대로입니다. 학습은 `save()` 를 부를 때만 일어납니다.

---

## 모델 만들기

```python
from my_ml import make

from my_ml import make, cos

ai = make("MyModel", [12, 128, 128], ["왼쪽", "오른쪽", "정지"])
```

| 인자 | 설명 |
|---|---|
| 1번째 | 모델 이름. 가중치 파일명(`이름_ml_memory.pth`)에 씁니다. |
| 2번째 | 레이어 구조 `[입력수, 은닉...]`. 은닉층은 몇 개든 됩니다. 출력은 여기 안 씁니다. `attn(n)` 을 끼우면 어텐션 층이 됩니다. |
| 3번째 | 출력 스펙. 고를 이름 목록 또는 `cos`. 여러 개면 리스트로 묶습니다. |
| `optimizer` | `'adam'`(기본) / `'sgd'` / `'rmsprop'` / `'adagrad'` |
| `sigma` | cos 가 값을 얼마나 넓게 탐험할지 (기본 1.0) |
| `entropy` | 고르는 쪽이 한 가지 답으로 굳는 것을 막는 힘 (기본 0.01) |

같은 이름의 파일이 있으면 이어서 학습합니다.
단, 저장된 구조가 요청한 `layers` 와 다르면 파일을 무시하고 새로 만듭니다.

---

## 강화학습

```python
step   = ai.rl([0.1, 0.2, ...])        # → Step(input, output)
scored = ai.reward(step, +1.0)         # → Scored(input, output, point)
ai.save(scored)                        # 학습 + 저장
```

`step.input` / `step.output` 으로 꺼내 쓰고, 언패킹도 됩니다.

```python
입력, 출력 = ai.rl([0.5])
```

### 묶어서 학습 (권장)

한 번에 여러 스텝을 넘기면 그만큼 빨라집니다.

```python
batch = []
for i in range(32):
    step = ai.rl([i / 32])
    point = +1.0 if step.output == "왼쪽" else -1.0
    batch.append(ai.reward(step, point))
ai.save(batch)
```

### 에피소드 단위 보상

게임이 끝난 뒤 결과를 알 때 씁니다.

```python
steps = [ai.rl(상태) for 상태 in 게임진행()]
ai.save(ai.episode(steps, +1.0 if 이겼으면 else -1.0))
```

### 고를 수 있는 것이 제한될 때

```python
step = ai.rl(상태, legal=["왼쪽", "정지"])   # 이 중에서만 고릅니다
```

---

## 숫자 출력 (cos)

고를 이름 대신 `cos` 를 넘기면 실수를 반환합니다.

```python
ai = make("NumModel", [2, 64], cos)

step = ai.rl([0.5, 0.5])
print(step.output)                       # 예: 1.63  (float)

오차 = abs(step.output - 3.0)
ai.save(ai.reward(step, 1.0 - 오차))     # 3 에 가까울수록 높은 점수
```

값은 **0 근처에서 시작합니다.** 원하는 범위가 따로 있으면 쓰는 쪽에서 펼쳐 쓰세요.

```python
파워 = 20 + step.output * 10      # 출력 -1~1 → 파워 10~30
```

---

## 출력 여러 개

레이어 마지막을 리스트로 주면 출력이 여러 갈래가 됩니다.
출력끼리 몸통 신경망을 공유합니다.

```python
ai = make("Game", [38, 128, 128],
          [["점프", "왼쪽", "오른쪽", "정지"], cos])

step = ai.rl(상태)
행동, 파워 = step.output          # ('왼쪽', 27.3)
```

고르기와 `cos` 를 자유롭게 섞을 수 있고, 개수 제한도 없습니다.

```python
make("M", [입력, 은닉, 은닉], [cos, cos, ["안녕", "친구들"], ["영어", "한국어"]])
```

### 출력마다 다른 보상

```python
ai.save(ai.reward(step, [행동점수, 파워점수]))
```

`None` 을 주면 그 출력은 이번 학습에서 빠집니다.
점프를 안 한 순간에 점프 파워를 학습시키면 안 되니까요.

```python
ai.save(ai.reward(step, [점수, 점수 if 점프했나 else None]))
```

### 출력마다 고를 수 있는 것 제한

```python
ai.rl(상태, legal=[["왼쪽", "정지"], None])
```

---

## 어텐션 층

은닉층 자리에 `attn(조각수)` 를 넣으면, 그 층에서 **값들끼리 서로 참조**합니다.

```python
from my_ml import make, attn

ai = make("M", [38, 128, attn(8), 128], ["왼쪽", "오른쪽"])
```

폭 128을 8조각(각 16칸)으로 나눠서, 어느 조각을 볼지 스스로 정합니다.
들어온 폭 그대로 나가므로 아무 데나 끼울 수 있고, 여러 개 넣어도 됩니다.

```python
make("M", [64, 128, attn(8, 2), 128, attn(4), 64], 출력)
#                    ↑ 조각 8, 헤드 2
```

**언제 쓰나** — 입력에 같은 종류가 여러 개 있고, 그 중 뭐가 중요한지가 상황마다 바뀔 때.
적이 여러 명, 맵 칸 여러 개 같은 경우입니다.
그냥 서로 다른 값들(체력, 속도, 거리)뿐이면 효과가 없고 계산만 늘어납니다.

**비용** — 조각 수의 제곱에 비례합니다. 조각을 너무 많이 나누면 느려집니다.

---

## 언어 모델

글자를 이어 쓰는 모델입니다. 정답을 따로 줄 필요가 없습니다 — 다음 글자가 곧 정답입니다.

```python
from my_ml import make_lm

lm = make_lm("내모델", dim=128, layers=3, heads=4, ctx=48)

lm.sl(텍스트, steps=2000)        # 학습
print(lm.gen("안녕", 40))        # 이어 쓰기
lm.save()
```

| 인자 | |
|---|---|
| `dim` | 폭. `heads` 로 나누어떨어져야 합니다 |
| `layers` | 층 수 |
| `heads` | 어텐션 헤드 수 |
| `ctx` | 한 번에 보는 글자 수 |

생성할 때 `temp` 를 낮추면 뻔하게, 높이면 엉뚱하게 씁니다.
`topk` 를 주면 상위 몇 개 중에서만 고릅니다.

```python
lm.gen("안녕", 40, temp=0.5, topk=5)
lm.loss("안녕하세요")            # 학습 없이 얼마나 잘 맞히는지
```

가중치는 `이름_lm.pth` 로 저장됩니다.

---

## 지도학습

정답을 알고 있을 때 씁니다.

```python
ai.sl([0.1, 0.2, 0.3], "왼쪽")   # 정답을 주면 학습하고 예측을 반환
ai.sl([0.1, 0.2, 0.3])           # 정답이 없으면 예측만
```

출력이 여러 개면 정답도 하나씩 줍니다. `None` 은 건너뜁니다.

```python
ai.sl(상태, ["점프", 1.0])        # 첫 출력 정답, 둘째 출력 목표값
ai.sl(상태, ["왼쪽", None])       # 둘째 출력은 학습 안 함
```

## 예측만 하기

학습도 샘플링도 없이, 가장 점수가 높은 것을 반환합니다.

```python
ai.predict([0.1, 0.2, 0.3])
```

---

## 그 밖에

```python
from my_ml import gc_disable, gc_collect

gc_disable()    # 학습 루프 전. D GC 를 멈춰 중간 끊김을 없앱니다.
...
gc_collect()    # 학습 후. GC 재개 + 수집.

resset("MyModel")   # 저장된 가중치 파일 삭제
```

---

## 예전 버전에서 넘어오기

v0.1 과 v0.2 는 API 가 호환되지 않습니다. 예전 코드는 그대로 돌아가지 않습니다.
예전 버전이 필요하면 `git checkout v0.1` 로 돌아갈 수 있습니다.

### 가중치 파일 변환

```python
from my_ml import change

change("MyModel")     # MyModel_ml_memory.pth 를 새 포맷으로
```

원본은 `.bak` 으로 남습니다. 이미 새 포맷이면 아무것도 하지 않습니다.
예전에 출력을 여러 개 썼다면 그대로 다 보존됩니다.

### 바뀐 것

| v0.1 | v0.2 |
|---|---|
| `make("M", ["A","B"], hidden_layers=[128])` | `make("M", [입력수, 128], ["A","B"])` |
| `chosen = ai.rl(acts, 체력, 라운드)` | `step = ai.rl([체력, 라운드])` |
| `ai.reward(+1)` — 여기서 학습 | `ai.save(ai.reward(step, +1))` — 여기서 학습 |
| `with ai.episode():` | `ai.episode(steps, 점수)` |
| `ai.save()` | `ai.save(scored)` — 인자 필수 |
| `ai.step()`, `ai.last_reward()` | 없어짐 |
| `make(..., reset=True)` | 없어짐 (`resset("M")` 로 파일 삭제) |

`reward()` 는 이름이 그대로지만 **더 이상 학습하지 않습니다.** 데이터만 만듭니다.
예전 코드가 에러 없이 조용히 학습만 안 되는 상태가 되니 주의하세요.

---

## 빌드

```powershell
$ldc   = "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"
& $ldc my_ml.d $pylib --O3 --release --shared --link-defaultlib-shared=false "-of=my_ml.pyd"
Remove-Item my_ml.obj, my_ml.lib, my_ml.exp -ErrorAction SilentlyContinue
```

`"-of=my_ml.pyd"` 의 따옴표는 필수입니다. 빼면 PowerShell 이 인자를 쪼개
`Error: unrecognized file extension pyd` 로 실패합니다.

---

## 성능

배치 1 온라인 학습(이 라이브러리의 주 용도)에서는 파이토치보다 빠릅니다.
직접 작성한 네이티브 파이토치 코드와 같은 조건으로 비교한 결과입니다.

| 구성 | 이 라이브러리 | 네이티브 PyTorch |
|---|---|---|
| `[1, 128, 3]` | **9.0 µs** | 707 µs |
| `[64, 128, 3]` | **91 µs** | 1,093 µs |
| `[256, 512, 3]` | 1,284 µs | **1,161 µs** |

망이 커지면(대략 13만 파라미터 이상) 파이토치가 유리해집니다.
배치 학습이 가능한 작업이라면 파이토치 쪽이 샘플당 훨씬 빠릅니다.
