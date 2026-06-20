# BlackBox AI — 사용 설명서

PyTorch 기반의 간결한 머신러닝 인터페이스입니다.  
내부 구조를 몰라도 세 가지 모듈 중 하나를 가져와 몇 줄의 코드만으로 AI를 학습시키고 예측에 활용할 수 있습니다.

---

## 모듈 개요

| 모듈 | 학습 방식 | 권장 용도 |
|---|---|---|
| `my_ml.py` | 강화학습 + 지도학습 통합 | 합법 액션 마스킹이 필요한 복잡한 환경 |
| `my_rl.py` | 강화학습 (Policy Gradient) | 보상 신호만으로 학습하는 환경 |
| `my_sl.py` | 지도학습 (Cross-Entropy) | 정답이 명확히 주어지는 분류 문제 |

---

## my_ml.py — 통합 모듈

강화학습과 지도학습을 하나의 인터페이스로 제공합니다.  
합법 액션 마스킹, 에피소드 단위 학습, 다중 출력 헤드를 지원합니다.

### 가져오기

```python
from my_ml import make
```

### 모델 생성

```python
ai = make(
    model_name    = "MyModel",          # 가중치 파일 이름에 사용됩니다.
    action_lists  = ["가위", "바위", "보"],  # 가능한 모든 액션의 목록입니다.
    hidden_layers = [128, 64],          # 은닉층 구조입니다. 기본값은 [128]입니다.
    optimizer     = "adam",             # 옵티마이저를 선택합니다. 기본값은 'adam'입니다.
    lr            = 0.01,               # 학습률입니다. 기본값은 0.01입니다.
    reset         = False               # True로 설정하면 기존 가중치를 삭제합니다.
)
```

**지원하는 옵티마이저**

| 이름 | 설명 | 권장 상황 |
|---|---|---|
| `'adam'` | Adam (기본값) | 대부분의 경우 무난하게 사용 |
| `'sgd'` | 확률적 경사하강법 | 단순한 문제, 세밀한 조정이 필요할 때 |
| `'rmsprop'` | RMSprop | 순차 데이터, 비정상(non-stationary) 환경 |
| `'adagrad'` | Adagrad | 희소(sparse) 데이터가 많은 경우 |

- `action_lists`에 문자열 리스트를 직접 전달하면 단일 출력 헤드로 동작합니다.
- 다중 출력이 필요한 경우 `[리스트A, 리스트B]` 형태로 전달합니다.
- 가중치는 `{model_name}_ml_memory.pth` 파일에 자동으로 저장됩니다.

---

### 강화학습 — `rl()` / `reward()`

보상 신호를 통해 학습합니다. 매 스텝마다 합법 액션 목록을 전달하면  
AI가 확률적으로 하나를 선택하여 반환합니다.

```python
# 기본 사용 예시
chosen = ai.rl(legal_actions, *환경값들)
# 결과 확인 후 보상 부여
ai.reward(+1.0)   # 양수: 잘한 선택, 음수: 잘못된 선택
```

```python
# 예시: 현재 합법 수가 ["가위", "보"]인 경우
chosen = ai.rl(["가위", "보"], 체력, 라운드, 상대_패턴)
if chosen == 정답:
    ai.reward(+1.0)
else:
    ai.reward(-1.0)
```

---

### 강화학습 — 에피소드 단위 (`episode()` / `last_reward()`)

여러 스텝에 걸친 결과를 한꺼번에 학습시킬 때 사용합니다.

```python
with ai.episode():
    move1 = ai.rl(legal_moves_1, *상태1)
    move2 = ai.rl(legal_moves_2, *상태2)
    move3 = ai.rl(legal_moves_3, *상태3)
    # 중간에 개별 보상을 줄 수도 있습니다.
    # ai.reward(+0.5)  →  해당 스텝은 last_reward() 대상에서 제외됩니다.

# with 블록 밖에서 지연 호출도 가능합니다.
ai.last_reward(+1.0)   # 보상받지 않은 모든 스텝에 일괄 역전파합니다.
```

---

### 지도학습 — `sl()`

정답을 마지막 인자로 전달하면 학습하고, 생략하면 예측만 수행합니다.

```python
# 학습 + 예측
predicted = ai.sl(legal_actions, *환경값들, 정답)

# 예측만 (학습 없음)
predicted = ai.sl(legal_actions, *환경값들)
```

```python
# 예시
result = ai.sl(["가위", "바위", "보"], 라운드, 점수, "바위")
```

**다중 헤드 학습** — 정답을 리스트로 전달합니다.

```python
result = ai.sl(
    [["가위", "바위", "보"], ["공격", "방어"]],
    *환경값들,
    ["바위", "공격"]   # 각 헤드의 정답을 순서대로 담은 리스트
)
```

---

### 저장 — `save()`

학습이 끝난 후 호출하면 가중치, 액션 목록, 은닉층 구조를 파일로 저장합니다.  
다음 실행 시 자동으로 불러옵니다.

```python
ai.save()
```

---

## my_rl.py — 강화학습 전용 모듈

합법 액션 마스킹 없이 전체 액션 공간에서 강화학습을 수행합니다.

### 가져오기 및 모델 생성

```python
from my_rl import make

ai = make(
    model_name   = "MyModel",
    action_lists = [["가위", "바위", "보"]],
    reset        = False
)
```

### 사용 방법

```python
# 환경값 전달 → 액션 선택
outputs = ai.env(체력, 라운드, 점수)
chosen  = outputs[0]   # 단일 헤드인 경우

# 보상 부여 → 학습
ai.reward(+1.0)

# 저장
ai.save()
```

> **참고** `my_rl.py`는 합법 액션 마스킹을 지원하지 않습니다.  
> 액션 마스킹이 필요한 경우 `my_ml.py`의 `rl()` 사용을 권장합니다.

---

## my_sl.py — 지도학습 전용 모듈

단일 입력값과 정답을 전달하는 간단한 분류 학습에 적합합니다.

### 가져오기 및 모델 생성

```python
from my_sl import make

ai = make(
    model_name   = "MyModel",
    action_lists = [["가위", "바위", "보"]],
    reset        = False
)
```

### 사용 방법

```python
# 학습 + 예측
result = ai.env(입력값, "바위")

# 예측만
result = ai.env(입력값)

# 저장
ai.save()
```

---

## 모델 초기화

`reset()` 또는 `resset()` 함수가 전역 빌트인으로 등록되어 있어  
어느 모듈을 불러왔더라도 바로 사용할 수 있습니다.

```python
import my_ml   # 또는 my_rl, my_sl 중 하나만 불러와도 됩니다.

reset("MyModel")    # MyModel 관련 가중치 파일을 모두 삭제합니다.
resset("MyModel")   # 동일한 동작입니다.
```

삭제 대상 파일:

- `MyModel_ml_memory.pth`
- `MyModel_sl_memory.pth`
- `MyModel_auto_memory.pth`

---

## 전체 사용 예시 (my_ml.py)

```python
from my_ml import make

모든_수 = ["가위", "바위", "보"]
ai = make("가위바위보_AI", 모든_수)

for 라운드 in range(1000):
    합법_수 = ["가위", "바위", "보"]   # 상황에 따라 제한 가능
    선택 = ai.rl(합법_수, 라운드 / 1000)

    정답 = "바위"   # 예시 정답
    if 선택 == 정답:
        ai.reward(+1.0)
    else:
        ai.reward(-1.0)

ai.save()
```

---

## 가중치 파일 위치

가중치 파일은 스크립트를 실행하는 **현재 작업 디렉터리**에 생성됩니다.

| 모듈 | 파일명 형식 |
|---|---|
| `my_ml.py` | `{model_name}_ml_memory.pth` |
| `my_rl.py` | `{model_name}_auto_memory.pth` |
| `my_sl.py` | `{model_name}_sl_memory.pth` |
