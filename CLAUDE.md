# dplayg

D언어로 구현한 RL/SL 라이브러리. `my_ml.d` → `my_ml.pyd` (.pyd = Python extension, `import my_ml`로 사용).
가중치는 `이름_ml_memory.pth` 로 저장한다 (자체 포맷, 현재 ver 6).

## 빌드

```powershell
$ldc   = "ldc2\ldc2-1.42.0-windows-x64\bin\ldc2.exe"
$pylib = "$env:LOCALAPPDATA\Programs\Python\Python313\libs\python313.lib"
& $ldc my_ml.d $pylib --O3 --release --shared --link-defaultlib-shared=false "-of=my_ml.pyd"
Remove-Item my_ml.obj, my_ml.lib, my_ml.exp -ErrorAction SilentlyContinue
```

`"-of=my_ml.pyd"` 의 따옴표 필수. 빼면 PowerShell 이 인자를 쪼개서
`Error: unrecognized file extension pyd` 로 빌드가 실패한다.

## 파일

- `my_ml.d` — D 소스 (신경망 + Python C API + Python 클래스 코드 인라인)
- `my_ml.pyd` — 빌드 결과물 (gitignore)
- `main.py` — 빈 파일. 사용자가 여기에 작성한다.

## API

```python
from my_ml import make, cos, attn, each, tok

# layers 는 [입력수, 은닉...] 만. 출력 개수는 outputs 에서 정해진다.
ai = make("이름", [입력수, 은닉...], ["액션A", "액션B"])     # 고르기 1개
ai = make("이름", [입력수, 은닉...], cos)                    # 숫자 1개
ai = make("이름", [입력수, 은닉...],                          # 출력 여러 개
          [cos, ["a","b"], ["c","d","e"]])

step   = ai.rl([입력...])          # 순수 → Step(input, output)
scored = ai.reward(step, 점수)     # 순수 → Scored(input, output, point)
ai.save(scored)                    # 역전파 + 파일 자동 저장 (여기서만 모델이 바뀜)
ai.save([scored, ...])             # 묶음도 가능

ai.predict([입력...])              # 샘플링/학습 없이 최선값
ai.sl([입력...], "정답")           # 지도학습 1스텝 후 예측
ai.episode(steps, 점수)            # [Step...] → [Scored...] 일괄 보상 (순수)

change("이름")                     # 예전 포맷 가중치 → 현재 포맷 (.bak 백업)

# attn: 항목끼리 서로 참조 (폭은 그대로).  each: 항목마다 따로 가공 (가중치 공유)
# 일반 층을 사이에 끼우면 항목 구분이 사라진다 -> attn 과 each 를 번갈아 쓴다
ai = make("이름", [12, attn(6), each(24), attn(6), each(24), 128], 출력)

# 토큰 입력: 번호만 넘기고 펼치는 건 안에서 (one-hot 을 넘기면 데이터가 수백 배)
ai = make("이름", [tok(어휘수, 길이, 폭), 256, attn(8), 256], 출력)

# 묶음 학습: 기울기를 모았다가 갱신 1번. 하나씩 부르는 것보다 10배 이상 빠르다.
ai.sl([입력1, 입력2, ...], [정답1, 정답2, ...])
```

출력이 여러 개면 `output`, `reward` 의 점수, `sl` 의 정답이 모두 리스트다.
점수/정답에 `None` 을 주면 그 출력은 학습에서 빠진다.

`rl()`/`reward()`/`episode()` 는 모델을 건드리지 않는다. 학습은 `save()` 와 `sl()` 뿐이다.

`make()` 의 `sigma`(cos 탐험 폭, 기본 1.0), `entropy`(고르기가 한 답으로 굳는 것을
막는 힘, 기본 0.01) 로 학습 성향을 조절한다.

## 주의

- `cos` 값은 0 근처에서 시작한다. 원하는 범위가 있으면 쓰는 쪽에서 펼쳐 쓴다.
  예) `파워 = 20 + 출력 * 10`
- attn 만 넣으면 효과가 거의 없다. each 와 같이 써야 한다 (87% -> 99% 사례).
- `sl()` 을 하나씩 부르면 문제마다 가중치 전체를 갱신해서 매우 느리다. 묶음으로 준다.
- 드문 행동을 지도학습시킬 때는 여러 번 반복해야 한다.
  안 그러면 흔한 행동만 답하는 쪽으로 굳는다.
  (전체의 4% 인 행동은 "안 한다" 고만 답해도 96점이라 그쪽으로 수렴한다)
