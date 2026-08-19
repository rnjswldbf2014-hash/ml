# dplayg

D언어로 구현한 RL/SL 라이브러리. `my_ml.d` → `my_ml.pyd` (.pyd = Python extension, `import my_ml`로 사용).
`my_ml.py`(PyTorch 버전)와 API 동일, `.pth` 파일 포맷 호환.

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
- `main.py` — 사용 예시 템플릿
- `my_ml.pyd` — 빌드 결과물 (gitignore)

## API

```python
from my_ml import make

ai = make("이름", [입력수, 은닉..., 출력수], ["액션A", "액션B"])   # 이산
ai = make("이름", [입력수, 은닉..., 출력수], "cos")                # 숫자 출력

step   = ai.rl([입력...])          # 순수 → Step(input, output)
scored = ai.reward(step, 점수)     # 순수 → Scored(input, output, point)
ai.save(scored)                    # 역전파 + 파일 자동 저장 (여기서만 모델이 바뀜)
ai.save([scored, ...])             # 묶음도 가능

ai.predict([입력...])              # 샘플링/학습 없이 argmax
ai.sl([입력...], "정답")           # 지도학습 1스텝 후 예측
ai.episode(steps, 점수)            # [Step...] → [Scored...] 일괄 보상 (순수)

change("이름")                     # v0.1 가중치 파일 → v0.2 포맷 (.bak 백업)
```

`rl()`/`reward()`/`episode()` 는 모델을 건드리지 않는다. 학습은 `save()` 뿐이다.
