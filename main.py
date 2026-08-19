# ─────────────────────────────────────────────
# 사용 예시 (Example)
# ─────────────────────────────────────────────
# 이 파일은 템플릿입니다. 아래를 참고해 자유롭게 작성하세요.
#
# 핵심 규칙:
#   rl()     — 순수. (입력, 출력) 데이터만 만든다. 모델 안 바뀜.
#   reward() — 순수. (입력, 출력, 보상) 데이터만 만든다. 모델 안 바뀜.
#   save()   — 여기서만 역전파 + 파일 저장이 일어난다.

from my_ml import make, gc_disable, gc_collect

# [입력 수, 은닉..., 출력 수] + 출력 이름
ai = make("MyModel", [1, 128, 128, 3], ["선택지A", "선택지B", "선택지C"])

gc_disable()  # D GC 비활성화 → 학습 중 GC 정지 없음

# 학습 루프 — 스텝을 모았다가 한꺼번에 save()
for epoch in range(300):
    batch = []
    for i in range(32):
        step = ai.rl([i / 32])                       # 순수: 데이터만
        point = +1.0 if step.output == "선택지A" else -1.0
        batch.append(ai.reward(step, point))         # 순수: 데이터만
    ai.save(batch)                                   # 여기서 학습 + 저장

gc_collect()

print("학습 결과:", ai.predict([0.5]))

# ── 스텝 하나씩 즉시 학습하려면 ────────────────
# step = ai.rl([0.5])
# ai.save(ai.reward(step, +1.0))

# ── 에피소드 전체에 같은 보상 ──────────────────
# steps = [ai.rl([i / 10]) for i in range(10)]
# ai.save(ai.episode(steps, +1.0))

# ── 숫자를 출력하려면 (cos 모드) ───────────────
# num_ai = make("NumModel", [2, 64, 1], "cos")
# step = num_ai.rl([0.5, 0.5])
# print(step.output)          # float
# num_ai.save(num_ai.reward(step, 1.0 - abs(step.output - 3.0)))

# ── 지도학습 ───────────────────────────────────
# ai.sl([0.5], "선택지A")     # 정답 주고 1스텝 학습
# ai.sl([0.5])                # 정답 없이 예측만
