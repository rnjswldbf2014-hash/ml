# ─────────────────────────────────────────────
# 사용 예시 (Example)
# ─────────────────────────────────────────────
# 이 파일은 템플릿입니다. 아래를 참고해 자유롭게 작성하세요.
# This file is a template. See README.md for usage.

from my_ml import make

# 액션 목록을 정의합니다.
actions = ["선택지A", "선택지B", "선택지C"]
ai = make("MyModel", actions)

# 학습 루프
for step in range(1000):
    legal = actions  # 현재 합법적인 선택지
    chosen = ai.rl(legal, step / 1000)

    # 보상 기준에 따라 수정하세요.
    ai.reward(+1.0 if chosen == "선택지A" else -1.0)

ai.save()
