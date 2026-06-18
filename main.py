import subprocess
import sys

# ==========================================
# 🚀 [강제 연동] 가상환경 내부에 chess 라이브러리 자동 강제 설치
# ==========================================
try:
    import chess
except ModuleNotFoundError:
    print("📦 현재 가상환경에 'chess' 라이브러리가 없습니다. 자동 강제 설치를 시작합니다...")
    # 현재 실행 중인 파이썬 실행 파일 주소(sys.executable)를 집어서 정확하게 pip 설치를 내립니다.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess"])
    import chess
    print("✅ 'chess' 라이브러리 설치 및 연동 성공!")

import random
from my_ml import make

# ==========================================
# 🌌 1. 알파제로 규격 4,672가지 액션 공간 정의
# ==========================================
print("🔄 1) 알파제로 규격 4,672가지 표준 대수기보법(SAN) 액션 공간 생성 중...")

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
pieces = ['N', 'B', 'R', 'Q', 'K', '']

global_4672_san_actions = []
for p in pieces:
    for f in files:
        for r in ranks:
            global_4672_san_actions.append(f"{p}{f}{r}")
            global_4672_san_actions.append(f"{p}x{f}{r}")
            if p == '':
                for f_from in files:
                    if f_from != f:
                        global_4672_san_actions.append(f"{f_from}x{f}{r}")
for f in files:
    for r in ranks:
        global_4672_san_actions.append(f"N{f}{r}")
        global_4672_san_actions.append(f"Nx{f}{r}")
for f in files:
    for promo in ['=Q', '=R', '=B', '=N']:
        global_4672_san_actions.append(f"{f}8{promo}")
        global_4672_san_actions.append(f"{f}1{promo}")
        for f_from in files:
            if f_from != f:
                global_4672_san_actions.append(f"{f_from}x{f}8{promo}")
                global_4672_san_actions.append(f"{f_from}x{f}1{promo}")

global_4672_actions_expanded = []
for action in global_4672_san_actions:
    global_4672_actions_expanded.append(action)
    global_4672_actions_expanded.append(action + "+")
    global_4672_actions_expanded.append(action + "#")

global_4672_actions_expanded.extend(["O-O", "O-O+", "O-O#", "O-O-O", "O-O-O+", "O-O-O#"])
global_final_4672 = list(set(global_4672_actions_expanded))[:4672]
master_actions_set = set(global_final_4672)




import chess  # 진짜 체스 룰 엔진 라이브러리
from my_ml import make

# ==========================================
# 🌌 1. 알파제로 규격 4,672가지 액션 공간 정의
# ==========================================
print("🔄 1) 알파제로 규격 4,672가지 표준 대수기보법(SAN) 액션 공간 생성 중...")

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
pieces = ['N', 'B', 'R', 'Q', 'K', '']

global_4672_san_actions = []
for p in pieces:
    for f in files:
        for r in ranks:
            global_4672_san_actions.append(f"{p}{f}{r}")
            global_4672_san_actions.append(f"{p}x{f}{r}")
            if p == '':
                for f_from in files:
                    if f_from != f:
                        global_4672_san_actions.append(f"{f_from}x{f}{r}")
for f in files:
    for r in ranks:
        global_4672_san_actions.append(f"N{f}{r}")
        global_4672_san_actions.append(f"Nx{f}{r}")
for f in files:
    for promo in ['=Q', '=R', '=B', '=N']:
        global_4672_san_actions.append(f"{f}8{promo}")
        global_4672_san_actions.append(f"{f}1{promo}")
        for f_from in files:
            if f_from != f:
                global_4672_san_actions.append(f"{f_from}x{f}8{promo}")
                global_4672_san_actions.append(f"{f_from}x{f}1{promo}")

global_4672_actions_expanded = []
for action in global_4672_san_actions:
    global_4672_actions_expanded.append(action)
    global_4672_actions_expanded.append(action + "+")
    global_4672_actions_expanded.append(action + "#")

global_4672_actions_expanded.extend(["O-O", "O-O+", "O-O#", "O-O-O", "O-O-O+", "O-O-O#"])
global_final_4672 = list(set(global_4672_actions_expanded))[:4672]
master_actions_set = set(global_final_4672)


# ==========================================
# 🧠 2. 형의 'my_ml' 정밀 문법에 맞춘 새 모델 생성 (reset=True)
# ==========================================
print("\n🤖 2) hidden_layers 파라미터를 주입하여 백지 상태의 심층 신경망을 생성합니다...")

# 🔥 형이 알려준 exact 문법 반영: [256, 128, 64] 심층 구조 탑재!
# reset=True로 기존에 남아있던 잔재를 완전히 초기화하고 처음부터 새로 배웁니다.
white_ai = make("AlphaZero_White_Net", global_final_4672, hidden_layers=[256, 128, 64], reset=True)
black_ai = make("AlphaZero_Black_Net", global_final_4672, hidden_layers=[256, 128, 64], reset=True)


# 📜 기보 -> 숫자 벡터 변환기 (입력 데이터)
CHESS_VOCAB = "abcdefgh12345678NBRQKx+=#O-:"
char_to_id = {char: idx + 1 for idx, char in enumerate(CHESS_VOCAB)}

def board_to_vector(board_obj, max_length=30):
    notation_text = board_obj.sans() if hasattr(board_obj, 'sans') else str(board_obj)
    vector = [0] * max_length
    for i, char in enumerate(notation_text[:max_length]):
        if char in char_to_id:
            vector[i] = char_to_id[char]
    return vector


# ==========================================
# 📈 3. 진짜 체스 룰 기반 최초 자가대국 학습 (Scratch Training)
# ==========================================
# 3층 심층 신경망 연산량을 고려해, 최초 학습은 200판 / 판당 12턴으로 스마트하게 최적화 완료!
total_matches = 200
print(f"\n🔄 3) [256, 128, 64] DNN 구조로 진짜 체스 자가대국 {total_matches}판 처음부터 독학 시작...")

for match in range(total_matches):
    board = chess.Board()
    
    with white_ai.episode(), black_ai.episode():
        for turn in range(50): # 최대 12턴(총 24보) 제한
            if board.is_game_over():
                break
                
            # ⚪ [백의 턴]
            state_vector = board_to_vector(board)
            legal_moves = [board.san(m) for m in board.legal_moves]
            legal_moves = [m for m in legal_moves if m in master_actions_set]
            if not legal_moves: legal_moves = ["e4"]
            
            white_move_san = white_ai.rl(legal_moves, *state_vector)
            try:
                board.push_san(white_move_san)
            except Exception:
                board.push_san(legal_moves[0])
                
            if board.is_game_over():
                break

            # ⚫ [흑의 턴]
            state_vector = board_to_vector(board)
            legal_moves = [board.san(m) for m in board.legal_moves]
            legal_moves = [m for m in legal_moves if m in master_actions_set]
            if not legal_moves: legal_moves = ["e5"]
            
            black_move_san = black_ai.rl(legal_moves, *state_vector)
            try:
                board.push_san(black_move_san)
            except Exception:
                board.push_san(legal_moves[0])

        # 🧼 실제 룰 기반 지연 보상 정산
        result = board.result()
        if result == "1-0":
            white_ai.last_reward(1.0)
            black_ai.last_reward(-1.0)
        elif result == "0-1":
            white_ai.last_reward(-1.0)
            black_ai.last_reward(1.0)
        else:
            white_ai.last_reward(0.0)
            black_ai.last_reward(0.0)

# 딥 러닝된 고차원 전술 가중치 첫 세이브
white_ai.save()
black_ai.save()
print("💾 새 심층 신경망 가중치 파일 저장 완료!")


# ==========================================
# ⚔️ 4. AI(백) vs 권지율(흑) 1:1 진짜 리얼 체스 대국
# ==========================================
print("\n" + "="*50)
print("   ♟️  WELCOME TO DEEP DNN ALPHAZERO CHESS  ♟️")
print("-> [256, 128, 64] 레이어를 통과하며 고차원 전술을 학습한 AI입니다.")
print("-> 형은 [흑(Black)], AI는 [백(White)]입니다.")
print("="*50 + "\n")

play_board = chess.Board()

# ⚪ [1턴: 백(AI)의 선공]
print("🤖 AI(백)가 심층 신경망을 거쳐 첫 수를 계산하고 있습니다...")
legal_moves = [play_board.san(m) for m in play_board.legal_moves]
legal_moves = [m for m in legal_moves if m in master_actions_set]
state_vector = board_to_vector(play_board)

ai_move = white_ai.rl(legal_moves, *state_vector)
play_board.push_san(ai_move)

print(f"➡️  AI(백)의 첫 수: {ai_move}")
print(f"🧱 체스판 상황:\n{play_board}\n")

# 🔄 핑퐁 대국 루프
while True:
    if play_board.is_game_over():
        print(f"🏁 게임 종료! 최종 결과: {play_board.result()}")
        break
        
    # ⚫ [흑(인간)의 턴]
    legal_moves = [play_board.san(m) for m in play_board.legal_moves]
    legal_moves = [m for m in legal_moves if m in master_actions_set]
    
    print(f"⚔️ 형이 둘 수 있는 진짜 합법 수 목록:\n{legal_moves}\n")
    user_move = input("👉 둘 수를 입력하세요 (종료: q): ").strip()
    
    if user_move.lower() == 'q':
        print("👋 대국을 종료합니다.")
        break
        
    if user_move not in legal_moves:
        print("❌ 규칙상 불가능한 수입니다. 목록에 표시된 진짜 합법 수만 입력하세요!\n")
        continue
        
    play_board.push_san(user_move)
    print(f"🧱 형의 선택 반영 후 체스판 상황:\n{play_board}\n")
    
    if play_board.is_game_over():
        print(f"🏁 게임 종료! 최종 결과: {play_board.result()}")
        break

    # ──────────────────────────────────────
    
    # ⚪ [백(AI)의 턴]
    print("🤖 AI(백)가 깊어진 은닉층 뉴런들을 거쳐 다음 전술을 탐색합니다...")
    legal_moves = [play_board.san(m) for m in play_board.legal_moves]
    legal_moves = [m for m in legal_moves if m in master_actions_set]
    state_vector = board_to_vector(play_board)
    
    ai_move = white_ai.rl(legal_moves, *state_vector)
    play_board.push_san(ai_move)
    
    print(f"➡️  AI(백)의 응수: {ai_move}")
    print(f"🧱 현재 체스판 상황:\n{play_board}\n")