import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import inspect
import builtins


# ─────────────────────────────────────────────
# [내부] 에피소드 컨텍스트 매니저
# ─────────────────────────────────────────────
class _EpisodeContext:
    """
    with ai.episode(): 블록 안에서 rl() 호출을 순서대로 기억합니다.
    - reward()     : 직전 rl() 하나에만 즉시 보상 (에피소드 버퍼에서 제거됨)
    - last_reward() : 아직 reward()를 받지 않은 모든 rl() 선택에 한꺼번에 역전파
                      with 블록 밖에서 지연 호출해도 동작합니다.
    """
    def __init__(self, ai):
        self._ai = ai

    def __enter__(self):
        self._ai._episode_log_probs = []   # 에피소드 버퍼 초기화
        self._ai._in_episode        = True
        return self._ai                    # as 절로 ai 객체 바로 쓸 수 있음

    def __exit__(self, *args):
        self._ai._in_episode = False       # 에피소드 끝 — 버퍼는 last_reward() 때까지 유지
        return False


# ─────────────────────────────────────────────
# [내부] 신경망 구조
# ─────────────────────────────────────────────
class _Network(nn.Module):
    def __init__(self, input_size, hidden_layers, action_lists):
        super().__init__()
        
        # 다층 은닉층 동적 구성
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        self.hidden = nn.Sequential(*layers)
        
        # 출력 레이어
        self.output_layers = nn.ModuleList([
            nn.Linear(prev_size, len(al)) for al in action_lists
        ])

    def forward(self, x):
        x = self.hidden(x)
        return [layer(x) for layer in self.output_layers]


# ─────────────────────────────────────────────
# [핵심] 통합 AI 클래스
# ─────────────────────────────────────────────
class BlackBoxAI:
    def __init__(self, model_name, action_lists=None, hidden_layers=None, reset=False):
        self.model_name = model_name
        
        if action_lists and isinstance(action_lists[0], str):
            self.action_lists = [action_lists]
        else:
            self.action_lists = action_lists

        # 🔧 Fix: 뮤터블 기본값 문제 — None을 기본으로 사용
        if hidden_layers is None:
            hidden_layers = [128]
        if isinstance(hidden_layers, int):
            self.hidden_layers = [hidden_layers]
        else:
            self.hidden_layers = list(hidden_layers)
            
        self.file_name    = f"{model_name}_ml_memory.pth"
        self.network      = None
        self.optimizer    = None
        self.criterion    = nn.CrossEntropyLoss()

        if reset and os.path.exists(self.file_name):
            os.remove(self.file_name)
            print(f" [{self.model_name}] 기존 학습 데이터를 삭제하고 초기화했습니다.")

        # RL 전용 버퍼 (직전 rl() 한 스텝)
        self._rl_log_probs     = []
        # 🔧 Fix: 에피소드 버퍼에 추가된 실제 객체 참조를 별도 보관
        self._rl_step_ref       = None
        # 에피소드 버퍼 (with episode(): 내 rl() 누적 — reward() 받은 것은 제외)
        self._episode_log_probs = []
        self._in_episode        = False

    # ── 내부 공통: 네트워크 초기화 및 가중치 로드 ──
    def _ensure_network(self, input_size):
        if self.network is not None:
            return

        saved_state_dict = None
        if os.path.exists(self.file_name):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                sig    = inspect.signature(torch.load)
                kwargs = {'map_location': device}
                if 'weights_only' in sig.parameters:
                    kwargs['weights_only'] = True
                loaded_data = torch.load(self.file_name, **kwargs)

                if isinstance(loaded_data, dict) and "state_dict" in loaded_data:
                    saved_state_dict = loaded_data["state_dict"]
                    if self.action_lists is None:
                        self.action_lists = loaded_data.get("action_lists", None)
                    if "hidden_layers" in loaded_data:
                        self.hidden_layers = loaded_data["hidden_layers"]
                else:
                    saved_state_dict = loaded_data

                # 이전 버전의 단일 은닉층(fc1) 구조를 새로운 다층 은닉층(hidden) 구조와 호환되도록 매핑
                fixed_state_dict = {}
                for k, v in saved_state_dict.items():
                    if k.startswith("fc1."):
                        fixed_state_dict[k.replace("fc1.", "hidden.0.")] = v
                    else:
                        fixed_state_dict[k] = v
                saved_state_dict = fixed_state_dict
                
                print(f" [{self.model_name}] 이전 학습 데이터를 성공적으로 불러왔습니다!")
            except Exception as e:
                print(f" [{self.model_name}] 데이터 불러오기 실패 (오류: {e}). 새로 시작합니다.")
                if os.path.exists(self.file_name):
                    os.remove(self.file_name)

        if self.action_lists is None:
            raise ValueError(f" [{self.model_name}] 모델을 처음 생성할 때는 action_lists(전체 액션)를 지정해야 합니다.")

        self.network   = _Network(input_size, self.hidden_layers, self.action_lists)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

        if saved_state_dict is not None:
            self.network.load_state_dict(saved_state_dict)
        else:
            print(f" [{self.model_name}] 새로 생성되었습니다.")

    # ── RL: 강화 학습 ──
    def rl(self, legal_actions, *env_args):
        """
        강화 학습 모드.
        첫 번째 인자로 현재 턴에 선택 가능한(합법적인) 액션 리스트를 전달하고,
        환경값들을 그 뒤에 인자로 전달하면 AI가 확률적으로 행동을 선택하고 반환합니다.
        이후 .reward(점수) 를 호출해 학습시킵니다.
        """
        if legal_actions and isinstance(legal_actions[0], str):
            legal_action_lists = [legal_actions]
        else:
            legal_action_lists = legal_actions

        input_data = [float(x) for x in env_args]
        self._ensure_network(len(input_data))

        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        predictions  = self.network(input_tensor)

        current_lps = []
        outputs     = []

        for i, pred in enumerate(predictions):
            all_actions = self.action_lists[i]
            legal_al    = legal_action_lists[i]
            
            # 합법적 액션들의 인덱스 매핑 및 Logits 추출 (액션 마스킹)
            indices       = [all_actions.index(a) for a in legal_al]
            masked_logits = pred[0][indices]
            
            probs  = F.softmax(masked_logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            chosen = dist.sample()

            current_lps.append(dist.log_prob(chosen))
            outputs.append(legal_al[chosen.item()])

        # 직전 스텝 버퍼 교체
        self._rl_log_probs = current_lps

        # 에피소드 진행 중이면 순서대로 누적 (같은 리스트 객체를 참조)
        if self._in_episode:
            self._episode_log_probs.append(current_lps)
            # 🔧 Fix: 에피소드 버퍼에 추가한 객체 참조를 저장 (reward()에서 제거용)
            self._rl_step_ref = current_lps
        else:
            self._rl_step_ref = None

        return outputs[0] if len(outputs) == 1 else outputs

    # ── SL: 지도 학습 ──
    def sl(self, legal_actions, *args):
        """
        지도 학습 모드.
        첫 번째 인자로 현재 턴에 선택 가능한(합법적인) 액션 리스트를 전달하고,
        환경값들을 먼저, 마지막 인자로 정답(문자열)을 전달합니다.
        정답 없이 환경값만 전달하면 예측 전용(학습 없음)으로 동작합니다.
        """
        if legal_actions and isinstance(legal_actions[0], str):
            legal_action_lists = [legal_actions]
        else:
            legal_action_lists = legal_actions

        if not args:
            return None

        # 🔧 Fix: 다중 헤드 지원 — 마지막 인자가 리스트(헤드별 정답)이거나
        # 단일 문자열인 경우 모두 처리
        last = args[-1]
        if isinstance(last, (list, tuple)) and len(last) == len(legal_action_lists) \
                and all(isinstance(a, str) for a in last):
            # 다중 헤드: 마지막 인자가 [정답0, 정답1, ...] 형태
            answers  = list(last)
            env_args = args[:-1]
        elif isinstance(last, str) and last in legal_action_lists[0]:
            # 단일 헤드: 마지막 인자가 문자열 정답
            answers  = [last]
            env_args = args[:-1]
        else:
            answers  = []
            env_args = args

        input_data = [float(x) for x in env_args]
        self._ensure_network(len(input_data))
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)

        # 학습 모드 (정답이 있을 때만) — 모든 헤드에 대해 처리
        if answers:
            self.network.train()
            predictions = self.network(input_tensor)
            total_loss  = None

            for i, answer in enumerate(answers):
                all_actions   = self.action_lists[i]
                legal_al      = legal_action_lists[i]
                indices       = [all_actions.index(a) for a in legal_al]
                masked_logits = predictions[i][:, indices]
                target_idx    = legal_al.index(answer)
                loss_i        = self.criterion(masked_logits, torch.LongTensor([target_idx]))
                total_loss    = loss_i if total_loss is None else total_loss + loss_i

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # 예측 반환
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(input_tensor)
            results = []
            for i, pred in enumerate(predictions):
                all_actions = self.action_lists[i]
                legal_al    = legal_action_lists[i]
                indices     = [all_actions.index(a) for a in legal_al]
                masked_logits = pred[:, indices]
                best_idx = torch.argmax(masked_logits, dim=1).item()
                results.append(legal_al[best_idx])
                
        return results[0] if len(results) == 1 else results

    # ── RL 보상 ──
    def reward(self, score):
        """
        rl() 호출 후 행동에 대한 보상을 줍니다. (Policy Gradient)
        양수 = 잘했어, 음수 = 잘못했어
        """
        if self.network is None or not self._rl_log_probs:
            return

        loss = sum(-lp * score for lp in self._rl_log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 🔧 Fix: _rl_step_ref(에피소드 버퍼에 추가된 실제 객체)로 제거
        # 이전 코드는 _rl_log_probs가 덮어써진 경우 참조가 달라 제거가 안 됐음
        if self._rl_step_ref is not None and self._rl_step_ref in self._episode_log_probs:
            self._episode_log_probs.remove(self._rl_step_ref)
            self._rl_step_ref = None

        # 직전 스텝 버퍼 초기화 (중복 역전파 방지)
        self._rl_log_probs = []

    # ── RL 에피소드 컨텍스트 ──
    def episode(self):
        """
        with 블록으로 rl() 선택들을 에피소드 단위로 묶습니다.
        """
        return _EpisodeContext(self)

    # ── RL 에피소드 전체 보상 ──
    def last_reward(self, score):
        """
        with ai.episode(): 블록이 끝난 뒤에도 호출 가능.
        reward()를 받지 않은 에피소드 내 모든 rl() 선택에 한꺼번에 역전파합니다.
        """
        # 🔧 Fix: _in_episode 조건 제거 — with 블록 밖 지연 호출도 허용
        # (독스트링에 명시된 동작과 일치하도록 수정)
        if not self._episode_log_probs:
            print(f" [{self.model_name}] 에피소드 버퍼가 비어 있습니다. "
                   "(episode() 블록과 rl()을 먼저 사용하거나 모든 스텝에 reward()를 이미 주었습니다.)")
            return

        # 에피소드 안의 모든 스텝 log_prob을 평탄화
        all_lps = [lp for step_lps in self._episode_log_probs for lp in step_lps]
        loss = sum(-lp * score for lp in all_lps)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 에피소드 버퍼 초기화
        self._episode_log_probs = []

    # ── 저장 ──
    def save(self):
        """학습된 가중치와 전체 액션 리스트, 은닉층 구조를 파일로 저장합니다."""
        if self.network is not None:
            save_data = {
                "state_dict": self.network.state_dict(),
                "action_lists": self.action_lists,
                "hidden_layers": self.hidden_layers
            }
            torch.save(save_data, self.file_name)
            print(f" [{self.model_name}] 저장 완료.")


# ─────────────────────────────────────────────
# [공개 API] 팩토리 함수
# ─────────────────────────────────────────────
def make(model_name, action_lists=None, hidden_layers=None, reset=False):
    """
    BlackBoxAI 인스턴스를 생성합니다.

    인자:
        model_name    : 모델 이름 (가중치 파일명에 사용)
        action_lists  : 가능한 모든 액션 목록의 리스트
        hidden_layers : 은닉층 구조 리스트 예) [128, 64]  (기본값: [128])
        reset         : True이면 기존 가중치를 삭제하고 처음부터 시작
    """
    return BlackBoxAI(model_name, action_lists, hidden_layers=hidden_layers, reset=reset)


# ─────────────────────────────────────────────
# [편의] resset() / reset() 전역 빌트인 등록
# ─────────────────────────────────────────────
def _global_resset(model_name):
    """my_ml, my_sl, my_rl 세 가지 형식의 가중치 파일을 모두 삭제합니다."""
    deleted = False
    for suffix in ["_ml_memory.pth", "_sl_memory.pth", "_auto_memory.pth"]:
        path = f"{model_name}{suffix}"
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f" [{model_name}] 모델 파일({path})이 초기화되었습니다.")
                deleted = True
            except Exception as e:
                print(f"오류: {path} 삭제 실패 ({e})")
    if not deleted:
        print(f" [{model_name}] 모델의 가중치 파일이 존재하지 않습니다. (이미 초기화 상태)")


builtins.resset = _global_resset
builtins.reset  = _global_resset
