import subprocess
import sys

# 라이브러리 임포트 시 자동으로 numpy 설치 여부를 확인하고 설치합니다.
try:
    import numpy
except ImportError:
    # 현재 사용 중인 가상환경의 파이썬 인터프리터로 설치를 진행합니다.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# [1층] 내부 인공지능 뇌 구조 (수정: 더 똑똑하게 128 노드로 확장!)
class _InternalNeuralNetwork(nn.Module):
    def __init__(self, input_size, action_lists):
        super(_InternalNeuralNetwork, self).__init__()
        self.action_lists = action_lists
        # 심사위원들 앞에서 더 정교한 판단을 하도록 64에서 128로 크기를 키웠어!
        self.fc1 = nn.Linear(input_size, 128)
        
        self.output_layers = nn.ModuleList()
        for action_list in action_lists:
            self.output_layers.append(nn.Linear(128, len(action_list)))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        results = []
        for layer in self.output_layers:
            results.append(layer(x))
        return results


# [2층] 사용자용 블랙박스 클래스 (수정: 자동 세이브/로드 및 보안 코드 적용)
class BlackBoxAI:
    def __init__(self, model_name, action_lists, reset=False):
        self.model_name = model_name
        self.action_lists = action_lists
        self.network = None
        self.optimizer = None
        self.file_name = f"{model_name}_auto_memory.pth"

        # 사용자가 리셋을 원하면 기존 파일을 삭제합니다.
        if reset and os.path.exists(self.file_name):
            os.remove(self.file_name)
            print(f" [{self.model_name}] 기존 학습 데이터를 삭제하고 초기화했습니다.")
        
        self.last_log_probs = []
        self.last_chosen_indices = []

    def env(self, *args):
        input_data = list(args)
        input_size = len(input_data)
        
        if self.network is None:
            self.network = _InternalNeuralNetwork(input_size, self.action_lists)
            self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
            
            if os.path.exists(self.file_name):
                try:
                    # CPU/GPU 장치 호환성 및 PyTorch 2.6+ weights_only 호환성 처리
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    import inspect
                    sig = inspect.signature(torch.load)
                    kwargs = {'map_location': device}
                    if 'weights_only' in sig.parameters:
                        kwargs['weights_only'] = True
                    checkpoint = torch.load(self.file_name, **kwargs)
                    self.network.load_state_dict(checkpoint)
                    print(f" [{self.model_name}] 이전 학습 데이터를 성공적으로 불러왔습니다!")
                except Exception as e:
                    print(f" [{self.model_name}] 데이터가 손상되었거나 구조가 달라 새로 시작합니다. (오류: {e})")
                    if os.path.exists(self.file_name):
                        os.remove(self.file_name)
            else:
                print(f"✨ [{self.model_name}] 새로 생성되었습니다.")

        # 입력을 배치(Batch) 형태로 변환 (1, input_size)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        predictions = self.network(input_tensor)
        
        self.last_log_probs = []
        self.last_chosen_indices = []
        outputs = []
        
        for i, pred in enumerate(predictions):
            # 확률 분포를 만들어서 행동을 샘플링합니다 (탐험 기능 추가)
            probs = F.softmax(pred[0], dim=-1)
            dist = torch.distributions.Categorical(probs)
            chosen_action_tensor = dist.sample()
            chosen_index = chosen_action_tensor.item()
            
            self.last_log_probs.append(dist.log_prob(chosen_action_tensor))
            self.last_chosen_indices.append(chosen_index)
            
            chosen_action = self.action_lists[i][chosen_index]
            outputs.append(chosen_action)
            
        return outputs

    def reward(self, score):
        """ 보상 점수를 받아 내부 오차를 계산하고 뇌세포 가중치를 업데이트하는 함수 """
        if self.network is None or not self.last_log_probs:
            return

        loss = 0
        for log_prob in self.last_log_probs:
            # Policy Gradient: 확률의 로그값에 보상을 곱해 손실을 계산합니다.
            loss += -log_prob * score 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 가중치 업데이트 후 로그 확률 버퍼 초기화 (중복 업데이트 방지)
        self.last_log_probs = []
        self.last_chosen_indices = []

    def save(self):
        """ 학습된 가중치를 파일로 저장합니다. 루프 밖에서 호출하는 것이 좋습니다. """
        if self.network is not None:
            torch.save(self.network.state_dict(), self.file_name)


#  [3층] [수정] main.py에서 'from my_rl import make'로 불러다 쓸 수 있게 만든 핵심 함수!
def make(model_name, action_lists, reset=False):
    """BlackBoxAI 인스턴스를 생성하는 팩토리 함수"""
    return BlackBoxAI(model_name, action_lists, reset=reset)

# 파이썬 대화형 셸(REPL) 등에서 resset('모델이름') / reset('모델이름')으로 모델을 직접 초기화할 수 있도록 내장(builtins) 함수 등록
def _global_resset(model_name):
    import os
    sl_file = f"{model_name}_sl_memory.pth"
    rl_file = f"{model_name}_auto_memory.pth"
    deleted = False
    for file_name in [sl_file, rl_file]:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f" [{model_name}] 모델 파일({file_name})이 초기화되었습니다.")
                deleted = True
            except Exception as e:
                print(f"오류: {file_name}을 삭제하는 중 문제가 발생했습니다. ({e})")
    if not deleted:
        print(f" [{model_name}] 모델의 가중치 파일이 존재하지 않습니다. (이미 초기화 상태)")

import builtins
builtins.resset = _global_resset
builtins.reset = _global_resset