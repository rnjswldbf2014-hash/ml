import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class _InternalSLNetwork(nn.Module):
    def __init__(self, input_size, action_lists):
        super(_InternalSLNetwork, self).__init__()
        self.action_lists = action_lists
        self.fc1 = nn.Linear(input_size, 128)
        self.output_layers = nn.ModuleList([nn.Linear(128, len(al)) for al in action_lists])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return [layer(x) for layer in self.output_layers]

class SLBlackBoxAI:
    def __init__(self, model_name, action_lists, reset=False):
        self.model_name = model_name
        self.action_lists = action_lists
        self.file_name = f"{model_name}_sl_memory.pth"
        self.network = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        if reset and os.path.exists(self.file_name):
            os.remove(self.file_name)
            print(f" [{self.model_name}] 초기화되었습니다.")

    def env(self, input_val, target_action=None):
        # 입력을 리스트로 변환 (숫자만 처리)
        input_data = [float(input_val)]
        input_size = len(input_data)

        if self.network is None:
            self.network = _InternalSLNetwork(input_size, self.action_lists)
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
                    print(f" [{self.model_name}] 데이터 불러오기 실패 (오류: {e}). 새로 시작합니다.")
                    if os.path.exists(self.file_name):
                        os.remove(self.file_name)

        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        
        # 학습 모드 (정답이 주어졌을 때)
        if target_action is not None:
            self.network.train()
            outputs = self.network(input_tensor)
            loss = 0
            # 첫 번째 액션 리스트에 대한 정답 인덱스 찾기
            try:
                target_idx = self.action_lists[0].index(target_action)
                target_tensor = torch.LongTensor([target_idx])
                loss = self.criterion(outputs[0], target_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except ValueError:
                print(f"Error: {target_action}은 액션 리스트에 없습니다.")

        # 예측 모드
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(input_tensor)
            results = []
            for i, pred in enumerate(predictions):
                idx = torch.argmax(pred, dim=1).item()
                results.append(self.action_lists[i][idx])
            return results[0] if len(results) == 1 else results

    def save(self):
        if self.network is not None:
            torch.save(self.network.state_dict(), self.file_name)
            print(f" [{self.model_name}] 저장 완료.")

def make(model_name, action_lists, reset=False):
    return SLBlackBoxAI(model_name, action_lists, reset=reset)

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