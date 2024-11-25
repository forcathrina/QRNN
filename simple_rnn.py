import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, w_ih=None, w_hh=None):
        super(RNN, self).__init__()

        # RNN 레이어 정의
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # 출력층 정의
        self.fc = nn.Linear(hidden_size, 1)

        # 초기화된 가중치가 제공되면 설정
        if w_ih is not None:
            with torch.no_grad():
                self.rnn.weight_ih_l0.fill_(w_ih)
        if w_hh is not None:
            with torch.no_grad():
                self.rnn.weight_hh_l0.fill_(w_hh)


    def forward(self, x):
        # RNN은 기본적으로 초기 h0 값을 자동으로 처리합니다.
        out, _ = self.rnn(x)  # RNN의 출력 x.shape (batch_size, sequence length, input_size)
        out = self.fc(out[:, -1, :])  # 전체 출력을 FC에 전달

        return out