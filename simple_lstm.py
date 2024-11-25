import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        # LSTM 레이어 정의 (드롭아웃 추가)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 출력층 정의
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM은 기본적으로 초기 hidden state와 cell state를 자동으로 처리합니다.
        out, _ = self.lstm(x)  # 초기 hidden state와 cell state는 자동으로 처리됨
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력만 사용
        return out
