from torch import nn, relu

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=4, batch_first=True)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_length, hidden_size)
        last_hidden_state = rnn_out[:, -1, :]  # Use the last time step's output
        output = self.fc(last_hidden_state)  # Predict future steps
        return output

class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4, batch_first=True)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        last_hidden_state = lstm_out[:, -1, :]  # Use the last time step's output
        output = self.fc(last_hidden_state)  # Predict future steps
        return output
    
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=4, kernel_size=2, padding=1)  # Change kernel_size
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # Conv1d expects (batch_size, channels, seq_length)
        x = self.pool(relu(self.conv1(x)))  # Apply Conv1D and Pooling
        x = x.transpose(1, 2)  

        output = self.fc(x[:, -1, :])  # Predict future steps
        return output