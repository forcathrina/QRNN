import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def load_sinusoidal_data(sequence_length, num_samples, batch_size, freq_range=(0.1, 1.0)):
    np.random.seed(42)

    # Sine wave 데이터 생성
    X = []
    y = []
    dates = []  # 날짜 정보 생성
    for i in range(num_samples):
        freq = np.random.uniform(*freq_range)  # 주파수 랜덤화
        phase = np.random.uniform(0, 2 * np.pi)  # 위상 랜덤화
        t = np.linspace(0, 2 * np.pi, sequence_length + 1)  # 시간 축 생성
        sine_wave = np.sin(freq * t + phase)  # sine wave 생성

        X.append(sine_wave[:-1])  # 시퀀스 입력
        y.append(sine_wave[-1])  # 다음 값 예측 목표
        dates.append(i)  # 샘플 순서를 날짜로 저장 (더 복잡한 날짜 생성 가능)

    X = np.array(X).reshape(-1, sequence_length, 1)  # [samples, sequence_length, 1]
    y = np.array(y).reshape(-1, 1)  # [samples, 1]

    # 데이터 스케일링
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X = scaler_X.fit_transform(X.reshape(-1, sequence_length)).reshape(-1, sequence_length, 1)
    y = scaler_y.fit_transform(y)

    # Train/Test split
    train_size = int(num_samples * 0.6)
    valid_size = int(num_samples * 0.2)
    test_size = num_samples - train_size - valid_size

    X_train, X_valid, X_test = X[:train_size], X[train_size:train_size + valid_size], X[train_size + valid_size:]
    y_train, y_valid, y_test = y[:train_size], y[train_size:train_size + valid_size], y[train_size + valid_size:]
    dates_test = dates[train_size + valid_size:]  # 테스트 날짜

    # DataLoader 생성
    train_loader = DataLoader(
        dataset=TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        dataset=TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            torch.tensor(dates_test, dtype=torch.float32)  # 날짜 정보 포함
        ),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader, scaler_X, scaler_y
