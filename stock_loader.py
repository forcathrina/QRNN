import numpy as np
import torch

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def load_stock_data(ticker, start_date, end_date, interval, sequence_length, batch_size):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    close_prices = data['Close'].values
    dates = data.index  # 날짜 정보 저장

    # 입력 데이터 스케일러 (X)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler_X.fit_transform(close_prices)

    sequences = []
    targets = []
    sequence_dates = []  # 각 시퀀스에 해당하는 마지막 날짜 저장

    for i in range(len(close_prices) - sequence_length):
        sequences.append(close_prices[i:i + sequence_length])
        targets.append(close_prices[i + sequence_length])
        sequence_dates.append(dates[i + sequence_length])  # 마지막 날짜 저장

    stock_sequences = torch.tensor(np.array(sequences), dtype=torch.float32)

    # 출력 데이터 스케일러 (y)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    targets = scaler_y.fit_transform(np.array(targets))
    stock_targets = torch.tensor(np.array(targets), dtype=torch.float32)

    stock_dates = torch.tensor(np.array([date.timestamp() for date in sequence_dates], dtype=np.float32),
                               dtype=torch.float32)

    # Split into train, validation, and test (60%, 20%, 20%)
    train_size = int(len(stock_sequences) * 0.6)  # 60% for training
    valid_size = int(len(stock_sequences) * 0.2)  # 20% for validation
    test_size = len(stock_sequences) - train_size - valid_size  # 20% for testing

    train_sequences = stock_sequences[:train_size]
    valid_sequences = stock_sequences[train_size:train_size + valid_size]
    test_sequences = stock_sequences[train_size + valid_size:]

    train_targets = stock_targets[:train_size]
    valid_targets = stock_targets[train_size:train_size + valid_size]
    test_targets = stock_targets[train_size + valid_size:]

    test_dates = stock_dates[train_size + valid_size:]  # 테스트 데이터 날짜

    # DataLoader 생성 (날짜 포함)
    stock_train_loader = DataLoader(
        dataset=TensorDataset(train_sequences, train_targets),
        batch_size=batch_size,
        shuffle=True
    )

    stock_valid_loader = DataLoader(
        dataset=TensorDataset(valid_sequences, valid_targets),
        batch_size=batch_size,
        shuffle=False
    )

    stock_test_loader = DataLoader(
        dataset=TensorDataset(test_sequences, test_targets, test_dates),
        batch_size=batch_size,
        shuffle=False
    )

    return stock_train_loader, stock_valid_loader, stock_test_loader, scaler_X, scaler_y


def load_soaring_stock_data(sequence_length, batch_size):
    tickers = [
        "005930.KS",  # Samsung Electronics
        "000660.KS",  # SK hynix
        "035420.KS",  # NAVER Corporation
        "051910.KS",  # Hyundai Motor
        "005380.KS",  # LG Electronics
        "036570.KS",  # Kakao Corporation
        "003550.KS",  # Hyundai Mobis
        "012330.KS",  # SK Telecom
        "035720.KS",  # Kakao Games
        "017670.KS",  # SK Innovation
        "000270.KS",  # Samsung SDI
        "207940.KS",  # Samsung Biologics
        "018250.KS",  # LG Chem
        "005490.KS",  # POSCO Holdings
        "066570.KS",  # SK Group
        "008770.KS",  # Lotte Chemical
        "033780.KS",  # Samsung Life Insurance
        "028260.KS",  # SK Networks
        "032830.KS",  # Samsung Fire & Marine Insurance
        "009150.KS",  # Hyundai Steel
        "003670.KS",  # LG Display
        "086790.KS",  # S-Oil
        "036460.KS",  # Hyundai Glovis
        "029780.KS",  # Celltrion Healthcare
        "018880.KS",  # Amorepacific
        "051900.KS",  # SKC
        "032640.KS",  # KakaoBank
        "016360.KS",  # Hyundai Heavy Industries
        "033920.KS",  # SK Networks
        "105560.KS",  # Hana Financial Group
        "035250.KS",  # Samsung Electro-Mechanics
        "009830.KS",  # Daewoo Shipbuilding & Marine Engineering
        "003490.KS",  # Samsung Life Insurance
        "009420.KS",  # Samsung Engineering
        "010950.KS",  # SK Chemicals
        "006400.KS",  # LG Innotek
        "000810.KS",  # Samsung Everland
        "071050.KS",  # Hanwha Q CELLS
        "016880.KS",  # POSCO International
        "071970.KS",  # Samsung SDS
        "032500.KS",  # SK Innovation
        "086790.KS",  # S-Oil
        "006800.KS",  # Hyundai Heavy Industries
        "009270.KS",  # Hanon Systems
        "010130.KS",  # LG Uplus
        "017800.KS",  # E-Mart
        "035760.KS",  # LG Household & Health Care
        "009450.KS",  # Hyundai Mipo Dockyard
        "019170.KS"  # POSCO Energy
    ]

    all_soaring_sequences = []

    all_not_soaring_sequences = []

    for ticker in tickers:
        data = yf.download(ticker, start='2000-01-01', end='2024-01-01', interval='1d', progress=False)
        close_prices = data['Close'].values.reshape(-1, 1)
        dates = data.index

        # Normalize the close prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices = scaler.fit_transform(close_prices)

        # Iterate over the close prices to create sequences
        for i in range(len(close_prices) - sequence_length):
            # Identify price spikes (20% increase or more) for soaring
            if close_prices[i + sequence_length] >= close_prices[i + sequence_length - 1] * 1.2:
                # Add to soaring lists
                all_soaring_sequences.append(close_prices[i:i + sequence_length])
            else:
                # Add to not_soaring lists
                all_not_soaring_sequences.append(close_prices[i:i + sequence_length])

    # Convert to numpy arrays and then to PyTorch tensors
    all_soaring_sequences = np.array(all_soaring_sequences)

    all_not_soaring_sequences = np.array(all_not_soaring_sequences)

    # Create DataLoader for "soaring"
    soaring_sequences_tensor = torch.tensor(all_soaring_sequences, dtype=torch.float32)
    soaring_loader = DataLoader(
        dataset=TensorDataset(soaring_sequences_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # Create DataLoader for "not soaring"
    not_soaring_sequences_tensor = torch.tensor(all_not_soaring_sequences, dtype=torch.float32)
    not_soaring_loader = DataLoader(
        dataset=TensorDataset(not_soaring_sequences_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    return soaring_loader, not_soaring_loader
