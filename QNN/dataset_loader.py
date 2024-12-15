import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def generate_multivariate_dataset_random_split(function, x_range, y_range, num_points, batch_size, train_size=0.6, valid_size=0.2, test_size=0.2):
    """
    Generates a dataset for a given multivariate function f(x, y) and randomly splits it into training, validation, and test sets.
    
    Parameters:
    - function: callable, the function f(x, y) to evaluate
    - x_range: tuple, the range of x values
    - y_range: tuple, the range of y values
    - num_points: int, the number of points to generate along each axis
    - batch_size: int, the batch size for the DataLoader
    - train_size: float, the proportion of data to be used for training (default is 0.6)
    - valid_size: float, the proportion of data to be used for validation (default is 0.2)
    - test_size: float, the proportion of data to be used for testing (default is 0.2)
    
    Returns:
    - train_loader: DataLoader for the training set
    - valid_loader: DataLoader for the validation set
    - test_loader: DataLoader for the testing set
    """
    
    # Generate the X, Y grid of points
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x_values, y_values)
    
    # Flatten X, Y to vectors
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    
    # Compute the function values f(x, y) for each point (x, y)
    Z = function(X_flat, Y_flat)
    
    # Create input sequences (X, Y) and corresponding targets Z
    data_points = np.column_stack((X_flat, Y_flat))  # shape: (num_samples, 2)
    
    # Sequence generation (similar to time series)
    sequences = []
    targets = []
    for i in range(len(data_points)):
        sequences.append([data_points[i]])
        targets.append([Z[i]])
    
    # Convert to tensors
    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
    
    # Randomly split the dataset into train, validation, and test sets
    total_size = len(sequences_tensor)
    
    # First, split off the test set (using test_size)
    train_valid_sequences, test_sequences, train_valid_targets, test_targets = train_test_split(
        sequences_tensor, targets_tensor, test_size=test_size, shuffle=True
    )
    
    # Now, split the remaining data into training and validation sets
    train_sequences, valid_sequences, train_targets, valid_targets = train_test_split(
        train_valid_sequences, train_valid_targets, test_size=valid_size / (1 - test_size), shuffle=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=TensorDataset(train_sequences, train_targets),
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        dataset=TensorDataset(valid_sequences, valid_targets),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=TensorDataset(test_sequences, test_targets),
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader, valid_loader, test_loader
