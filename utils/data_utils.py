"""Data loading and preprocessing utilities for deep learning experiments."""

import numpy as np
from typing import Tuple, Optional


def load_dataset(path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Template function - Load a dataset from file (raises NotImplementedError).
    
    This is a template function that should be implemented based on your specific
    dataset format (CSV, NPY, HDF5, etc.).
    
    Args:
        path: Path to the dataset file
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (features, labels)
        
    Example:
        >>> # Implement based on your data format
        >>> # X, y = load_dataset('data/train.csv', normalize=True)
    """
    # Placeholder implementation - customize for your dataset
    raise NotImplementedError("Implement dataset loading for your specific format")


def normalize_data(data: np.ndarray, mean: Optional[float] = None, 
                   std: Optional[float] = None) -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Input data array
        mean: Optional mean to use (computed if None)
        std: Optional std to use (computed if None)
        
    Returns:
        Normalized data array
        
    Example:
        >>> normalized = normalize_data(train_data)
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    return (data - mean) / (std + 1e-8)


def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Feature array
        y: Label array
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """
    Create mini-batches from data.
    
    Args:
        X: Feature array
        y: Label array
        batch_size: Size of each batch
        
    Yields:
        Tuples of (X_batch, y_batch)
        
    Example:
        >>> for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
        ...     # Train on batch
    """
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        yield X[i:end], y[i:end]
