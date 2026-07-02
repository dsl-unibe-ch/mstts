"""Configuration file for pytest fixtures and shared utilities."""

import numpy as np
import pytest


@pytest.fixture
def binary_data():
    """Generate a simple binary dataset for testing."""
    return np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]]
    )


@pytest.fixture
def imbalanced_data():
    """Generate an imbalanced binary dataset for testing."""
    # Create an imbalanced dataset
    n_samples = 100
    data = np.zeros((n_samples, 3), dtype=int)

    # First column: 10% ones
    data[:10, 0] = 1

    # Second column: 50% ones
    data[:50, 1] = 1

    # Third column: 90% ones
    data[:90, 2] = 1

    return data


@pytest.fixture
def non_binary_data():
    """Generate a non-binary integer dataset for testing."""
    return np.array(
        [[3, 0, 2, 0], [1, 2, 0, 0], [2, 0, 1, 0], [0, 1, 3, 1], [1, 0, 0, 2], [0, 2, 1, 1], [2, 1, 0, 0], [0, 3, 1, 2]]
    )


@pytest.fixture
def realistic_data():
    """Generate a more realistic dataset with patterns for testing."""
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    n_features = 10

    # Create a dataset with some correlated features
    data = np.zeros((n_samples, n_features), dtype=int)

    # Add some pattern to make features correlated
    for i in range(n_samples):
        # First half of columns will be correlated
        if i % 3 == 0:
            data[i, :5] = 1
        elif i % 3 == 1:
            data[i, [0, 2, 4]] = 1
        else:
            data[i, [1, 3]] = 1

        # Second half has different pattern
        if i % 2 == 0:
            data[i, 5:] = 1
        else:
            data[i, [6, 8]] = 1

    return data
