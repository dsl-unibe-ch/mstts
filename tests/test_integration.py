"""Integration tests for MSTTS module."""

import numpy as np

from mstts.mstts import get_row_selection, train_test_split


def test_integration_flow():
    """Test the complete workflow of MSTTS with realistic data."""
    # Create a more realistic dataset with multiple label columns
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    n_features = 10

    # Create a dataset with some correlated features
    rows = np.zeros((n_samples, n_features), dtype=int)

    # Add some pattern to make features correlated
    for i in range(n_samples):
        # First half of columns will be correlated
        if i % 3 == 0:
            rows[i, :5] = 1
        elif i % 3 == 1:
            rows[i, [0, 2, 4]] = 1
        else:
            rows[i, [1, 3]] = 1

        # Second half has different pattern
        if i % 2 == 0:
            rows[i, 5:] = 1
        else:
            rows[i, [6, 8]] = 1

    # First test get_row_selection with various parameters
    selected_80 = get_row_selection(rows, label_frac=0.8, v=2)
    assert selected_80 is not None
    assert len(selected_80) > 0

    # Then test train_test_split
    train_idx, test_idx = train_test_split(rows, test_frac=0.2, v=2)
    assert train_idx is not None
    assert test_idx is not None

    # Verify disjoint sets that cover all data
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx).union(set(test_idx)) == set(range(n_samples))

    # Verify class distribution in train and test sets
    train_rows = rows[train_idx]
    test_rows = rows[test_idx]

    train_distribution = train_rows.sum(axis=0) / len(train_rows)
    test_distribution = test_rows.sum(axis=0) / len(test_rows)

    # Check that distributions are reasonably similar (within 15% difference)
    assert all(abs(train_distribution - test_distribution) < 0.15)


def test_with_imbalanced_data():
    """Test MSTTS with highly imbalanced data."""
    # Create an imbalanced dataset
    n_samples = 100
    rows = np.zeros((n_samples, 3), dtype=int)

    # First column: 10% ones
    rows[:10, 0] = 1

    # Second column: 50% ones
    rows[:50, 1] = 1

    # Third column: 90% ones
    rows[:90, 2] = 1

    # Test train_test_split with this imbalanced data
    train_idx, test_idx = train_test_split(rows, test_frac=0.2, test_prec=0.1, v=2)

    assert train_idx is not None
    assert test_idx is not None

    test_rows = rows[test_idx]

    # Check that rare class (column 0) is represented in test set
    assert test_rows[:, 0].sum() > 0

    # Check that common class (column 2) is appropriately distributed
    assert 0.15 <= test_rows[:, 2].sum() / rows[:, 2].sum() <= 0.25


def test_with_non_binary_data():
    """Test MSTTS with non-binary data."""
    # Create dataset with integer values
    rows = np.array(
        [[3, 0, 2, 0], [1, 2, 0, 0], [2, 0, 1, 0], [0, 1, 3, 1], [1, 0, 0, 2], [0, 2, 1, 1], [2, 1, 0, 0], [0, 3, 1, 2]]
    )

    # Test both functions
    selected = get_row_selection(rows, v=2)
    assert selected is not None

    train_idx, test_idx = train_test_split(rows, v=2)
    assert train_idx is not None
    assert test_idx is not None

    # Check distribution of values
    test_rows = rows[test_idx]

    # Sum of values in each column should be proportionally distributed
    column_sums = rows.sum(axis=0)
    test_sums = test_rows.sum(axis=0)
    test_fractions = test_sums / column_sums

    # Default test fraction is 0.2 ± 0.15
    assert all(test_fractions >= 0.05)
    assert all(test_fractions <= 0.35)
