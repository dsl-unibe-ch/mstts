"""Tests for the train_test_split function."""

import numpy as np
import pytest
from mstts.mstts import train_test_split


def test_train_test_split_simple():
    """Test train_test_split with simple binary data."""
    # Create a simple test case with binary data
    rows = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    # Scale up: fractional stratification is only meaningful once column totals
    # are large enough that the requested fraction is achievable with integer counts.
    rows = np.tile(rows, (5, 1))

    # Test with default parameters (should select ~20% for test)
    train_idx, test_idx = train_test_split(rows, v=2)

    # Check that we got valid selections
    assert train_idx is not None
    assert test_idx is not None
    assert isinstance(train_idx, list)
    assert isinstance(test_idx, list)

    # Check that train and test indices are disjoint and cover all rows
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx).union(set(test_idx)) == set(range(len(rows)))

    # Check that the fraction of test rows meets our expectations
    test_rows = rows[test_idx]
    column_sums = rows.sum(axis=0)
    test_sums = test_rows.sum(axis=0)

    # Default is 0.2 +/- 0.15, so between 0.05 and 0.35 of each column should be in test
    assert all((test_sums / column_sums) >= 0.05)
    assert all((test_sums / column_sums) <= 0.35)


def test_train_test_split_fraction():
    """Test train_test_split with different fractions."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )
    rows = np.tile(rows, (5, 1))  # scale up so fractional bounds are achievable

    # Test with 30% test fraction
    train_idx, test_idx = train_test_split(rows, test_frac=0.3, test_prec=0.1, v=2)

    assert train_idx is not None
    assert test_idx is not None

    test_rows = rows[test_idx]
    column_sums = rows.sum(axis=0)
    test_sums = test_rows.sum(axis=0)

    # Should be 0.3 +/- 0.1, so between 0.2 and 0.4
    assert all((test_sums / column_sums) >= 0.2)
    assert all((test_sums / column_sums) <= 0.4)


def test_train_test_split_n_strict():
    """Test train_test_split with n_strict parameter."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )
    rows = np.tile(rows, (5, 1))  # scale up so fractional bounds are achievable

    # Test with only first 2 columns as strict
    train_idx, test_idx = train_test_split(rows, test_frac=0.25, test_prec=0.1, v=2, n_strict=2)

    assert train_idx is not None
    assert test_idx is not None

    test_rows = rows[test_idx]
    column_sums = rows.sum(axis=0)
    test_sums = test_rows.sum(axis=0)

    # First 2 columns should be 0.25 +/- 0.1, so between 0.15 and 0.35
    assert all((test_sums[:2] / column_sums[:2]) >= 0.15)
    assert all((test_sums[:2] / column_sums[:2]) <= 0.35)

    # The other columns might not be within this range


def test_train_test_split_empty():
    """Test train_test_split with empty data."""
    rows = np.array([])

    # Should handle empty data gracefully
    with pytest.raises(IndexError):
        train_test_split(rows, v=0)


def test_train_test_split_edge_cases():
    """Test train_test_split with edge cases."""
    # All zeros
    rows = np.zeros((5, 3))
    train_idx, test_idx = train_test_split(rows, v=2)
    assert train_idx is not None
    assert test_idx is not None

    # All ones
    rows = np.ones((5, 3))
    train_idx, test_idx = train_test_split(rows, v=2)
    assert train_idx is not None
    assert test_idx is not None

    # Single column
    rows = np.array([[1], [0], [1], [1], [0]])
    train_idx, test_idx = train_test_split(rows, v=2)
    assert train_idx is not None
    assert test_idx is not None

    # Single row - this is an edge case where split doesn't make sense
    rows = np.array([[1, 0, 1]])
    train_idx, test_idx = train_test_split(rows, v=2)
    assert train_idx is not None or test_idx is not None  # At least one should be defined


def test_train_test_split_precision():
    """Test train_test_split with different precision values."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )
    rows = np.tile(rows, (5, 1))  # scale up so fractional bounds are achievable

    # Test with higher precision (narrower range)
    train_idx, test_idx = train_test_split(rows, test_frac=0.25, test_prec=0.05, v=2)

    assert train_idx is not None
    assert test_idx is not None

    test_rows = rows[test_idx]
    column_sums = rows.sum(axis=0)
    test_sums = test_rows.sum(axis=0)

    # Should be 0.25 +/- 0.05, so between 0.2 and 0.3
    assert all((test_sums / column_sums) >= 0.2)
    assert all((test_sums / column_sums) <= 0.3)
