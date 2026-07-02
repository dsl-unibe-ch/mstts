"""Tests for the get_row_selection function."""

import numpy as np
import pytest
from mstts.mstts import get_row_selection


def test_get_row_selection_simple():
    """Test get_row_selection with simple binary data."""
    # Create a simple test case with binary data
    rows = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    # Scale up: fractional stratification is only meaningful once column totals
    # are large enough that the requested fraction is achievable with integer counts.
    rows = np.tile(rows, (5, 1))

    # Test with default parameters (should select ~80% of data)
    selected = get_row_selection(rows, v=2)

    # Check that we got a valid selection
    assert selected is not None
    assert isinstance(selected, list)

    # Check that the selected rows are a subset of the original rows
    assert all(idx < len(rows) for idx in selected)

    # Check that the fraction of selected rows meets our expectations
    selected_rows = rows[selected]
    column_sums = rows.sum(axis=0)
    selected_sums = selected_rows.sum(axis=0)

    # Default is 0.8 +/- 0.15, so between 0.65 and 0.95 of each column should be selected
    assert all((selected_sums / column_sums) >= 0.65)
    assert all((selected_sums / column_sums) <= 0.95)


def test_get_row_selection_fraction():
    """Test get_row_selection with different fractions."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )

    # Test with 50% fraction
    selected = get_row_selection(rows, label_frac=0.5, label_prec=0.1, v=2)

    assert selected is not None

    selected_rows = rows[selected]
    column_sums = rows.sum(axis=0)
    selected_sums = selected_rows.sum(axis=0)

    # Should be 0.5 +/- 0.1, so between 0.4 and 0.6
    assert all((selected_sums / column_sums) >= 0.4)
    assert all((selected_sums / column_sums) <= 0.6)


def test_get_row_selection_n_strict():
    """Test get_row_selection with n_strict parameter."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )
    rows = np.tile(rows, (5, 1))  # scale up so fractional bounds are achievable

    # Test with only first 2 columns as strict
    selected = get_row_selection(rows, label_frac=0.75, label_prec=0.1, v=2, n_strict=2)

    assert selected is not None

    selected_rows = rows[selected]
    column_sums = rows.sum(axis=0)
    selected_sums = selected_rows.sum(axis=0)

    # First 2 columns should be 0.75 +/- 0.1, so between 0.65 and 0.85
    assert all((selected_sums[:2] / column_sums[:2]) >= 0.65)
    assert all((selected_sums[:2] / column_sums[:2]) <= 0.85)

    # The other columns might not be within this range


def test_get_row_selection_empty():
    """Test get_row_selection with empty data."""
    rows = np.array([])

    # Should handle empty data gracefully
    with pytest.raises(IndexError):
        get_row_selection(rows, v=2)


def test_get_row_selection_edge_cases():
    """Test get_row_selection with edge cases."""
    # All zeros
    rows = np.zeros((5, 3))
    selected = get_row_selection(rows, v=2)
    assert selected is not None

    # All ones
    rows = np.ones((5, 3))
    selected = get_row_selection(rows, v=2)
    assert selected is not None

    # Single column
    rows = np.array([[1], [0], [1], [1], [0]])
    selected = get_row_selection(rows, v=2)
    assert selected is not None

    # Single row
    rows = np.array([[1, 0, 1]])
    selected = get_row_selection(rows, v=2)
    assert selected is not None
    assert len(selected) <= 1


def test_get_row_selection_precision():
    """Test get_row_selection with different precision values."""
    rows = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]]
    )
    rows = np.tile(rows, (5, 1))  # scale up so fractional bounds are achievable

    # Test with higher precision (narrower range)
    selected = get_row_selection(rows, label_frac=0.75, label_prec=0.05, v=2)

    assert selected is not None

    selected_rows = rows[selected]
    column_sums = rows.sum(axis=0)
    selected_sums = selected_rows.sum(axis=0)

    # Should be 0.75 +/- 0.05, so between 0.7 and 0.8
    assert all((selected_sums / column_sums) >= 0.7)
    assert all((selected_sums / column_sums) <= 0.8)
