"""
MSTTS - Multi-Class Stratified Train Test Splitter

This module provides functionality for splitting datasets with multiple target classes
while maintaining the stratification across all classes.

Developer: Mykhailo Vladymyrov for needs of projects at UniBern, Switzerland
License: Apache License 2.0

Copyright 2023-2025 Mykhailo Vladymyrov, DSL, University of Bern, Switzerland
"""

import numpy as np
from ortools.sat.python import cp_model


def get_row_selection(  # noqa: C901  (solver status handling is inherently branchy)
    rows: np.ndarray | list[list],
    label_frac: float = 0.8,
    label_prec: float = 0.15,
    v: int = 1,
    n_strict: int | None = None,
    non_strict_weight: int | float = 1,
) -> list[int] | None:
    """
    Select indexes for subset of rows taking ~ `label_frac` positive samples

    given array m x n of boolean 0/1 `rows`, find such combination of rows such
     that sum of each column is within the range of
     [lo, hi] = [label_frac - label_prec, label_frac + label_prec] of
     total sum of the column.
    Use google ortools CP-SAT solver

    Args:
        rows: array of rows, with numbers,
        label_frac: desired fraction of total sum in each column to be selectes
        label_prec: precision of the fraction. default (80+-15% of all samples)
        v: verbose, print INFO messages if != 0, print all messages if == 2
        n_strict: if not None, use n_strict first columns as target columns,
        the rest only contribute to the optimization objective sum term
        non_strict_weight: weight of non-strict columns in the optimization objective.
        1 by default, can be float <1 to reduce the weight (0.1 min) or int >1 to
        increase the weight


    Returns:
        selected_rows: list of indexes of selected rows, or None if no solution found

    """
    # CP-SAT requires integer coefficients; cast so float inputs (e.g. np.zeros) work.
    rows = np.asarray(rows, dtype=int)
    n = rows.shape[0]
    m = rows.shape[1]

    strict_weight = 1
    if isinstance(non_strict_weight, float):
        non_strict_weight = int(10 * non_strict_weight)
        strict_weight = 10

    n_positive_samples = rows.sum(axis=0)

    n_strict = n_strict if n_strict is not None else m

    n_positive_thres_lo = (n_positive_samples * (label_frac - label_prec)).astype(
        int
    )  # min expected number of positive samples
    n_positive_thres_lo = np.maximum(n_positive_thres_lo, 0)  # if negative, clip to 0
    n_positive_thres_hi = (n_positive_samples * (label_frac + label_prec)).astype(
        int
    )  # max expected number of positive samples
    n_positive_thres_hi = np.minimum(n_positive_thres_hi, n_positive_samples)  # if more than total, clip to total
    n_positive_thres_best = (n_positive_samples * label_frac).astype(
        int
    )  # best expected number of positive samples - ideal target

    if v == 2:
        print(n_positive_samples)

        print(n_positive_thres_lo)
        print(n_positive_thres_best)
        print(n_positive_thres_hi)

    model = cp_model.CpModel()  # use CP-SAT solver

    # variables
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]  # will be a flag whether row i is selected or not

    # will match difference between the sum of selected rows and the target for each column
    d = [model.NewIntVar(-int(2 * n), int(2 * n), f"d_{i}") for i, n in enumerate(n_positive_samples)]

    a = [model.NewIntVar(0, int(2 * n), f"a_{i}") for i, n in enumerate(n_positive_samples)]  # absolute value of d
    # constraints

    # sum of each column
    sums = [
        cp_model.LinearExpr.Sum([rows[i, j] * x[i] for i in range(n)]) for j in range(m)
    ]  # sum of selected rows for each column

    for j in range(m):
        if j < n_strict:  # for the first n_strict columns, the sum must be within the range
            model.Add(sums[j] >= n_positive_thres_lo[j])
            model.Add(sums[j] <= n_positive_thres_hi[j])

        model.Add(sums[j] - n_positive_thres_best[j] == d[j])  # match the difference between the sum and the target
        model.AddMaxEquality(a[j], [d[j], -d[j]])  # match a to the absolute value of d, i.e. max (d, -d)

    # objective - min sum(abs(sum(rows[i,j] * x[i] for i in range(n))-n_positive_thres_best[j]) for j in range(m))
    # not function abs cant be used

    weighted_a = [a[j] * (strict_weight if j < n_strict else non_strict_weight) for j in range(m)]

    obj = cp_model.LinearExpr.Sum(weighted_a)  # sum of absolute values of differences
    model.Minimize(obj)  # minimize the sum of absolute values of differences

    solver = cp_model.CpSolver()

    # configure # cores
    solver.parameters.num_search_workers = 8

    # configure time limit

    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)
    # Compare on the status name (str) rather than the enum value: ortools ships type
    # info on some platforms and not others, so `status == cp_model.OPTIMAL` trips
    # mypy's strict-equality check where the enum is typed. StatusName is always str.
    status_name = solver.StatusName(status)

    selected_rows = []
    if status_name == "OPTIMAL":
        if v:
            print("Objective value =", solver.ObjectiveValue())
        for i in range(n):
            if solver.Value(x[i]):
                if v == 2:
                    print(i, rows[i])
                selected_rows.append(i)
    elif status_name == "INFEASIBLE":
        print("No solution found")
        return None

    elif status_name == "UNKNOWN":
        print("Time limit reached")
        return None

    elif status_name == "FEASIBLE":
        if v:
            print("Feasible solution found")
            print("Objective value =", solver.ObjectiveValue())
        for i in range(n):
            if solver.Value(x[i]):
                if v == 2:
                    print(rows[i])
                selected_rows.append(i)

    if v:
        print(f"selected {len(selected_rows)} out of {len(rows)} rows")

        selected_rows_sum = rows[selected_rows].sum(axis=0)

        for j in range(m):
            pct = selected_rows_sum[j] / n_positive_samples[j] * 100 if n_positive_samples[j] else 0.0
            print(
                f"{n_positive_thres_lo[j]:>3} <= {selected_rows_sum[j]:>3} ?= {n_positive_thres_best[j]:>3} <= {n_positive_thres_hi[j]:>3} of {n_positive_samples[j]:>3}  ({pct:.1f}%)"
            )

    return selected_rows


def train_test_split(
    rows: np.ndarray | list[list],
    test_frac: float = 0.2,
    test_prec: float = 0.15,
    v: int = 1,
    n_strict: int | None = None,
    non_strict_weight: int | float = 1,
) -> tuple[list[int] | None, list[int] | None]:
    """
    Split rows into train and test sets, taking ~ `test_frac` positive samples in the test set

    Args:
        rows: array of rows, with numbers,
        test_frac: desired fraction of total sum in each column to be selectes
        test_prec: precision of the fraction
        v: verbose, print suff on the way which is not critical messages
        n_strict: if not None, use n_strict first columns as target columns,
        the rest only contribute to the optimization objective sum term
        non_strict_weight: weight of non-strict columns in the optimization objective

    Returns:
        train_rows, test_rows: two lists of indexes of rows for train and test sets
    """
    rows = np.array(rows)
    n = rows.shape[0]
    m = rows.shape[1]

    if v:
        print(f"Total {n} rows, {m} columns")

    selected_rows = get_row_selection(
        rows,
        label_frac=1 - test_frac,
        label_prec=test_prec,
        v=v,
        n_strict=n_strict,
        non_strict_weight=non_strict_weight,
    )

    if selected_rows is None:
        return None, None

    selected_set = set(selected_rows)
    all_set = set(range(n))
    test_rows = list(all_set - selected_set)
    train_rows = list(selected_set)

    if v:
        print(f"Train rows: {len(train_rows)}, Test rows: {len(test_rows)}")

    return train_rows, test_rows
