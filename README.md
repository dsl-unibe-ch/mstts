# MSTTS

**M**ulti-class **S**tratified **T**rain-**T**est **S**plitter — split a dataset into
train and test sets while preserving the per-column distribution of *many* target
classes at once.

Unlike a single-label stratified split, MSTTS balances **all** target columns
simultaneously by solving a constraint-optimization problem with Google OR-Tools
(CP-SAT). The split is deterministic: for the same data and parameters you always get
the same result. If no split satisfies the constraints, the solver says so instead of
returning a bad split.

## Installation

```bash
pip install mstts
```

Or install the latest development version straight from GitHub:

```bash
pip install git+ssh://git@github.com/dsl-unibe-ch/mstts.git
```

## Quick start

The package exposes two functions:

- `train_test_split(rows, ...)` — split row indices into train and test sets.
- `get_row_selection(rows, ...)` — lower level: pick a subset of rows that hits a target
  fraction per column. `train_test_split` is built on top of it.

`rows` is an `(n_samples, n_columns)` array-like of **non-negative integers** — usually
one-hot / multi-hot class indicators, but any non-negative counts work. Float arrays are
cast to `int`.

```python
import numpy as np
from mstts import train_test_split

# 8 samples, 4 binary class columns
rows = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 1],
])

train_idx, test_idx = train_test_split(rows, test_frac=0.25, test_prec=0.1)
# -> two lists of row indices; disjoint and together covering every row
```

Working from a `pandas.DataFrame`? Pass the class columns as an array and index back:

```python
class_cols = ["class1", "class2", "class3"]
train_idx, test_idx = train_test_split(df[class_cols].to_numpy(), test_frac=0.2)
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
```

If no feasible split exists (or the 60 s solver time limit is hit),
`train_test_split` returns `(None, None)` and `get_row_selection` returns `None`.

## Thresholds & allowances

You specify a **target fraction** and an **allowance** (half-width of the tolerated
window) per split.

For `train_test_split`, they are expressed in terms of the **test** set:

| Parameter   | Default | Meaning                                                        |
| ----------- | ------- | -------------------------------------------------------------- |
| `test_frac` | `0.2`   | Target fraction of each column's total to place in the test set. |
| `test_prec` | `0.15`  | Allowance: the test fraction may land anywhere in `test_frac ± test_prec`. |

For `get_row_selection`, the same idea applies to the **selected** subset:

| Parameter    | Default | Meaning                                                       |
| ------------ | ------- | ------------------------------------------------------------- |
| `label_frac` | `0.8`   | Target fraction of each column's total to include.            |
| `label_prec` | `0.15`  | Allowance: the included fraction may land in `label_frac ± label_prec`. |

(`train_test_split` simply calls `get_row_selection` with `label_frac = 1 - test_frac`
and `label_prec = test_prec`, selecting the train set and returning its complement as
the test set.)

So the defaults keep **65 %–95 % of every column in the train set** (ideal 80 %), i.e.
5 %–35 % in the test set. Within that window the solver aims for the exact ideal
(`frac × column_total`), not just anywhere in range — see weighting below.

**Integer-count caveat.** Bounds are enforced on integer counts, and
`int(total × frac)` truncates. For a column with a small total the achievable fraction
is coarse — a column whose total is `2` can only be split `0 % / 50 % / 100 %`, so an
`80 % ± 15 %` request is infeasible for it. Give each column enough positive samples, or
widen the allowance, for tight fractions to be reachable.

## Strict vs. priority columns, and weighting

Columns come in two kinds, controlled by `n_strict`:

- **Strict columns** — the first `n_strict` columns. Their included count is a **hard
  constraint**: it *must* fall inside the `frac ± prec` window. They are *also* pulled
  toward the ideal target by the objective.
- **Priority columns** (a.k.a. non-strict) — the remaining columns. They are **not**
  range-constrained, but they still contribute to the objective, so the solver tries to
  bring them close to their target when it can.

`n_strict=None` (the default) treats **all** columns as strict.

The solver minimizes a weighted sum of per-column deviations from the ideal target:

```
minimize  Σ_j  weight_j · | included_count_j − round(frac × total_j) |
```

`non_strict_weight` sets the **relative importance of priority columns vs. strict
columns** in that objective. Because CP-SAT needs integer coefficients, the value is
read in two modes:

| `non_strict_weight` | Strict weight | Priority weight | Effect                                                    |
| ------------------- | ------------- | --------------- | --------------------------------------------------------- |
| `1` (default, int)  | 1             | 1               | Equal importance.                                         |
| int `> 1` (e.g. `2`, `5`) | 1       | that int        | Priority columns matter **more** than strict ones' objective term. |
| float `1.0`         | 10            | 10              | Equal importance.                                         |
| float `< 1.0` (e.g. `0.5`, `0.1`) | 10  | `int(10 × value)` | Priority columns matter **less** than strict ones (down-weighting; `~0.1` is the practical minimum — below `0.05` it rounds to `0` and priority columns are ignored). |

In short: pass an **int ≥ 1** to make priority columns weigh as much as or more than
strict columns, or a **float in `(0, 1]`** to weigh them less. Strict columns always
keep their hard window regardless of the weight.

### Verbosity

Both functions take `v`: `0` = silent, `1` = summary (objective, per-column
achieved-vs-target report), `2` = full debug (also prints the raw thresholds).

## Tests

Run the suite with:

```bash
uv run --extra test pytest        # or simply: pytest
```

15 tests across three files:

**`tests/test_get_row_selection.py`** — the core selector:
- *simple* / *fraction* / *precision* — the selection hits the requested per-column
  fraction within the allowance for the default (80 %), a 50 %, and a tight
  (75 % ± 5 %) target.
- *n_strict* — with only the first columns strict, those stay inside the window while
  the rest are left free.
- *edge_cases* — all-zeros, all-ones, single-column, and single-row inputs all return a
  valid selection.
- *empty* — empty input raises `IndexError`.

**`tests/test_train_test_split.py`** — the same matrix of cases for the train/test
wrapper (default / fraction / precision sizes, `n_strict`, edge cases, empty input), and
additionally checks that the train and test index sets are **disjoint and cover every
row**.

**`tests/test_integration.py`** — end-to-end on larger, realistic data:
- *integration_flow* — a 100×10 correlated dataset; verifies the train and test class
  distributions match within 15 %.
- *imbalanced_data* — columns with 10 % / 50 % / 90 % prevalence; verifies the rare
  class still appears in the test set and the common class lands near the target
  fraction.
- *non_binary_data* — integer count values rather than 0/1, confirming the solver
  handles non-binary columns.

> **Note on the toy datasets.** Because bounds apply to integer counts (see the caveat
> above), the small unit-test fixtures are scaled up with `np.tile` so the requested
> fractions are actually achievable — a column whose total is `2` cannot be split
> `80 / 20`.

## License

Apache 2.0
