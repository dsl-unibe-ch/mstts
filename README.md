# MSTTS

Multi-class Stratified Train Test Spliter is a Python package that provides a way to split datasets into training and testing sets while preserving the distribution of multiple binary classes.
## Installation
while in develipment PyPI package is not available.
You can install it via pip directly form git:
```bash
pip install git+ssh://git@github.com:dsl-unibe-ch/mstts.git
```

## Usage
You choose target % if the training set, and the method will run constraint SP-SAT to
find the best split.
If best split exist, thy result will be same each time you run it.

You can have 2 types of columnts: strict and priority.
- strict columns: the distribution of these columns will be preserved withing margin of
  trainf_frac_window (ToDo - fix name)
- priority columns: the distribution of these columns will also weighted in the
  optimisation term but the selected fraction is not guaranteed to be within the margin.


```python
from mstts import MulticlassStratifiedTrainTestSpliter
import pandas as pd
import numpy as np
# Create a sample dataframe

data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'class1': np.random.randint(0, 2, size=100),
    'class2': np.random.randint(0, 2, size=100),
    'class3': np.random.randint(0, 2, size=100)
}
df = pd.DataFrame(data)

# Define feature columns and class columns
strict_cols = ['class1', 'class2', 'class3']
priority_cols = ['class4', 'class5']  # Assuming these columns exist in your dataframe


# Initialize the splitter
splitter = MulticlassStratifiedTrainTestSpliter(.....

```

## License
Apache 2.0
