# Utilities

This directory contains helper modules and utility functions used across the repository.

## Modules

- `data_utils.py` - Data loading and preprocessing utilities
- `model_utils.py` - Model creation and management helpers
- `training_utils.py` - Training loop and evaluation utilities
- `visualization_utils.py` - Plotting and visualization helpers

## Usage

Import utilities in your notebooks or scripts:

```python
from utils.data_utils import load_dataset
from utils.visualization_utils import plot_training_curves
```

## Development

When adding new utilities:
1. Keep functions focused and single-purpose
2. Add docstrings with parameter descriptions
3. Include type hints where applicable
4. Add example usage in docstrings
5. Keep dependencies minimal

## Testing

Test utility functions before using in experiments:
```python
# Example test
from utils.data_utils import normalize_data
import numpy as np

data = np.random.randn(100, 10)
normalized = normalize_data(data)
assert normalized.mean() < 0.1  # Check normalization
```
