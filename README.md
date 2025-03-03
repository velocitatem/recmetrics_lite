# RecMetrics Lite

A streamlined library for recommender system evaluation metrics with minimal dependencies. This is a lightweight version of the recmetrics library, with compatibility for modern Python package versions.

## Installation

```bash
# Basic installation
pip install recmetrics-lite

# With plotting extras (for metrics_plot using plotly)
pip install recmetrics-lite[plots]
```

## Features

RecMetrics Lite provides metrics and plotting functions for evaluating recommender systems:

### Metrics

- Coverage and catalog coverage
- Personalization 
- Intra-list similarity
- Novelty
- MSE and RMSE
- Precision and recall
- Mean Average Recall at K (MAR@K)

### Visualization

- Long tail plots
- Coverage plots
- Personalization plots
- Intra-list similarity plots
- MAR@K and MAP@K plots
- Class separation plots
- ROC plots
- Precision-recall curves
- Metrics radar plots (requires plotly)

## Usage Example

```python
import numpy as np
import pandas as pd
from recmetricslite import coverage, personalization, recommender_precision

# Sample data
catalog = ['item1', 'item2', 'item3', 'item4', 'item5']
predicted = [['item1', 'item2'], ['item3', 'item4']]
actual = [['item1', 'item3'], ['item2', 'item4']]

# Calculate metrics
cov = coverage(predicted, catalog)
pers = personalization(predicted)
prec = recommender_precision(predicted, actual)

print(f"Coverage: {cov:.2f}")
print(f"Personalization: {pers:.2f}")
print(f"Precision: {prec:.2f}")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.