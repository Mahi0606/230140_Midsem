# Data Directory — partB

## Dataset Used

**Name:** `make_moons` (scikit-learn built-in)

**Source:** `sklearn.datasets.make_moons`  
**No download required.** The dataset is generated programmatically.

## How to Regenerate

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_raw, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X = StandardScaler().fit_transform(X_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
```

## Files in This Directory

| File | Description |
|------|-------------|
| `X_train.npy` | Training features (280 × 2), StandardScaler normalised |
| `X_test.npy` | Test features (120 × 2), StandardScaler normalised |
| `y_train.npy` | Training labels (280,), binary {0, 1} |
| `y_test.npy` | Test labels (120,), binary {0, 1} |
| `results_summary.json` | Accuracy/F1 summary from all experiments |

## How Datasets are Used

- **task_2_1.ipynb:** Dataset introduction and visualisation
- **task_2_2.ipynb:** iSVM training and decision boundary reproduction
- **task_2_3.ipynb:** Comparison against paper's Table 1 results
- **task_3_1.ipynb:** Ablation study (same dataset)
- **task_3_2.ipynb:** Failure mode uses `make_blobs` (also generated inline, no files needed)

## Why make_moons?

The `make_moons` dataset has two interleaving crescent-shaped clusters — structurally analogous to the two-Gaussian-component binary classification problem in Setting 1 and Figure 1 of the paper. It has nonlinear class boundaries that require an RBF kernel (demonstrating the kernel ablation) and natural cluster structure (demonstrating the DP mixture benefit).
