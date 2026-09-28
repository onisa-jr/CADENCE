# CADENCE: Confidence-Adaptive Dual-Expert Network for Time Series Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: UCR 109](https://img.shields.io/badge/UCR%20Archive-109%20Datasets-success.svg)](https://www.timeseriesclassification.com)
[![Grand Mean: 0.8864](https://img.shields.io/badge/Accuracy-0.8864%20(%232%20Global)-brightgreen.svg)]()
[![Compute: 17.5s](https://img.shields.io/badge/Speed-17.5s%20per%20dataset-orange.svg)]()

Official implementation of **CADENCE** (*Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification*).

**Author:** Onisa Mapunda (`onisajr@gmail.com`)

---

## Highlights

- **Decoupled Dual-Expert Architecture:** Harmonizes a closed-form *Convolutional Linear Expert* (10,000 MiniRocket dilated features with Woodbury L2 ridge classification) with a *Distributional Interval Expert* (competing Hydra kernels, Cornish-Fisher dyadic intervals on kinematics $X, \Delta X, \Delta^2 X$, FFT spectral quantiles, ExtraTrees, 1,851 features).
- **Confidence-Adaptive Meta-Router:** Employs an internal 30% validation partition with rare-class preservation to dynamically navigate three regimes:
  - **Regime I: Pure Branch B Dominance** ($\Delta < -0.08$): Skips Branch A at inference.
  - **Regime II: Competitive Confidence Blending** ($|\Delta| \le 0.08$): Smoothly weights probabilities via $w_A = \mathrm{clip}(0.5 + 2.0\Delta, 0.15, 0.85)$.
  - **Regime III: Pure Branch A Dominance** ($\Delta > 0.08$): Skips Branch B at inference.
- **Full 100% Training Refit:** Selected models are refit on all available training data prior to test evaluation.
- **Top-Tier Empirical Performance:** Evaluated across the 109 equal-length datasets of the UCR Time Series Archive over 30 official resamples (3,270 runs), CADENCE achieves a grand mean accuracy of **0.8864**.
  - **#2 Among Evaluated Classifiers** across the archive, within 0.31 percentage points of HIVE-COTE 2.0 (0.8895, $p_{\mathrm{Holm}} = 0.295$, no statistically significant difference).
  - Outperforms **Hydra+MultiRocket** (0.8818, +0.46 pp), **MultiRocket** (0.8797, +0.67 pp), **HIVE-COTE 1.0** (0.8786, +0.78 pp, $p=0.048$), and **TS-CHIEF** (0.8761, +1.03 pp, $p=0.035$).
- **CPU-Native Efficiency:** Average runtime is **17.53 seconds** per evaluation run on a standard CPU. Entire 3,270-run benchmark completed in **15.92 CPU hours** (~7.5 hours wall-clock via bidirectional multiprocessing).

---

## Quickstart

```python
import numpy as np
from aeon.datasets import load_classification
from cadence import CADENCEClassifier

# Load any UCR dataset (e.g., GunPoint)
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# Fit CADENCE with automatic confidence-adaptive routing
model = CADENCEClassifier(n_jobs=4, random_state=42)
model.fit(X_train, y_train)

# Predict class labels and probabilities
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
acc = model.score(X_test, y_test)

print(f"Accuracy:        {acc:.4f}")
print(f"Routing Info:    {model.get_routing_info()}")
```

---

## Repository Structure

```
├── cadence/                     # Core CADENCE package
│   ├── __init__.py             # Package exports
│   ├── branch_a.py             # Convolutional Linear Expert (MiniRocket + Ridge L2)
│   ├── branch_b.py             # Distributional Interval Expert (Hydra + MQ + FFT)
│   ├── classifier.py           # CADENCEClassifier (scikit-learn BaseEstimator)
│   ├── moment_quant.py         # Thread-safe Cornish-Fisher dyadic intervals (JIT)
│   ├── router.py               # Adaptive Meta-Router with safe validation split
│   └── spectral.py             # Real FFT energy bands & spectral quantiles
├── cadence.py                  # Convenience root module
├── model.py                    # Backward-compatibility alias
├── evaluate.py                 # High-throughput benchmark evaluation CLI
├── requirements.txt            # Dependency specifications
├── REPRODUCIBILITY.md          # Complete, step-by-step reproducibility guide
├── results/                    # Official benchmark results across all 109 UCR datasets
│   ├── <Dataset>_results.csv   # Per-dataset 30-resample evaluation logs
│   ├── summary_results.csv     # Master summary table (0.8864 grand mean)
│   └── full_routing_decisions_109.csv # Per-dataset routing regimes
└── paper/                      # Publication suite
    ├── build_paper_pdf.py      # Manuscript PDF compiler (6 pages, booktabs, vector)
    ├── build_paper_docx.py     # Manuscript DOCX compiler (publication Word document)
    ├── cadence_paper_preview.pdf # Generated PDF paper
    ├── cadence_paper.docx      # Generated DOCX paper
    ├── generate_benchmark_table.py # Table 1 generator with Wilcoxon-Holm tests
    ├── generate_figures.py     # Vector figures (Figures 1-6) generator
    ├── main.tex                # Formal LaTeX manuscript
    └── references.bib          # BibTeX bibliography
```

---

## Benchmarking & Reproducibility

For comprehensive reproduction instructions, see [`REPRODUCIBILITY.md`](file:///home/onisajr/Documents/TS_classfication/REPRODUCIBILITY.md).

### Run on a Single Dataset:
```bash
python evaluate.py --dataset GunPoint --resamples 30
```

### Run the Complete 109-Dataset Suite (Bidirectional Multiprocessing):
```bash
# Terminal 1 (Ascending order)
python evaluate.py --all --order asc --output_dir results

# Terminal 2 (Descending order)
python evaluate.py --all --order desc --output_dir results
```

### Reproduce Paper Table 1 (Statistical Rankings):
```bash
python paper/generate_benchmark_table.py
```

### Reproduce All Vector Figures:
```bash
python paper/generate_figures.py
```

### Compile Publication Paper:
```bash
# Compile PDF manuscript
python paper/build_paper_pdf.py

# Compile Word DOCX manuscript
python paper/build_paper_docx.py
```

---

## Citation

```bibtex
@article{mapunda2026cadence,
  title={CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification},
  author={Mapunda, Onisa},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
