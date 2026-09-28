# Reproducibility Guide for CADENCE

**CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification**  
*Author:* Onisa Mapunda (`onisajr@gmail.com`)

This guide provides step-by-step instructions to reproduce all empirical results, benchmark rankings, statistical tests, ablation studies, figures, and publication documents reported in the CADENCE paper.

---

## 1. System Requirements & Environment Setup

CADENCE is designed to be **100% CPU-native**. No GPU, CUDA, or specialized hardware accelerator is required.

- **Operating System:** Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+), macOS, or Windows 10/11.
- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13.
- **Hardware Profile:** 
  - Standard dual-core or quad-core x86_64 CPU (tested on an Intel Core i7-7600U @ 2.80GHz).
  - 8 GB RAM minimum (16 GB recommended for concurrent dataset runs).
- **Benchmark Runtimes:**
  - Average runtime per single dataset evaluation (1 seed): **17.53 seconds**.
  - Total benchmark suite (109 datasets × 30 resamples = 3,270 evaluations): **15.92 CPU hours** (~7.5 hours wall-clock time via bidirectional dual-runner execution).

### Installation

Clone the repository and install the verified dependencies:

```bash
git clone https://github.com/onisajr/CADENCE.git
cd CADENCE

# Recommended: create a virtual environment
python3 -m venv env
source env/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 2. Quickstart: 5-Line scikit-learn Usage

```python
import numpy as np
from aeon.datasets import load_classification
from cadence import CADENCEClassifier

# 1. Load an exemplary UCR dataset (e.g. GunPoint)
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# 2. Instantiate and fit CADENCE (automatic adaptive routing & full refit)
clf = CADENCEClassifier(n_jobs=4, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate accuracy and inspect routing decision
accuracy = clf.score(X_test, y_test)
routing = clf.get_routing_info()

print(f"Test Accuracy:    {accuracy:.4f}")
print(f"Selected Regime:  {routing['decision']} (Validation Margin Δ = {routing['diff']:+.4f})")
```

---

## 3. Reproducing Single-Dataset Experiments

To evaluate CADENCE on any individual UCR dataset over the 30 standard resamples:

```bash
# Evaluate on GunPoint across 30 seeds
python evaluate.py --dataset GunPoint --resamples 30

# Evaluate on InlineSkate (Regime I: Pure Branch B Dominance, +9.66 pp over Hydra+MultiRocket)
python evaluate.py --dataset InlineSkate --resamples 30

# Evaluate on PigAirwayPressure (Regime III: Pure Branch A Dominance, +9.86 pp gain)
python evaluate.py --dataset PigAirwayPressure --resamples 30

# Evaluate on Phoneme (Regime II: Competitive Confidence Blending)
python evaluate.py --dataset Phoneme --resamples 30
```

Results are saved to `results/<DatasetName>_results.csv` and progressively aggregated into `results/summary_results.csv`.

---

## 4. Reproducing the Full 109-Dataset Benchmark (3,270 Runs)

The official evaluation follows the 109 equal-length dataset protocol established by MiniRocket (Dempster et al., KDD 2021) and MultiRocket (Tan et al., DMKD 2022). Datasets are fetched automatically from the UCR Time Series Archive using `aeon`.

- **Seed 0:** The exact original UCR train/test split.
- **Seeds 1–29:** Stratified random resamples with preserved split sizes and strict zero train/test leakage.

### Single-Process Sequential Execution:
```bash
python evaluate.py --all --resamples 30 --output_dir results
```

### High-Throughput Bidirectional Execution (2x Wall-Clock Speedup):
You can run two concurrent runners simultaneously. Atomic file locking (`.lock`) and thread-safe master summary merging prevent work duplication:

```bash
# Terminal 1 (Forward order: ACSF1 -> Yoga)
python evaluate.py --all --order asc --output_dir results

# Terminal 2 (Reverse order: Yoga -> ACSF1)
python evaluate.py --all --order desc --output_dir results
```

---

## 5. Reproducing Table 1: Benchmark Rankings & Wilcoxon-Holm Tests

To compute the grand mean accuracy, standard deviations, mean ranks, pairwise win/tie/loss counts, and two-sided Wilcoxon signed-rank test p-values with Holm-Bonferroni correction across all 26 published benchmark baselines:

```bash
python paper/generate_benchmark_table.py
```

### Expected Output Summary:
| Rank | Classifier | Paradigm | Mean Acc ± Std | Mean Rank | W / T / L vs CADENCE | $p_{\mathrm{Holm}}$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| -- | *Oracle Ceiling (Post-hoc)* | Per-dataset Best | *0.9067 ± 0.1062* | -- | -- | -- |
| 1 | **HIVE-COTE 2.0** | Heterogeneous Ensemble | **0.8895 ± 0.1159** | 6.22 | 63 / 7 / 39 | 0.295 (n.s.) |
| 2 | **CADENCE (Ours)** | Adaptive Dual-Expert | **0.8864 ± 0.1120** | 8.26 | -- | -- |
| 3 | Hydra+MultiRocket | Convolutional Hybrid | 0.8818 ± 0.1218 | 7.67 | 46 / 7 / 56 | 1.000 |
| 4 | MultiRocket (100k) | Convolutional Pooling | 0.8800 ± 0.1218 | 8.03 | 48 / 7 / 54 | 1.000 |
| 5 | MultiRocket (50k default) | Convolutional Pooling | 0.8797 ± 0.1222 | 8.19 | 47 / 10 / 52 | 1.000 |
| 6 | HIVE-COTE 1.0 | Hierarchical Ensemble | 0.8786 ± 0.1228 | 10.80 | 59 / 7 / 43 | **0.048** (sig.) |
| 7 | TS-CHIEF | Metric / Tree Ensemble | 0.8761 ± 0.1281 | 10.52 | 61 / 7 / 41 | **0.035** (sig.) |
| 8 | MultiRocket (10k) | Convolutional Pooling | 0.8749 ± 0.1254 | 9.83 | 59 / 6 / 44 | **0.014** (sig.) |
| 9 | MiniRocket | Convolutional Transform | 0.8724 ± 0.1300 | 11.40 | 72 / 6 / 31 | **< 0.001** (sig.) |
| 10 | InceptionTime | Deep ConvNet Ensemble | 0.8721 ± 0.1285 | 11.95 | 67 / 5 / 37 | **0.050** (sig.) |
| 11 | Hydra | Competing Dilated Kernels | 0.8714 ± 0.1319 | 10.45 | 66 / 5 / 38 | **0.005** (sig.) |

---

## 6. Reproducing Table 2: Controlled Component Ablation

To evaluate specific components or isolate individual experts, run `evaluate.py` with the `--mode` flag:

```bash
# Branch A Only (MiniRocket + Ridge L2 Woodbury)
python evaluate.py --all --mode pure_a --output_dir results_ablation_branch_a

# Branch B Only (Hydra + MomentQuant + FFT + ExtraTrees)
python evaluate.py --all --mode pure_b --output_dir results_ablation_branch_b

# Fixed 50/50 Soft Blending (No Adaptive Validation Router)
python evaluate.py --all --mode fixed_blend --output_dir results_ablation_fixed_blend

# CADENCE Full System (Confidence-Adaptive Meta-Router)
python evaluate.py --all --mode adaptive --output_dir results
```

### Ablation Results Summary (109 Datasets, 30 Resamples):
| Configuration | Modified Component | Mean Accuracy | Delta to Full (pp) |
| :--- | :--- | :---: | :---: |
| Branch A Only | MiniRocket + Ridge (No Interval Expert) | 0.8724 | -1.40 pp |
| Branch B Only | Hydra + MQ + FFT + ExtraTrees (No Conv Expert) | 0.8729 | -1.35 pp |
| Fixed 50/50 Soft Blend | Naive Probability Averaging | 0.8741 | -1.23 pp |
| Static Routing ($K \ge 12$) | Class-Count Heuristic Threshold | 0.8813 | -0.51 pp |
| **CADENCE (Full System)** | **Confidence-Adaptive Meta-Router** | **0.8864** | **--** |

---

## 7. Reproducing All Paper Figures

All vector graphics (PDF) and high-resolution raster images (PNG) are generated directly from the underlying experimental CSV data:

```bash
python paper/generate_figures.py
```

Output figures in `paper/figures/`:
- `fig1_architecture.pdf`: Architecture, validation routing, full refit, and test inference flow.
- `fig2_routing.pdf`: Piecewise routing & dynamic blend weights as a function of validation margin $\Delta$.
- `fig3_scatter.pdf`: Pairwise scatter plots of CADENCE against Hydra+MultiRocket and HIVE-COTE 2.0.
- `fig4_pareto.pdf`: Empirical Pareto frontier (Accuracy vs. CPU runtime on log scale).
- `fig5_ablation.pdf`: Systematic component ablations.
- `fig6_cd.pdf`: Critical Difference diagram across all 109 datasets (Wilcoxon test with Holm correction, $\alpha=0.05$).

---

## 8. Compiling Paper Documents (PDF & DOCX)

### Compiling the PDF Manuscript:
```bash
python paper/build_paper_pdf.py
```
Outputs: `paper/cadence_paper_preview.pdf` (Exactly 6 pages, formatted with Computer Modern equations, monochrome academic booktabs tables, vector figures, and clickable reference links).

### Compiling the DOCX Manuscript:
```bash
python paper/build_paper_docx.py
```
Outputs: `paper/cadence_paper.docx` (Publication-grade Word document with full text, equation cards, figures, tables, and references).

---

## 9. File & Directory Structure

```
.
├── cadence/                     # Core Python package
│   ├── __init__.py             # Exports CADENCEClassifier and sub-modules
│   ├── branch_a.py             # Convolutional Linear Expert (MiniRocket + Ridge)
│   ├── branch_b.py             # Distributional Interval Expert (Hydra + MQ + FFT)
│   ├── classifier.py           # Unified CADENCEClassifier scikit-learn estimator
│   ├── moment_quant.py         # Thread-safe Cornish-Fisher dyadic intervals (JIT)
│   ├── router.py               # Confidence-adaptive meta-router & safe validation split
│   └── spectral.py             # Real FFT energy bands & spectral quantiles
├── cadence.py                  # Convenience root module
├── model.py                    # Backward-compatibility alias interface
├── evaluate.py                 # Universal evaluation benchmark CLI
├── requirements.txt            # Pinned dependency requirements
├── REPRODUCIBILITY.md          # Step-by-step reproducibility instructions
├── README.md                   # Project overview and documentation
├── results/                    # Canonical benchmark results for all 109 UCR datasets
│   ├── <Dataset>_results.csv   # Per-dataset 30-resample evaluation logs
│   ├── summary_results.csv     # Master summary table (0.8864 grand mean accuracy)
│   ├── full_routing_decisions_109.csv # Per-dataset routing regimes (Regimes I, II, III)
│   └── models_ranking_summary.csv     # Global benchmark rank comparisons
└── paper/                      # Publication suite
    ├── build_paper_pdf.py      # Publication PDF generator (ReportLab)
    ├── build_paper_docx.py     # Publication DOCX generator (python-docx)
    ├── cadence_paper_preview.pdf # Compiled 6-page paper preview PDF
    ├── cadence_paper.docx      # Compiled paper Word document
    ├── generate_benchmark_table.py # Table 1 generator & statistical test calculator
    ├── generate_figures.py     # Figures 1–6 vector generator
    ├── generate_equations.py   # High-resolution LaTeX equation card generator
    ├── main.tex                # Formal LaTeX manuscript source
    ├── references.bib          # BibTeX bibliography database
    └── figures/                # Vector PDFs, renders, and equation cards
```
