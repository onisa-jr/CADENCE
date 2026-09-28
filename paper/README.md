# CADENCE: Publication Package

**Title:** CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification  
**Author:** Onisa Pascal ([onisajr@gmail.com](mailto:onisajr@gmail.com))  
**Target Submission:** Top-tier Machine Learning / Data Mining Conference or Journal (e.g., KDD, NeurIPS, ICML, Data Mining and Knowledge Discovery, Machine Learning)

-

## 1. Directory Structure

```
paper/
├── main.tex                       # Primary LaTeX publication manuscript
├── references.bib                 # Verified BibTeX bibliography (23 peer-reviewed citations)
├── cadence_paper_preview.pdf      # Compiled high-resolution PDF preview
├── generate_paper_figures.py      # Python generator for publication figures
├── build_paper_pdf.py             # Automated ReportLab PDF compiler
├── figures/                       # Vector (PDF) and high-DPI (PNG) figures
│   ├── cadence_architecture.pdf
│   ├── cadence_architecture.png
│   ├── routing_mechanism.pdf
│   ├── routing_mechanism.png
│   ├── pairwise_scatter.pdf
│   ├── pairwise_scatter.png
│   ├── model_comparison.pdf
│   ├── model_comparison.png
│   ├── pareto_frontier.pdf
│   ├── pareto_frontier.png
│   ├── ablation.pdf
│   └── ablation.png
├── tables/                        # Modular LaTeX tables
│   ├── table_notation.tex         # Table 1: Notation & architectural definitions
│   ├── table_benchmarks.tex       # Table 2: Overall UCR 109-dataset world ranking
│   ├── table_breakthrough.tex     # Table 3: Selected dataset gains and deficits
│   ├── table_ablation.tex         # Table 4: Component ablation study
│   ├── table_complexity.tex       # Table 5: Theoretical & practical complexity
│   └── table_routing_sample.tex   # Table 6: Empirical validation routing decisions
└── supplementary/                 # Supplementary Material
    ├── supplementary.tex          # Supplementary documentation
    └── full_109_dataset_results.tex # Complete 109-dataset empirical accuracy table
```

-

## 2. Key Empirical Findings (109 Datasets $\times$ 30 Resamples = 3,270 Runs)

- **World Ranking:**
  1. *Maximum (Oracle Ceiling):* 0.9067
  2. **HIVE-COTE 2.0:** 0.8895 (4-component meta-ensemble)
  3. **CADENCE (Ours):** **0.8864**
  4. **Hydra + MultiRocket:** 0.8818
  5. MultiRocket (100k): 0.8800
  6. MultiRocket (standard): 0.8797
  7. HIVE-COTE 1.0: 0.8786
  8. TS-CHIEF: 0.8761
  9. MiniRocket: 0.8724
  10. InceptionTime: 0.8721
- **Computational Runtime:**
  - CADENCE average run time: **17.53 seconds**.
  - Total compute time for 3,270 seed runs: **15.92 CPU hours** (~7.5 wall-clock hours via bidirectional concurrent execution on an Intel Core i7-7600U dual-core CPU).
  - Trailing HIVE-COTE 2.0 by only 0.31% while running orders of magnitude faster without GPUs.

-

## 3. How to Compile the LaTeX Manuscript

If `pdflatex` or `latexmk` is available on your machine:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For the supplementary material:
```bash
cd paper/supplementary/
pdflatex supplementary.tex
```

To regenerate the standalone PDF preview via Python ReportLab:
```bash
python3 paper/build_paper_pdf.py
```

-

## 4. Reproducing the Experiments

To evaluate CADENCE across all 109 UCR datasets:
```bash
python3 evaluate.py --all --model_version v3 --order asc --output_dir results_v3
```
Or run bidirectional execution in two terminals:
```bash
# Terminal 1: Forward evaluation (ACSF1 -> Yoga)
python3 evaluate.py --all --model_version v3 --order asc --output_dir results_v3

# Terminal 2: Reverse evaluation (Yoga -> ACSF1)
python3 evaluate.py --all --model_version v3 --order desc --output_dir results_v3
```
Atomic lockfiles prevent duplicated work and merge completed datasets automatically.
