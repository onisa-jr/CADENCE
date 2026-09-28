# CADENCE Publication Figures Suite

This directory contains the publication-grade vector figures for the paper:
**"CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification"**
Author: Onisa Pascal (`onisajr@gmail.com`)

All figures meet the visual and statistical standards of top-tier time-series classification literature (e.g., ROCKET, MiniRocket, Hydra, HIVE-COTE 2.0).

---

## 1. Generated Figure Files

| File | Type | Description |
|:---|:---:|:---|
| [`fig1_architecture.pdf`](fig1_architecture.pdf) | Vector PDF | End-to-end CADENCE system diagram: singleton-safe validation split, dual expert branches, meta-router, decision regimes, 100% full refit step, and test-time conditional pruning path. |
| [`fig1_architecture.tex`](fig1_architecture.tex) | TikZ LaTeX | Full standalone TikZ source code (`\documentclass[tikz]{standalone}`) with vector styling. |
| [`fig2_routing.pdf`](fig2_routing.pdf) | Vector PDF | Discontinuous effective prediction weights ($w_A, w_B$) vs. validation margin $\Delta \in [-0.25, 0.25]$ with $\tau = \pm 0.08$ step transitions and honest open/closed boundary markers. |
| [`fig3_scatter.pdf`](fig3_scatter.pdf) | Vector PDF | Two-panel pairwise scatter plots (CADENCE vs. Hydra+MultiRocket and CADENCE vs. HIVE-COTE 2.0) across all 109 UCR datasets with non-overlapping leader lines for top 5 wins and losses. |
| [`fig4_pareto.pdf`](fig4_pareto.pdf) | Vector PDF | Empirical Pareto frontier of classification accuracy vs. mean training runtime per dataset (log scale) demonstrating CADENCE's non-dominated efficiency (17.53 s). |
| [`fig5_ablation.pdf`](fig5_ablation.pdf) | Vector PDF | Systematic horizontal bar chart showing 7 ablation configurations with error bars ($\pm 1\sigma$ across 30 resamples), explicit axis break, and percentage point (`pp`) deltas. |
| [`fig6_cd.pdf`](fig6_cd.pdf) | Vector PDF | Critical Difference diagram following Demšar / Fawaz standards with **Rank 1 on the right**, Holm-corrected Wilcoxon signed-rank test ($\alpha = 0.05$), and highlighted CADENCE ranking. |

---

## 2. Input Data Sources & Reproducibility

Every plotted number is loaded directly from experimental data files without hardcoding:

1. **`MultiRocket_Hydra_Ensemble_abdulaziz.xlsx`**:
   - Master benchmark archive containing published 30-resample accuracy distributions across 109 UCR datasets for 25 state-of-the-art models (HIVE-COTE 2.0, Hydra+MultiRocket, TS-CHIEF, HIVE-COTE 1.0, MultiRocket, InceptionTime, MiniRocket, etc.).
2. **`results_v3/summary_results.csv`**:
   - Per-dataset empirical mean accuracies, standard deviations, and timings for CADENCE across all 109 datasets and 30 resamples ($109 \times 30 = 3,270$ runs).
3. **`results_v3/full_routing_decisions_109.csv`**:
   - Empirical routing distribution on the official benchmark splits: 82 / 109 datasets in Regime II (75.2%), 18 / 109 in Regime III (16.5%), and 9 / 109 in Regime I (8.3%).
4. **Published Cluster Timings (Middlehurst et al., 2021; Tan et al., 2022)**:
   - Total published training runtimes across 112 datasets normalized per dataset: TS-CHIEF (32,685 s), HC1 (13,730 s), HC2 (10,935 s), STC (3,725 s), InceptionTime (2,783 s), TDE (2,424 s), DrCIF (1,459 s), Arsenal (897 s), Rocket (91.6 s), MultiRocket 50k (8.45 s), MultiRocket 10k (2.35 s), MiniRocket (1.31 s), CADENCE (17.53 s).

---

## 3. Reproduction Command

To reproduce all 6 figures in vector PDF format, run:
```bash
/home/onisajr/envs/global/bin/python3 paper/generate_figures.py
```

---

## 4. LaTeX Publication Captions

### Figure 1: Architecture
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_architecture.pdf}
\caption{End-to-end architecture of CADENCE. Phase 1 (Training): Input series undergo a singleton-safe 30\% stratified validation split ($D_{\text{train}}$ only) to evaluate Branch A (MiniRocket linear expert) and Branch B (MomentQuant + ExtraTrees interval expert). The Meta-Router computes $\Delta = \text{Val}_A - \text{Val}_B$ against threshold $\tau = 0.08$. Selected expert models are refitted on 100\% of the training partition. Phase 2 (Test-Time Inference): Unseen query series are dynamically routed; dominant regimes (24.8\% of datasets) prune the redundant expert to accelerate test inference.}
\label{fig:architecture}
\end{figure*}
```

### Figure 2: Routing Characteristics
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_routing.pdf}
\caption{CADENCE routing characteristics as a function of validation margin $\Delta = \text{Val}_A - \text{Val}_B$ with threshold $\tau = 0.08$. Effective prediction weights $w_A$ (solid blue) and $w_B$ (dashed orange) exhibit step discontinuities at $\pm \tau$ where dominance triggers exclusive routing (Regimes I and III), while competitive datasets undergo bounded linear blending (Regime II). Boundary jumps are marked with open and closed markers.}
\label{fig:routing}
\end{figure}
```

### Figure 3: Pairwise Scatter
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_scatter.pdf}
\caption{Pairwise classification accuracy across 109 UCR datasets (30 resamples per dataset). (a) CADENCE vs. Hydra+MultiRocket (46 wins, 7 ties, 56 losses, mean gain $+0.46$ pp). (b) CADENCE vs. HIVE-COTE 2.0 (39 wins, 7 ties, 63 losses, mean difference $-0.32$ pp, two-sided Wilcoxon $p = 0.074, p_{\text{Holm}} = 0.295$, not statistically significant). Top 5 wins and deficits are annotated with leader lines.}
\label{fig:pairwise_scatter}
\end{figure*}
```

### Figure 4: Pareto Frontier
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_pareto.pdf}
\caption{Empirical Pareto frontier of classification accuracy versus mean training runtime per dataset (logarithmic scale) across 109 UCR datasets (30 resamples). Non-dominated classifiers are joined by the solid green frontier line. CADENCE (17.53 s, 88.64\%) delivers near meta-ensemble accuracy at over $620\times$ faster training time than HIVE-COTE 2.0 (10,935 s). TIMING NOTE: CADENCE evaluated on a dual-core Intel Core i7-7600U laptop CPU; baseline runtimes normalized from 112-dataset cluster totals in Middlehurst et al. (2021).}
\label{fig:pareto}
\end{figure}
```

### Figure 5: Systematic Ablation
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_ablation.pdf}
\caption{Systematic component ablation across 109 UCR datasets (30 stratified resamples). Bars represent mean classification accuracy starting from a labeled axis break at 0.850; error bars denote $\pm 1\sigma$ across resamples. Deltas are reported in percentage points (pp) relative to the full CADENCE pipeline ($0.8864$).}
\label{fig:ablation}
\end{figure}
```

### Figure 6: Critical Difference Diagram
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_cd.pdf}
\caption{Critical Difference diagram of mean ranks across 109 UCR datasets for 13 leading classifiers, following the Demšar / Fawaz convention with Rank 1 on the right. Statistical significance determined via two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction ($\alpha = 0.05$). Horizontal crossbars connect cliques of classifiers between which no statistically significant difference is detected. CADENCE forms an elite non-significant clique with HIVE-COTE 2.0 and Hydra+MultiRocket while significantly outperforming MiniRocket ($p < 0.001$), TS-CHIEF ($p = 0.035$), and HIVE-COTE 1.0 ($p = 0.048$).}
\label{fig:cd}
\end{figure*}
```
