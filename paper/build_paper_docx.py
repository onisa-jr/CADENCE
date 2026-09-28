#!/usr/bin/env python3
"""
Publication-Grade DOCX Generator for CADENCE Manuscript.
Produces a clean, professionally formatted Microsoft Word document (.docx)
with:
- Full paper metadata (Title, Author: Onisa Mapunda, Email: onisajr@gmail.com).
- High-resolution figures (Architecture, Routing Regimes, CD diagram, Pairwise Scatter, Pareto & Ablation).
- Vector-crisp Computer Modern equation cards (Equations 1 to 6) with explicit term definitions.
- Academic booktabs-style tables (Table 1 Benchmark Rankings & Table 2 Component Ablation).
- Running headers, page-numbered footers, and academic hanging-indent references.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUTPUT_DOCX = "paper/cadence_paper.docx"

def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    """Set inner padding for table cells in dxa (twips)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_booktabs_borders(table, num_header_rows=1):
    """Applies clean academic booktabs styling: top rule, header bottom rule, bottom rule; no vertical lines."""
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    
    # Disable vertical and inner horizontal borders
    for b in ['left', 'right', 'insideH', 'insideV']:
        el = OxmlElement(f'w:{b}')
        el.set(qn('w:val'), 'none')
        tblBorders.append(el)
        
    # Top rule (1.2 pt)
    top = OxmlElement('w:top')
    top.set(qn('w:val'), 'single')
    top.set(qn('w:sz'), '10')
    top.set(qn('w:color'), '0F172A')
    tblBorders.append(top)
    
    # Bottom rule (1.2 pt)
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '10')
    bottom.set(qn('w:color'), '0F172A')
    tblBorders.append(bottom)
    tblPr.append(tblBorders)
    
    # Header bottom rule (0.6 pt)
    for r_idx in range(num_header_rows):
        for cell in table.rows[r_idx].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = tcPr.first_child_found_in('w:tcBorders')
            if tcBorders is None:
                tcBorders = OxmlElement('w:tcBorders')
                tcPr.append(tcBorders)
            bot = OxmlElement('w:bottom')
            bot.set(qn('w:val'), 'single')
            bot.set(qn('w:sz'), '6')
            bot.set(qn('w:color'), '0F172A')
            tcBorders.append(bot)

def add_header_footer(doc):
    """Configures running header and page numbering."""
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    
    # Running header (pages 2+)
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hrun = hp.add_run("CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate TSC")
    hrun.font.name = "Times New Roman"
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = RGBColor(100, 116, 139)
    
    # Running footer (all pages)
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    frun = fp.add_run("Page ")
    frun.font.name = "Times New Roman"
    frun.font.size = Pt(9)
    frun.font.color.rgb = RGBColor(100, 116, 139)
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    fp._p.append(fldSimple)

def build_docx():
    doc = docx.Document()
    
    # Set standard 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    add_header_footer(doc)
    
    # Setup standard font
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(10.5)
    font.color.rgb = RGBColor(30, 41, 59)
    
    def add_p(text="", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=6, bold=False, italic=False, font_size=10.5, color=RGBColor(30, 41, 59)):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if text:
            run = p.add_run(text)
            run.bold = bold
            run.italic = italic
            run.font.name = 'Times New Roman'
            run.font.size = Pt(font_size)
            run.font.color.rgb = color
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_bullet(lead_bold, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_lead = p.add_run(lead_bold + " ")
        r_lead.bold = True
        r_lead.font.name = 'Times New Roman'
        r_lead.font.size = Pt(10)
        r_lead.font.color.rgb = RGBColor(30, 41, 59)
        r_txt = p.add_run(text)
        r_txt.font.name = 'Times New Roman'
        r_txt.font.size = Pt(10)
        r_txt.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(9)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(100, 116, 139)
        run.italic = True
        return p

    def add_where_clause(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r_lead = p.add_run("where: ")
        r_lead.bold = True
        r_lead.font.name = 'Times New Roman'
        r_lead.font.size = Pt(9.2)
        r_lead.font.color.rgb = RGBColor(71, 85, 105)
        r_body = p.add_run(text)
        r_body.font.name = 'Times New Roman'
        r_body.font.size = Pt(9.2)
        r_body.font.color.rgb = RGBColor(71, 85, 105)
        return p

    # -------------------------------------------------------------
    # TITLE, AUTHOR & METADATA
    # -------------------------------------------------------------
    add_p("CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification",
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=8, bold=True, font_size=18, color=RGBColor(15, 23, 42))
    
    add_p("Onisa Mapunda", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2, bold=True, font_size=12, color=RGBColor(30, 41, 59))
    add_p("onisajr@gmail.com", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14, italic=True, font_size=10, color=RGBColor(37, 99, 235))
    
    # -------------------------------------------------------------
    # ABSTRACT & KEYWORDS
    # -------------------------------------------------------------
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.25)
    p_abs.paragraph_format.right_indent = Inches(0.25)
    p_abs.paragraph_format.space_before = Pt(4)
    p_abs.paragraph_format.space_after = Pt(8)
    p_abs.paragraph_format.line_spacing = 1.15
    
    r_lead = p_abs.add_run("Abstract—")
    r_lead.bold = True
    r_lead.font.name = 'Times New Roman'
    r_lead.font.size = Pt(10)
    
    abs_text = (
        "Time series classification (TSC) has long been characterized by a sharp trade-off between "
        "classification accuracy and computational scalability. Heterogeneous meta-ensembles like HIVE-COTE 2.0 [5] "
        "achieve state-of-the-art accuracy by combining representations across temporal, frequency, shapelet, and dictionary "
        "domains, but require days or weeks of compute. Conversely, ultra-fast random convolutional transforms such as "
        "MiniRocket [2] and Hydra [3] provide orders-of-magnitude speedups, but struggle with phase-independent statistical "
        "distributions, kinematic transitions, and catastrophic decision tree fragmentation on datasets with large class counts "
        "when combined with tree heads. Naive feature concatenation or static model averaging fails to resolve this tension, "
        "often inducing negative transfer and parameter dilution.\n\n"
        "In this work, we present CADENCE (Confidence-Adaptive Dual-Expert Network for time series Classification Excellence), "
        "a unified, CPU-native dual-expert architecture. CADENCE decouples representation learning into two specialized pathways: "
        "(i) a Convolutional Linear Expert pairing 10,000 deterministic dilated features with closed-form L2-regularized Woodbury "
        "ridge classification, and (ii) a Distributional Interval Expert pairing competing dilated kernels (Hydra) with thread-safe "
        "dyadic Cornish-Fisher moment approximations across signal kinematics (X, ΔX, Δ²X) and FFT spectral energy bands, fitted with "
        "an entropy-based ExtraTrees ensemble. To mediate between these paradigms, CADENCE incorporates an internal validation meta-router "
        "with rare-class preservation that dynamically selects between pure expert routing and confidence-weighted soft blending, followed "
        "by a full refit on 100% of training data. Evaluated across all 109 datasets of the standard equal-length UCR Time Series Archive [6] "
        "over 30 resamples (3,270 total evaluations), CADENCE achieves a grand mean accuracy of 0.8864. This ranks #2 among evaluated "
        "classifiers across the archive, surpassed only by HIVE-COTE 2.0 (0.8895, p_Holm = 0.295, no statistically significant difference), "
        "and ahead of Hydra+MultiRocket (0.8818, +0.46 percentage points, p_Holm = 1.000), MultiRocket [4] (0.8797, +0.67 percentage points), "
        "and HIVE-COTE 1.0 [6b] (0.8786, +0.78 percentage points, p_Holm = 0.048, statistically significant), closing the gap to HIVE-COTE 2.0 "
        "to just 0.31 percentage points while completing an evaluation run in an average of only 17.53 seconds on a standard dual-core CPU."
    )
    r_body = p_abs.add_run(abs_text)
    r_body.font.name = 'Times New Roman'
    r_body.font.size = Pt(10)
    
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.left_indent = Inches(0.25)
    p_kw.paragraph_format.right_indent = Inches(0.25)
    p_kw.paragraph_format.space_before = Pt(2)
    p_kw.paragraph_format.space_after = Pt(14)
    r_kw_title = p_kw.add_run("Keywords—")
    r_kw_title.bold = True
    r_kw_title.font.name = 'Times New Roman'
    r_kw_title.font.size = Pt(9.5)
    r_kw_body = p_kw.add_run("Time series classification, dual-expert routing, MiniRocket, Hydra, Cornish-Fisher expansion, confidence blending, UCR archive, Wilcoxon signed-rank test.")
    r_kw_body.italic = True
    r_kw_body.font.name = 'Times New Roman'
    r_kw_body.font.size = Pt(9.5)

    # -------------------------------------------------------------
    # 1. INTRODUCTION
    # -------------------------------------------------------------
    add_h1("1. Introduction")
    add_p(
        "Time series classification (TSC) is a core problem in applied machine learning, underlying critical applications in "
        "electrocardiology [6], industrial anomaly detection, seismology, wearable motion analytics, and acoustic speech processing [7]. "
        "The 109 univariate, equal-length datasets of the UCR Time Series Archive [6] exhibit tremendous structural diversity: series lengths "
        "range from dozens to thousands of time points, sample sizes vary from tens to thousands of instances, and class cardinalities span binary "
        "discrimination (K=2) to fine-grained categorization (K=60)."
    )
    add_p(
        "Historically, achieving top accuracy demanded heavy heterogeneous meta-ensembles, most notably HIVE-COTE 2.0 (HC2) [5], which combines "
        "STC, TDE, DrCIF, and Arsenal. While HC2 achieves an unmatched published benchmark accuracy of 0.8895, its computational cost is substantial, "
        "requiring an average of approximately 3 hours per dataset (over 340 CPU hours across the 112 archive datasets)."
    )
    add_p(
        "In response, the field shifted toward randomized convolutional transforms, initiated by ROCKET [1], refined by MiniRocket [2], "
        "MultiRocket [4], and Hydra [3]. By projecting time series into high-dimensional linear spaces via dilated convolutions pooled with "
        "Proportion of Positive Values (PPV), these methods run in seconds. However, single-model convolutional transforms suffer from key failure "
        "modes: (1) inability to capture phase-free statistical distributions, velocity, and acceleration kinematics; (2) severe decision tree "
        "fragmentation on datasets with large class counts (K ≥ 12) when tree heads are naively applied; and (3) negative transfer when concatenating "
        "heterogeneous feature spaces into a single linear regularizer."
    )
    
    add_h2("1.1 Contributions")
    add_bullet("Decoupled Dual-Expert Architecture:", "Harmonizes a closed-form convolutional linear expert (10,000 features) with an interval-distributional expert (Hydra, Cornish-Fisher moments on kinematics, FFT spectral quantiles, ExtraTrees, 1,851 features).")
    add_bullet("Confidence-Adaptive Meta-Router with Full Refit:", "Introduces an internal validation router with rare-class preservation navigating three distinct regimes: Pure Branch A, Pure Branch B, and Smooth Confidence Blending, refitting models on 100% of training data.")
    add_bullet("Conditional Inference-Time Branch Pruning:", "Completely skips the unselected branch during dominance regimes (24.8% of datasets), reducing test-time feature extraction overhead.")
    add_bullet("Exhaustive Benchmark Evaluation (3,270 Runs):", "Evaluated on all 109 UCR datasets across 30 resamples, attaining 0.8864 grand mean accuracy (#2 among evaluated classifiers across the archive).")
    add_bullet("Rigorous Statistical Testing:", "Reports two-sided Wilcoxon signed-rank test p-values with Holm-Bonferroni correction, mean ranks, and per-dataset win/tie/loss counts against 25 published benchmark models.")

    # Figure 1: Architecture
    fig1_path = "paper/Figure 1: Architectural overview of the CADENCE dual-expert classifier, validation routing, full refit, and test inference..png"
    if not os.path.exists(fig1_path):
        fig1_path = "paper/figures/fig1_architecture_render-1.png"
    if os.path.exists(fig1_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(fig1_path, width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_caption("Figure 1: Architectural overview of the CADENCE dual-expert classifier, validation routing, full refit, and test inference.")

    # -------------------------------------------------------------
    # 2. RELATED WORK
    # -------------------------------------------------------------
    add_h1("2. Related Work")
    add_p(
        "2.1 Classical TSC: Distance-based methods (1NN-DTW) [7] offer robust elastic alignment but suffer from quadratic complexity O(N L²). "
        "Dictionary methods (BOSS, WEASEL) discretize windows into symbolic Fourier words. Interval ensembles (TSF, CIF, DrCIF) extract summary "
        "statistics across intervals."
    )
    add_p(
        "2.2 Random Convolutions: ROCKET [1] demonstrated that random kernels pooled with PPV produce linearly separable features. MiniRocket [2] "
        "introduced deterministic integer kernels and precomputed dilations, while Hydra [3] introduced competing kernel groups. MultiRocket [4] "
        "added first-order differences and four pooling operators."
    )
    add_p(
        "2.3 Ensembles & Meta-Ensembles: HIVE-COTE 2.0 [5] combines 4 distinct classifiers via CAWPE meta-weighting. TS-CHIEF [8] integrates "
        "dictionary, interval, and distance criteria into trees. InceptionTime [8b] adapts deep residual networks for time series. "
        "While highly accurate, these meta-ensembles require substantial cluster compute. In contrast, CADENCE introduces an internal validation "
        "meta-router to conditionally select or blend experts on a lightweight CPU budget."
    )

    # -------------------------------------------------------------
    # 3. METHODOLOGY
    # -------------------------------------------------------------
    add_h1("3. The CADENCE Methodology")
    add_p(
        "3.1 Branch A: Convolutional Linear Expert: MiniRocket [2] extracts D_A = 10,000 features using length-9 integer kernels "
        "ω ∈ {-1, 2}^9 (84 unique configurations with 3 weights equal to 2 and 6 weights equal to -1) and Proportion of Positive Values (PPV) pooling:"
    )
    
    # Equation 1
    if os.path.exists("paper/figures/equations/eq1.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq1.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("At is convolution activation at step t; X is the series; *d denotes convolution with dilation d; ω ∈ {-1, 2}^9 is the length-9 kernel; φ_PPV is PPV pooling; b is quantile bias; and 1(·) is the indicator function.")

    add_p(
        "Standardized features ~F_A are classified via multiclass Ridge regression with closed-form L2 regularization across 15 log-spaced candidate "
        "alphas in [10^-3, 10^4]. When N < D_A, the dual Woodbury matrix identity is evaluated:"
    )
    
    # Equation 2
    if os.path.exists("paper/figures/equations/eq2.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq2.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("W* is the closed-form weight matrix; ~F_A is standardized features; Y is one-vs-rest labels; α is the L2 penalty; and I is the identity matrix evaluated in primal (N ≥ D_A) or dual Woodbury space (N < D_A).")

    add_p("Posterior class probabilities are derived via softmax over raw decision values:")
    
    # Equation 3
    if os.path.exists("paper/figures/equations/eq3.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq3.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("P_A(y = k | X) is the posterior class probability; f_A,k(X) is linear scoring output; W* is fitted weights; and b_A is the class bias vector.")

    add_p(
        "3.2 Branch B: Distributional Interval Expert: Combines 16 groups of 8 competing Hydra kernels (128 features) [3], "
        "dyadic intervals up to depth 5 with Cornish-Fisher [9] quantile approximations across raw series (X), velocity (ΔX), and "
        "acceleration (Δ²X) (1,701 features), and real FFT spectral quantiles (22 features). Total feature dimension D_B = 1,851. "
        "The Cornish-Fisher expansion estimates arbitrary quantiles q_α in linear time O(M) without sorting:"
    )
    
    # Equation 4
    if os.path.exists("paper/figures/equations/eq4.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq4.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("q_α(S) is estimated quantile at level α; μ and σ are segment mean and standard deviation; z_α = Φ^-1(α) is standard normal quantile; S is sample skewness; K is sample kurtosis; and Ψ(·) is the asymptotic polynomial expansion.")

    add_p(
        "Features are fitted with 100 Extremely Randomized Trees (ExtraTrees) [10] using Shannon entropy and 10% random feature subsampling per split."
    )
    add_p(
        "3.3 Confidence-Adaptive Meta-Router: Partitions training data into an internal 30% held-out validation set while safely preserving "
        "singleton classes. Validation accuracies Val_A and Val_B are evaluated on held-out data. The routing decision is governed by "
        "empirical difference Δ = Val_A - Val_B with threshold τ = 0.08:"
    )
    
    # Equation 5
    if os.path.exists("paper/figures/equations/eq5.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq5.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("y*(X) is the predicted class label; Δ = Val_A - Val_B is empirical validation accuracy margin; τ = 0.08 is regime threshold; P_A and P_B are expert posteriors; and P_blend is the blended probability vector.")

    add_p("where in the competitive regime, predictions are smoothly blended with weight w_A:")
    
    # Equation 6
    if os.path.exists("paper/figures/equations/eq6.png"):
        p_eq = doc.add_paragraph()
        p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_eq.paragraph_format.space_before = Pt(4)
        p_eq.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/equations/eq6.png", width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_where_clause("P_blend is the blended probability vector; w_A is dynamic weight assigned to Branch A; (1 - w_A) is weight assigned to Branch B; and clip(·, 0.15, 0.85) enforces robust probability bounds.")

    # Figure 2: Routing Regimes
    if os.path.exists("paper/figures/fig2_routing_render-1.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/fig2_routing_render-1.png", width=Inches(5.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_caption("Figure 2: Piecewise routing and blending regimes as a function of validation margin Δ with step transitions.")

    # -------------------------------------------------------------
    # 4. RESULTS & BENCHMARKS
    # -------------------------------------------------------------
    add_h1("4. Experimental Results and Benchmark Rankings")
    add_p(
        "Following the 109-dataset, 30-resample evaluation protocol of MiniRocket [2] and MultiRocket [4], CADENCE was evaluated "
        "across the 109 equal-length datasets of the UCR Time Series Archive [6] over the 30 official resamples (3,270 total evaluations). "
        "Table 1 reports the grand mean accuracy against published state-of-the-art benchmarks."
    )

    # Table 1: Benchmark Rankings
    table_rows = [
        ["Rank", "Classifier", "Algorithmic Paradigm", "Mean Acc ± Std", "Mean Rank", "W / T / L vs Ours", "p_Holm"],
        ["--", "Oracle Ceiling (Post-hoc)", "Per-dataset Best", "0.9067 ± 0.1062", "--", "--", "--"],
        ["1", "HIVE-COTE 2.0", "Heterogeneous Meta-Ensemble", "0.8895 ± 0.1159", "6.22", "63 / 7 / 39", "0.295"],
        ["2", "CADENCE (Ours)", "Adaptive Dual-Expert", "0.8864 ± 0.1120", "8.26", "--", "--"],
        ["3", "Hydra+MultiRocket", "Convolutional Hybrid", "0.8818 ± 0.1218", "7.67", "46 / 7 / 56", "1.000"],
        ["4", "MultiRocket (100k)", "Convolutional Pooling", "0.8800 ± 0.1218", "8.03", "48 / 7 / 54", "1.000"],
        ["5", "MultiRocket (50k default)", "Convolutional Pooling", "0.8797 ± 0.1222", "8.19", "47 / 10 / 52", "1.000"],
        ["6", "HIVE-COTE 1.0", "Hierarchical Meta-Ensemble", "0.8786 ± 0.1228", "10.80", "59 / 7 / 43", "0.048"],
        ["7", "TS-CHIEF", "Metric / Tree Ensemble", "0.8761 ± 0.1281", "10.52", "61 / 7 / 41", "0.035"],
        ["8", "MultiRocket (10k)", "Convolutional Pooling", "0.8749 ± 0.1254", "9.83", "59 / 6 / 44", "0.014"],
        ["9", "MiniRocket", "Convolutional Transform", "0.8724 ± 0.1300", "11.40", "72 / 6 / 31", "< 0.001"],
        ["10", "InceptionTime", "Deep ConvNet Ensemble", "0.8721 ± 0.1285", "11.95", "67 / 5 / 37", "0.050"],
        ["11", "Hydra", "Competing Dilated Kernels", "0.8714 ± 0.1319", "10.45", "66 / 5 / 38", "0.005"],
        ["12-26", "ROCKET, Arsenal, DrCIF, TDE, STC, CIF, WEASEL, etc.", "Classical & Interval Baselines", "0.8645 - 0.7954", "11.76 - 21.28", "≥ 76 wins", "< 0.001"]
    ]

    t1 = doc.add_table(rows=len(table_rows), cols=7)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_booktabs_borders(t1, num_header_rows=1)

    col_widths = [Inches(0.5), Inches(1.8), Inches(1.8), Inches(1.3), Inches(0.8), Inches(0.9), Inches(0.7)]
    for r_idx, row_vals in enumerate(table_rows):
        row = t1.rows[r_idx]
        is_header = (r_idx == 0)
        is_cadence = (row_vals[1].startswith("CADENCE"))
        is_hc2 = (row_vals[1].startswith("HIVE-COTE 2.0"))
        
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            
            # Alignments
            if is_header or c_idx in [0, 4, 5, 6]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif c_idx == 3:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            run = p.add_run(val)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.5 if not is_header else 8.8)
            if is_header or is_cadence or is_hc2:
                run.bold = True
            if r_idx == 1: # Oracle ceiling
                run.italic = True
                
    add_caption("Table 1: Benchmark ranking across the 109 UCR Archive datasets over 30 resamples (Wilcoxon tests with Holm correction).")

    # Figure 3: Critical Difference
    if os.path.exists("paper/figures/fig6_cd_render-1.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/fig6_cd_render-1.png", width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_caption("Figure 3: Critical Difference diagram of mean ranks across 109 UCR datasets (Wilcoxon signed-rank test with Holm correction, alpha = 0.05, Rank 1 on right).")

    # Figure 4: Pairwise Scatter
    if os.path.exists("paper/figures/fig3_scatter_render-1.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture("paper/figures/fig3_scatter_render-1.png", width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_caption("Figure 4: Pairwise scatter plots of CADENCE against Hydra+MultiRocket and HIVE-COTE 2.0 with readable callouts.")

    # -------------------------------------------------------------
    # 5. BREAKTHROUGHS & ABLATIONS
    # -------------------------------------------------------------
    add_h1("5. Breakthroughs and Component Ablation")
    add_p(
        "CADENCE achieves decisive wins on specialized sensor and kinematic datasets over Hydra+MultiRocket: "
        "SemgHandMovementCh2 (0.8713 vs 0.7717, +9.96 pp), PigAirwayPressure (0.8220 vs 0.7234, +9.86 pp), "
        "InlineSkate (0.6047 vs 0.5081, +9.66 pp), and DistalPhalanxTW (0.7746 vs 0.6969, +7.77 pp). "
        "CADENCE also directly surpasses HIVE-COTE 2.0 on 39 datasets (e.g. DistalPhalanxTW +7.29 pp, InlineSkate +5.92 pp, Earthquakes +4.99 pp). "
        "On Seed 0 (the exact original UCR split), CADENCE obtains 0.7482 on Earthquakes (identical to majority class rate) and 0.7050 on DistalPhalanxTW."
    )

    # Table 2: Ablation Study
    ablation_rows = [
        ["Configuration", "Component Modified", "Mean Accuracy", "Delta to Full (pp)"],
        ["Branch A Only", "MiniRocket + Ridge (No Interval Expert)", "0.8724", "-1.40 pp"],
        ["Branch B Only", "Hydra + MQ + FFT + ExtraTrees (No Conv Expert)", "0.8729", "-1.35 pp"],
        ["Fixed 50/50 Soft Blend", "Naive Averaging (No Dynamic Router)", "0.8741", "-1.23 pp"],
        ["Static Routing (K ≥ 12)", "Class-Count Heuristic Threshold", "0.8813", "-0.51 pp"],
        ["CADENCE (Full System)", "Confidence-Adaptive Meta-Router", "0.8864", "--"]
    ]

    t2 = doc.add_table(rows=len(ablation_rows), cols=4)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_booktabs_borders(t2, num_header_rows=1)

    t2_widths = [Inches(1.8), Inches(2.8), Inches(1.2), Inches(1.2)]
    for r_idx, row_vals in enumerate(ablation_rows):
        row = t2.rows[r_idx]
        is_header = (r_idx == 0)
        is_full = (r_idx == len(ablation_rows) - 1)
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = t2_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            if is_header or c_idx in [2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            run = p.add_run(val)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.8)
            if is_header or is_full:
                run.bold = True
                
    add_caption("Table 2: Controlled component ablation across all 109 datasets (30 resamples). MQ denotes MomentQuant.")

    # Figure 5: Pareto & Ablation Plots
    if os.path.exists("paper/figures/fig4_pareto_render-1.png") and os.path.exists("paper/figures/fig5_ablation_render-1.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        
        # Side-by-side table for the two figures
        t_figs = doc.add_table(rows=1, cols=2)
        t_figs.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        cell_l = t_figs.rows[0].cells[0]
        cell_r = t_figs.rows[0].cells[1]
        cell_l.width = Inches(3.2)
        cell_r.width = Inches(3.2)
        
        p_l = cell_l.paragraphs[0]
        p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_l = p_l.add_run()
        r_l.add_picture("paper/figures/fig4_pareto_render-1.png", width=Inches(3.1))
        
        p_r = cell_r.paragraphs[0]
        p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_r = p_r.add_run()
        r_r.add_picture("paper/figures/fig5_ablation_render-1.png", width=Inches(3.1))
        
        add_caption("Figure 5: Left: Empirical Pareto frontier vs. cluster training runtime. Right: Systematic component ablations.")

    # -------------------------------------------------------------
    # 6. ROUTING ANALYSIS & EFFICIENCY
    # -------------------------------------------------------------
    add_h1("6. Routing Analysis and Computational Efficiency")
    add_p(
        "Exact Routing Distribution (N = 109 Datasets): Across all 109 evaluated datasets, 82 datasets (75.2%) trigger "
        "Regime II (Competitive Blending, e.g. Phoneme, ArrowHead, ECG200), 18 datasets (16.5%) trigger Regime III (Pure Branch A Dominance, "
        "e.g. PigAirwayPressure, PigCVP, Fish), and 9 datasets (8.3%) trigger Regime I (Pure Branch B Dominance, e.g. Semg, InlineSkate). "
        "In total, 27 datasets (24.8%) trigger dominance regimes where test inference skips one branch completely."
    )
    add_p(
        "Pareto Efficiency: CADENCE executes in an average of 17.53 seconds per seed run. The entire 3,270-run benchmark completed in "
        "15.92 CPU hours (~7.5 hours wall-clock time via bidirectional execution on an Intel Core i7-7600U). Published cluster runtimes "
        "report TS-CHIEF at 1,016.87h (~32,685s per dataset), HIVE-COTE 1.0 at 427.18h (~13,730s), and HIVE-COTE 2.0 at 340.21h (~10,935s). "
        "CADENCE establishes a superior Pareto frontier, nearing HC2 accuracy while running orders of magnitude faster."
    )

    # -------------------------------------------------------------
    # 7. DISCUSSION & CONCLUSION
    # -------------------------------------------------------------
    add_h1("7. Discussion, Limitations, and Conclusion")
    add_p(
        "Discussion & Limitations: CADENCE demonstrates that combining complementary representations through confidence-adaptive routing "
        "provides a scalable alternative to monolithic ensembles. Limitations include testing on univariate benchmarks, using a global margin threshold "
        "(τ = 0.08), discrete validation granularity on small training sets (N ≤ 30), and relying on softmax calibration. "
        "On certain long-range contour outline and spectral sets (ShapesAll, EthanolLevel), HIVE-COTE 2.0 still holds an advantage."
    )
    add_p(
        "Conclusion: CADENCE provides a fast, CPU-native classifier achieving 0.8864 accuracy across the 109 UCR datasets. Ranking #2 among "
        "evaluated classifiers across the archive, CADENCE closes the gap to HIVE-COTE 2.0 to just 0.31 percentage points while running in seconds, "
        "offering a practical tool for machine learning researchers and practitioners."
    )

    # -------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------
    add_h1("References")
    
    ref_list = [
        "[1] A. Dempster, F. Petitjean, and G. I. Webb, 'ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels,' Data Mining and Knowledge Discovery, vol. 34, no. 5, pp. 1454-1495, 2020. DOI: 10.1007/s10618-020-00701-z",
        "[2] A. Dempster, D. F. Schmidt, and G. I. Webb, 'MiniRocket: A very fast (almost) deterministic transform for time series classification,' in Proc. 27th ACM SIGKDD, 2021, pp. 248-257. DOI: 10.1145/3447548.3467231",
        "[3] A. Dempster, D. F. Schmidt, and G. I. Webb, 'Hydra: Competing convolutional kernels for fast and accurate time series classification,' Data Mining and Knowledge Discovery, vol. 37, no. 5, pp. 1779-1805, 2023. DOI: 10.1007/s10618-023-00939-3",
        "[4] C. W. Tan, A. Dempster, C. Bergmeir, and G. I. Webb, 'MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification,' Data Mining and Knowledge Discovery, vol. 36, no. 5, pp. 1623-1646, 2022. DOI: 10.1007/s10618-022-00844-1",
        "[5] M. Middlehurst, J. Large, M. Flynn, J. Lines, A. Bostrom, and A. Bagnall, 'HIVE-COTE 2.0: A new meta ensemble for time series classification,' Machine Learning, vol. 110, no. 11, pp. 3211-3243, 2021. DOI: 10.1007/s10994-021-06057-9",
        "[6] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, 'The UCR time series archive,' IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, pp. 1293-1305, 2019. DOI: 10.1109/JAS.2019.1911747",
        "[6b] J. Lines, S. Taylor, and A. Bagnall, 'Time series classification with HIVE-COTE: The hierarchical vote collective of transformation-based ensembles,' ACM TKDD, vol. 12, no. 5, pp. 1-35, 2018. DOI: 10.1145/3182382",
        "[7] A. Bagnall, J. Lines, A. Bostrom, J. Large, and E. Keogh, 'The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances,' Data Mining and Knowledge Discovery, vol. 31, no. 3, pp. 606-660, 2017. DOI: 10.1007/s10618-016-0483-9",
        "[8] A. Shifaz, C. Pelletier, F. Petitjean, and G. I. Webb, 'TS-CHIEF: A scalable and accurate forest algorithm for time series classification,' Data Mining and Knowledge Discovery, vol. 34, no. 3, pp. 742-775, 2020. DOI: 10.1007/s10618-020-00679-8",
        "[8b] H. Ismail Fawaz et al., 'InceptionTime: Finding AlexNet for time series classification,' Data Mining and Knowledge Discovery, vol. 34, no. 6, pp. 1936-1962, 2020. DOI: 10.1007/s10618-020-00710-y",
        "[9] E. A. Cornish and R. A. Fisher, 'Moments and cumulants in the specification of distributions,' Revue de l'Institut International de Statistique, vol. 5, no. 4, pp. 307-320, 1938. DOI: 10.2307/1400905",
        "[10] P. Geurts, D. Ernst, and L. Wehenkel, 'Extremely randomized trees,' Machine Learning, vol. 63, no. 1, pp. 3-42, 2006. DOI: 10.1007/s10994-006-6226-1"
    ]
    
    for ref_str in ref_list:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        p_ref.paragraph_format.space_before = Pt(1)
        p_ref.paragraph_format.space_after = Pt(2.5)
        p_ref.paragraph_format.line_spacing = 1.15
        run = p_ref.add_run(ref_str)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(51, 65, 85)

    doc.save(OUTPUT_DOCX)
    print(f"Publication DOCX successfully created at: {OUTPUT_DOCX}")

if __name__ == "__main__":
    build_docx()
