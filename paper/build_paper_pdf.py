"""
Builds an ultra-modern, publication-grade research paper PDF for CADENCE.
Features:
- Hidden Table of Contents (PDF Outline Bookmarks tree in navigation sidebar).
- Clickable internal citation links: clicking [1], [2], etc. jumps down to the corresponding reference.
- Modernized mathematical equation cards with distinct equation numbers (1), (2), (3)...
- Embedded modernized figures (architecture, routing, pairwise scatter, model rankings, Pareto, ablation).
- Running headers and page-numbered footers.
- Zero long hyphens (all standard single hyphens -).
- Accurate 109-dataset statistics, Wilcoxon tests, and percentage points notation.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Flowable, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

OUTPUT_PDF = "paper/cadence_paper_preview.pdf"

class BookmarkFlowable(Flowable):
    """Adds a bookmark anchor and outline entry (hidden Table of Contents) to the PDF."""
    def __init__(self, key, title, level=0):
        super().__init__()
        self.key = key
        self.title = title
        self.level = level

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=False)

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to calculate total page count, running headers, and running footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#64748B'))

        # Running header (pages 2+) - neutral title only, no author name to preserve double-blind compliance
        if self._pageNumber > 1:
            self.drawString(40, 755, "CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate TSC")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(40, 749, 572, 749)

        # Running footer (all pages) - standard clean submission numbering, zero draft labels
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(40, 42, 572, 42)
        
        page_text = f"{self._pageNumber}"
        self.drawRightString(572, 30, page_text)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Typography & Styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=8
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2
    )

    email_style = ParagraphStyle(
        'EmailStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=14
    )

    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.4,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=9
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.0,
        leading=14.0,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=12.0,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.4,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#334155'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.4,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=2.5
    )

    where_style = ParagraphStyle(
        'WhereStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.6,
        leading=9.8,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#475569'),
        leftIndent=8,
        spaceBefore=2,
        spaceAfter=4
    )

    math_box_style = ParagraphStyle(
        'MathBoxStyle',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#1E1B4B')
    )

    math_num_style = ParagraphStyle(
        'MathNumStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        alignment=TA_RIGHT,
        textColor=colors.HexColor('#64748B')
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#64748B'),
        spaceBefore=3,
        spaceAfter=7
    )

    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor('#0F172A'),
        alignment=TA_CENTER
    )

    td_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor('#0F172A')
    )

    td_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0F172A')
    )

    td_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor('#0F172A')
    )

    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.6,
        textColor=colors.HexColor('#334155'),
        spaceAfter=1.8
    )

    story = []

    def make_math_card(eq_text, eq_num):
        table = Table([
            [Paragraph(eq_text, math_box_style), Paragraph(f"({eq_num})", math_num_style)]
        ], colWidths=[490, 42])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#E2E8F0')),
            ('ROUNDEDCORNERS', [3, 3, 3, 3]),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        return table

    # ---------------------------------------------------------
    # TITLE & ABSTRACT
    # ---------------------------------------------------------
    story.append(Paragraph("CADENCE: A Confidence-Adaptive Dual-Expert Network for Fast and Accurate Time Series Classification", title_style))
    story.append(Paragraph("Onisa Mapunda", author_style))
    story.append(Paragraph('<a href="mailto:onisajr@gmail.com" color="#2563EB">onisajr@gmail.com</a>', email_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    story.append(BookmarkFlowable("sec_abstract", "Abstract", level=0))
    abstract_html = (
        "<b>Abstract</b> - Time series classification (TSC) has long been characterized by a sharp trade-off between "
        "classification accuracy and computational scalability. Heterogeneous meta-ensembles like HIVE-COTE 2.0 "
        '<a href="#ref5" color="#2563EB"><b>[5]</b></a> achieve state-of-the-art accuracy by combining representations across temporal, '
        "frequency, shapelet, and dictionary domains, but require days or weeks of compute. Conversely, ultra-fast random "
        'convolutional transforms such as MiniRocket <a href="#ref2" color="#2563EB"><b>[2]</b></a> and Hydra '
        '<a href="#ref3" color="#2563EB"><b>[3]</b></a> provide orders-of-magnitude speedups, but struggle with phase-independent '
        "statistical distributions, kinematic transitions, and catastrophic decision tree fragmentation on datasets with large class counts "
        "when combined with tree heads. Naive feature concatenation or static model averaging fails to resolve this tension, "
        "often inducing negative transfer and parameter dilution.<br/><br/>"
        "In this work, we present <b>CADENCE</b> (<b>C</b>onfidence-<b>A</b>daptive <b>D</b>ual-<b>E</b>xpert <b>N</b>etwork for time series "
        "<b>C</b>lassification <b>E</b>xcellence), a unified, CPU-native dual-expert architecture. CADENCE decouples representation "
        "learning into two specialized pathways: (i) a <i>Convolutional Linear Expert</i> pairing 10,000 deterministic dilated features "
        "with closed-form L2-regularized Woodbury ridge classification, and (ii) a <i>Distributional Interval Expert</i> pairing "
        "competing dilated kernels (Hydra) with thread-safe dyadic Cornish-Fisher moment approximations across signal kinematics "
        "(X, &Delta;X, &Delta;&sup2;X) and FFT spectral energy bands, fitted with an entropy-based ExtraTrees ensemble. "
        "To mediate between these paradigms, CADENCE incorporates an internal validation meta-router with rare-class preservation that dynamically "
        "selects between pure expert routing and confidence-weighted soft blending, followed by a full refit on 100% of training data. "
        'Evaluated across all 109 datasets of the standard equal-length UCR Time Series Archive <a href="#ref6" color="#2563EB"><b>[6]</b></a> '
        "over 30 resamples (3,270 total evaluations), CADENCE achieves a grand mean accuracy of <b>0.8864</b>. "
        "This ranks <b>#2 among evaluated classifiers</b> across the archive, surpassed only by HIVE-COTE 2.0 (0.8895, p_Holm = 0.295, "
        "no statistically significant difference), and ahead of Hydra+MultiRocket (0.8818, +0.46 percentage points, p_Holm = 1.000), "
        'MultiRocket <a href="#ref4" color="#2563EB"><b>[4]</b></a> (0.8797, +0.67 percentage points), and HIVE-COTE 1.0 '
        '<a href="#ref6b" color="#2563EB"><b>[6]</b></a> (0.8786, +0.78 percentage points, p_Holm = 0.048, statistically significant), '
        "closing the gap to HIVE-COTE 2.0 to just 0.31 percentage points while completing an evaluation run in an average of only <b>17.53 seconds</b> "
        "on a standard dual-core CPU."
    )
    story.append(Paragraph(abstract_html, abstract_body))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # ---------------------------------------------------------
    # 1. INTRODUCTION
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_intro", "1. Introduction", level=0))
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Time series classification (TSC) is a core problem in applied machine learning, underlying critical applications in "
        'electrocardiology <a href="#ref6" color="#2563EB"><b>[6]</b></a>, industrial anomaly detection, seismology, wearable motion analytics, '
        'and acoustic speech processing <a href="#ref7" color="#2563EB"><b>[7]</b></a>. The 109 univariate, equal-length datasets of the UCR '
        'Time Series Archive <a href="#ref6" color="#2563EB"><b>[6]</b></a> exhibit tremendous structural diversity: series lengths range from dozens '
        "to thousands of time points, sample sizes vary from tens to thousands of instances, and class cardinalities span binary discrimination "
        "(K=2) to fine-grained categorization (K=60).", body_style))
    story.append(Paragraph(
        'Historically, achieving top accuracy demanded heavy heterogeneous meta-ensembles, most notably HIVE-COTE 2.0 (HC2) '
        '<a href="#ref5" color="#2563EB"><b>[5]</b></a>, which combines STC, TDE, DrCIF, and Arsenal. While HC2 achieves an unmatched '
        "published benchmark accuracy of 0.8895, its computational cost is substantial, requiring an average of approximately 3 hours per dataset "
        "(over 340 CPU hours across the 112 archive datasets).", body_style))
    story.append(Paragraph(
        'In response, the field shifted toward randomized convolutional transforms, initiated by ROCKET '
        '<a href="#ref1" color="#2563EB"><b>[1]</b></a>, refined by MiniRocket <a href="#ref2" color="#2563EB"><b>[2]</b></a>, '
        'MultiRocket <a href="#ref4" color="#2563EB"><b>[4]</b></a>, and Hydra <a href="#ref3" color="#2563EB"><b>[3]</b></a>. '
        "By projecting time series into high-dimensional linear spaces via dilated convolutions pooled with Proportion of Positive Values "
        "(PPV), these methods run in seconds. However, single-model convolutional transforms suffer from key failure modes: "
        "(1) inability to capture phase-free statistical distributions, velocity, and acceleration kinematics; "
        "(2) severe decision tree fragmentation on datasets with large class counts (K &ge; 12) when tree heads are naively applied; and "
        "(3) negative transfer when concatenating heterogeneous feature spaces into a single linear regularizer.", body_style))

    story.append(BookmarkFlowable("sec_contrib", "1.1 Contributions", level=1))
    story.append(Paragraph("1.1 Contributions", h2_style))
    story.append(Paragraph("&bull; <b>Decoupled Dual-Expert Architecture:</b> Harmonizes a closed-form convolutional linear expert (10,000 features) with an interval-distributional expert (Hydra, Cornish-Fisher moments on kinematics, FFT spectral quantiles, ExtraTrees, 1,851 features).", bullet_style))
    story.append(Paragraph("&bull; <b>Confidence-Adaptive Meta-Router with Full Refit:</b> Introduces an internal validation router with rare-class preservation navigating three distinct regimes: Pure Branch A, Pure Branch B, and Smooth Confidence Blending, refitting models on 100% of training data.", bullet_style))
    story.append(Paragraph("&bull; <b>Conditional Inference-Time Branch Pruning:</b> Completely skips the unselected branch during dominance regimes (24.8% of datasets), reducing test-time feature extraction overhead.", bullet_style))
    story.append(Paragraph("&bull; <b>Exhaustive Benchmark Evaluation (3,270 Runs):</b> Evaluated on all 109 UCR datasets across 30 resamples, attaining 0.8864 grand mean accuracy (#2 among evaluated classifiers across the archive).", bullet_style))
    story.append(Paragraph("&bull; <b>Rigorous Statistical Testing:</b> Reports two-sided Wilcoxon signed-rank test p-values with Holm-Bonferroni correction, mean ranks, and per-dataset win/tie/loss counts against 25 published benchmark models.", bullet_style))

    # Architecture Image
    story.append(Spacer(1, 4))
    if os.path.exists("paper/figures/fig1_architecture_render-1.png"):
        story.append(Image("paper/figures/fig1_architecture_render-1.png", width=470, height=327, hAlign='CENTER'))
        story.append(Paragraph("Figure 1: Architectural overview of the CADENCE dual-expert classifier, validation routing, full refit, and test inference.", caption_style))

    # ---------------------------------------------------------
    # 2. RELATED WORK
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_related", "2. Related Work", level=0))
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        '<b>2.1 Classical TSC:</b> Distance-based methods (1NN-DTW) <a href="#ref7" color="#2563EB"><b>[7]</b></a> offer robust elastic '
        'alignment but suffer from quadratic complexity O(N L&sup2;). Dictionary methods (BOSS, WEASEL) discretize windows into symbolic Fourier words. '
        'Interval ensembles (TSF, CIF, DrCIF) extract summary statistics across intervals.', body_style))
    story.append(Paragraph(
        '<b>2.2 Random Convolutions:</b> ROCKET <a href="#ref1" color="#2563EB"><b>[1]</b></a> demonstrated that random kernels pooled with PPV '
        'produce linearly separable features. MiniRocket <a href="#ref2" color="#2563EB"><b>[2]</b></a> introduced deterministic integer kernels '
        'and precomputed dilations, while Hydra <a href="#ref3" color="#2563EB"><b>[3]</b></a> introduced competing kernel groups. MultiRocket '
        '<a href="#ref4" color="#2563EB"><b>[4]</b></a> added first-order differences and four pooling operators.', body_style))
    story.append(Paragraph(
        '<b>2.3 Ensembles & Meta-Ensembles:</b> HIVE-COTE 2.0 <a href="#ref5" color="#2563EB"><b>[5]</b></a> combines 4 distinct classifiers '
        'via CAWPE meta-weighting. TS-CHIEF <a href="#ref8" color="#2563EB"><b>[8]</b></a> integrates dictionary, interval, and distance criteria into trees. '
        'InceptionTime <a href="#ref8b" color="#2563EB"><b>[8b]</b></a> adapts deep residual networks for time series. '
        'While highly accurate, these meta-ensembles require substantial cluster compute. In contrast, CADENCE introduces an internal validation '
        'meta-router to conditionally select or blend experts on a lightweight CPU budget.', body_style))

    # ---------------------------------------------------------
    # 3. METHODOLOGY
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_method", "3. The CADENCE Methodology", level=0))
    story.append(Paragraph("3. The CADENCE Methodology", h1_style))
    story.append(Paragraph(
        '<b>3.1 Branch A: Convolutional Linear Expert:</b> MiniRocket <a href="#ref2" color="#2563EB"><b>[2]</b></a> extracts D_A = 10,000 features '
        'using length-9 integer kernels &omega; &in; {-1, 2}&sup9; (84 unique configurations with 3 weights equal to 2 and 6 weights equal to -1) '
        'and Proportion of Positive Values (PPV) pooling:', body_style))
    
    # Equation 1: Convolution & PPV (Line-by-line formatted)
    eq1_img = "paper/figures/equations/eq1.png"
    if os.path.exists(eq1_img):
        story.append(Image(eq1_img, width=520, height=63, hAlign='CENTER'))
    else:
        eq1_text = "A_t = (X *_d &omega;)_t = &sum;_{j=0}^8 &omega;_j &middot; X_{t + j&middot;d}, &nbsp; &phi;_{PPV}(X, &omega;, b) = (1 / (L - 8d)) &sum;_t &Iopf;(A_t > b)"
        story.append(make_math_card(eq1_text, 1))
    story.append(Paragraph("<b>where:</b> <i>A<sub>t</sub></i> is convolution activation at step <i>t</i>; <i>X</i> is the series; &lowast;<sub><i>d</i></sub> denotes convolution with dilation <i>d</i>; &omega; &in; {-1, 2}&sup9; is the length-9 kernel; &phi;<sub>PPV</sub> is PPV pooling; <i>b</i> is quantile bias; and <b>1</b>(&middot;) is the indicator function.", where_style))

    story.append(Paragraph(
        "Standardized features &tilde;F_A are classified via multiclass Ridge regression with closed-form L2 regularization across 15 log-spaced candidate alphas in [10^-3, 10^4]. "
        "When N < D_A, the dual Woodbury matrix identity is evaluated:", body_style))
    
    # Equation 2: Woodbury Solution (Line-by-line formatted)
    eq2_img = "paper/figures/equations/eq2.png"
    if os.path.exists(eq2_img):
        story.append(Image(eq2_img, width=520, height=63, hAlign='CENTER'))
    else:
        eq2_text = "W* = (&tilde;F_A&Topf; &tilde;F_A + &alpha; I_{D_A})&supmin;&sup1; &tilde;F_A&Topf; Y = &tilde;F_A&Topf; (&tilde;F_A &tilde;F_A&Topf; + &alpha; I_N)&supmin;&sup1; Y"
        story.append(make_math_card(eq2_text, 2))
    story.append(Paragraph("<b>where:</b> <i>W</i>* is the closed-form weight matrix; &tilde;<i>F</i><sub><i>A</i></sub> is standardized features; <i>Y</i> is one-vs-rest labels; &alpha; is the L<sub>2</sub> penalty; and <i>I</i> is the identity matrix evaluated in primal (<i>N</i> &ge; <i>D<sub>A</sub></i>) or dual Woodbury space (<i>N</i> &lt; <i>D<sub>A</sub></i>).", where_style))

    story.append(Paragraph("Posterior class probabilities are derived via softmax over raw decision values:", body_style))
    
    # Equation 3: Softmax Probabilities (Line-by-line formatted)
    eq3_img = "paper/figures/equations/eq3.png"
    if os.path.exists(eq3_img):
        story.append(Image(eq3_img, width=520, height=63, hAlign='CENTER'))
    else:
        eq3_text = "P_A(y = k | X) = exp(f_{A, k}(X)) / &sum;_{j=1}^K exp(f_{A, j}(X)), &nbsp; where &nbsp; f_A(X) = W*&Topf; &tilde;F_A(X) + b_A"
        story.append(make_math_card(eq3_text, 3))
    story.append(Paragraph("<b>where:</b> <i>P<sub>A</sub></i>(<i>y</i> = <i>k</i> | <i>X</i>) is the posterior class probability; <i>f<sub>A,k</sub></i>(<i>X</i>) is linear scoring output; <i>W</i>* is fitted weights; and <i>b<sub>A</sub></i> is the class bias vector.", where_style))

    story.append(Paragraph(
        '<b>3.2 Branch B: Distributional Interval Expert:</b> Combines 16 groups of 8 competing Hydra kernels (128 features) '
        '<a href="#ref3" color="#2563EB"><b>[3]</b></a>, dyadic intervals up to depth 5 with Cornish-Fisher '
        '<a href="#ref9" color="#2563EB"><b>[9]</b></a> quantile approximations across raw series (X), velocity (&Delta;X), and acceleration (&Delta;&sup2;X) '
        '(1,701 features), and real FFT spectral quantiles (22 features). Total feature dimension D_B = 1,851. '
        'The Cornish-Fisher expansion estimates arbitrary quantiles q_&alpha; in linear time O(M) without sorting:', body_style))

    # Equation 4: Cornish-Fisher Expansion (Line-by-line formatted)
    eq4_img = "paper/figures/equations/eq4.png"
    if os.path.exists(eq4_img):
        story.append(Image(eq4_img, width=520, height=63, hAlign='CENTER'))
    else:
        eq4_text = "q_&alpha;(S) = &mu; + &sigma; [ z_&alpha; + (S/6)(z_&alpha;&sup2; - 1) + (K/24)(z_&alpha;&sup3; - 3z_&alpha;) - (S&sup2;/36)(2z_&alpha;&sup3; - 5z_&alpha;) ]"
        story.append(make_math_card(eq4_text, 4))
    story.append(Paragraph("<b>where:</b> <i>q</i><sub>&alpha;</sub>(<i>S</i>) is estimated quantile at level &alpha;; &mu; and &sigma; are segment mean and standard deviation; <i>z</i><sub>&alpha;</sub> = &Phi;<sup>&minus;1</sup>(&alpha;) is standard normal quantile; <i>S</i> is sample skewness; <i>K</i> is sample kurtosis; and &Psi;(&middot;) is the asymptotic polynomial expansion.", where_style))

    story.append(Paragraph(
        'Features are fitted with 100 Extremely Randomized Trees (ExtraTrees) <a href="#ref10" color="#2563EB"><b>[10]</b></a> '
        'using Shannon entropy and 10% random feature subsampling per split.', body_style))

    story.append(Paragraph(
        "<b>3.3 Confidence-Adaptive Meta-Router:</b> Partitions training data into an internal 30% held-out validation set while safely preserving "
        "singleton classes. Validation accuracies Val_A and Val_B are evaluated on held-out data. The routing decision is governed by "
        "empirical difference &Delta; = Val_A - Val_B with threshold &tau; = 0.08:", body_style))

    # Equation 5: Routing Rule (Line-by-line formatted)
    eq5_img = "paper/figures/equations/eq5.png"
    if os.path.exists(eq5_img):
        story.append(Image(eq5_img, width=520, height=85, hAlign='CENTER'))
    else:
        eq5_text = "&ycirc;(X) = argmax_k P_A(k|X) &nbsp; [if &Delta; > 0.08], &nbsp; argmax_k P_B(k|X) &nbsp; [if &Delta; < -0.08], &nbsp; argmax_k P_{blend}(k|X) &nbsp; [if |&Delta;| &le; 0.08]"
        story.append(make_math_card(eq5_text, 5))
    story.append(Paragraph("<b>where:</b> <i>y</i>*(<i>X</i>) is the predicted class label; &Delta; = Val<sub><i>A</i></sub> &minus; Val<sub><i>B</i></sub> is empirical validation accuracy margin; &tau; = 0.08 is regime threshold; <i>P<sub>A</sub></i> and <i>P<sub>B</sub></i> are expert posteriors; and <i>P</i><sub>blend</sub> is the blended probability vector.", where_style))

    story.append(Paragraph("where in the competitive regime, predictions are smoothly blended with weight w_A:", body_style))
    
    # Equation 6: Soft Blend Weight (Line-by-line formatted)
    eq6_img = "paper/figures/equations/eq6.png"
    if os.path.exists(eq6_img):
        story.append(Image(eq6_img, width=520, height=63, hAlign='CENTER'))
    else:
        eq6_text = "P_{blend}(y = k | X) = w_A P_A(y = k | X) + (1 - w_A) P_B(y = k | X), &nbsp; where &nbsp; w_A = clip(0.5 + 2.0&Delta;, 0.15, 0.85)"
        story.append(make_math_card(eq6_text, 6))
    story.append(Paragraph("<b>where:</b> <i>P</i><sub>blend</sub> is the blended probability vector; <i>w<sub>A</sub></i> is dynamic weight assigned to Branch A; (1 &minus; <i>w<sub>A</sub></i>) is weight assigned to Branch B; and clip(&middot;, 0.15, 0.85) enforces robust probability bounds.", where_style))

    if os.path.exists("paper/figures/fig2_routing_render-1.png"):
        story.append(Spacer(1, 4))
        story.append(Image("paper/figures/fig2_routing_render-1.png", width=410, height=235, hAlign='CENTER'))
        story.append(Paragraph("Figure 2: Piecewise routing and blending regimes as a function of validation margin &Delta; with step transitions.", caption_style))

    # ---------------------------------------------------------
    # 4. RESULTS & BENCHMARKS
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_results", "4. Experimental Results & Benchmark Rankings", level=0))
    story.append(Paragraph("4. Experimental Results and Benchmark Rankings", h1_style))
    story.append(Paragraph(
        'Following the 109-dataset, 30-resample evaluation protocol of MiniRocket <a href="#ref2" color="#2563EB"><b>[2]</b></a> and MultiRocket '
        '<a href="#ref4" color="#2563EB"><b>[4]</b></a>, CADENCE was evaluated across the 109 equal-length datasets of the UCR Time Series Archive '
        '<a href="#ref6" color="#2563EB"><b>[6]</b></a> over the 30 official resamples (3,270 total evaluations). '
        'Table 1 reports the grand mean accuracy against published state-of-the-art benchmarks.', body_style))

    # Benchmark Table
    table_data = [
        [Paragraph("Rank", th_style), Paragraph("Classifier", th_style), Paragraph("Algorithmic Paradigm", th_style), Paragraph("Mean Acc &plusmn; Std", th_style), Paragraph("Mean Rank", th_style), Paragraph("W / T / L vs Ours", th_style), Paragraph("p_Holm", th_style)],
        [Paragraph("--", td_center), Paragraph("<i>Oracle Ceiling (Post-hoc)</i>", td_style), Paragraph("Per-dataset Best", td_style), Paragraph("<i>0.9067 &plusmn; 0.1062</i>", td_center), Paragraph("--", td_center), Paragraph("--", td_center), Paragraph("--", td_center)],
        [Paragraph("<b>1</b>", td_center), Paragraph("<b>HIVE-COTE 2.0</b>", td_bold), Paragraph("Heterogeneous Meta-Ensemble", td_style), Paragraph("<b>0.8895 &plusmn; 0.1159</b>", td_bold), Paragraph("<b>6.22</b>", td_bold), Paragraph("63 / 7 / 39", td_center), Paragraph("0.295", td_center)],
        [Paragraph("<b>2</b>", td_center), Paragraph("<b>CADENCE (Ours)</b>", td_bold), Paragraph("<b>Adaptive Dual-Expert</b>", td_style), Paragraph("<b>0.8864 &plusmn; 0.1120</b>", td_bold), Paragraph("<b>8.26</b>", td_bold), Paragraph("<b>--</b>", td_center), Paragraph("<b>--</b>", td_center)],
        [Paragraph("3", td_center), Paragraph("Hydra+MultiRocket", td_style), Paragraph("Convolutional Hybrid", td_style), Paragraph("0.8818 &plusmn; 0.1218", td_center), Paragraph("7.67", td_center), Paragraph("46 / 7 / 56", td_center), Paragraph("1.000", td_center)],
        [Paragraph("4", td_center), Paragraph("MultiRocket (100k)", td_style), Paragraph("Convolutional Pooling", td_style), Paragraph("0.8800 &plusmn; 0.1218", td_center), Paragraph("8.03", td_center), Paragraph("48 / 7 / 54", td_center), Paragraph("1.000", td_center)],
        [Paragraph("5", td_center), Paragraph("MultiRocket (50k default)", td_style), Paragraph("Convolutional Pooling", td_style), Paragraph("0.8797 &plusmn; 0.1222", td_center), Paragraph("8.19", td_center), Paragraph("47 / 10 / 52", td_center), Paragraph("1.000", td_center)],
        [Paragraph("6", td_center), Paragraph("HIVE-COTE 1.0", td_style), Paragraph("Hierarchical Meta-Ensemble", td_style), Paragraph("0.8786 &plusmn; 0.1228", td_center), Paragraph("10.80", td_center), Paragraph("59 / 7 / 43", td_center), Paragraph("0.048", td_center)],
        [Paragraph("7", td_center), Paragraph("TS-CHIEF", td_style), Paragraph("Metric / Tree Ensemble", td_style), Paragraph("0.8761 &plusmn; 0.1281", td_center), Paragraph("10.52", td_center), Paragraph("61 / 7 / 41", td_center), Paragraph("0.035", td_center)],
        [Paragraph("8", td_center), Paragraph("MultiRocket (10k)", td_style), Paragraph("Convolutional Pooling", td_style), Paragraph("0.8749 &plusmn; 0.1254", td_center), Paragraph("9.83", td_center), Paragraph("59 / 6 / 44", td_center), Paragraph("0.014", td_center)],
        [Paragraph("9", td_center), Paragraph("MiniRocket", td_style), Paragraph("Convolutional Transform", td_style), Paragraph("0.8724 &plusmn; 0.1300", td_center), Paragraph("11.40", td_center), Paragraph("72 / 6 / 31", td_center), Paragraph("&lt; 0.001", td_center)],
        [Paragraph("10", td_center), Paragraph("InceptionTime", td_style), Paragraph("Deep ConvNet Ensemble", td_style), Paragraph("0.8721 &plusmn; 0.1285", td_center), Paragraph("11.95", td_center), Paragraph("67 / 5 / 37", td_center), Paragraph("0.050", td_center)],
        [Paragraph("11", td_center), Paragraph("Hydra", td_style), Paragraph("Competing Dilated Kernels", td_style), Paragraph("0.8714 &plusmn; 0.1319", td_center), Paragraph("10.45", td_center), Paragraph("66 / 5 / 38", td_center), Paragraph("0.005", td_center)],
        [Paragraph("12-26", td_center), Paragraph("ROCKET, Arsenal, DrCIF, TDE, STC, CIF, WEASEL, etc.", td_style), Paragraph("Classical & Interval Baselines", td_style), Paragraph("0.8645 - 0.7954", td_center), Paragraph("11.76 - 21.28", td_center), Paragraph("&ge; 76 wins", td_center), Paragraph("&lt; 0.001", td_center)],
    ]
    t1 = Table(table_data, colWidths=[38, 132, 140, 95, 52, 60, 43])
    t1.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.HexColor('#0F172A')),
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor('#0F172A')),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.HexColor('#0F172A')),
        ('LINEBELOW', (0,1), (-1,1), 0.4, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.0),
        ('TOPPADDING', (0,0), (-1,-1), 2.0),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: Benchmark ranking across the 109 UCR Archive datasets over 30 resamples (Wilcoxon tests with Holm correction).", caption_style))

    # Critical Difference Diagram
    if os.path.exists("paper/figures/fig6_cd_render-1.png"):
        story.append(Spacer(1, 4))
        story.append(Image("paper/figures/fig6_cd_render-1.png", width=500, height=240, hAlign='CENTER'))
        story.append(Paragraph("Figure 3: Critical Difference diagram of mean ranks across 109 UCR datasets (Wilcoxon signed-rank test with Holm correction, alpha = 0.05, Rank 1 on right).", caption_style))

    # Pairwise Scatter & Rankings
    if os.path.exists("paper/figures/fig3_scatter_render-1.png"):
        story.append(Spacer(1, 4))
        story.append(Image("paper/figures/fig3_scatter_render-1.png", width=500, height=230, hAlign='CENTER'))
        story.append(Paragraph("Figure 4: Pairwise scatter plots of CADENCE against Hydra+MultiRocket and HIVE-COTE 2.0 with readable callouts.", caption_style))

    # ---------------------------------------------------------
    # 5. BREAKTHROUGHS & ABLATIONS
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_ablation", "5. Breakthroughs & Component Ablation", level=0))
    story.append(Paragraph("5. Breakthroughs and Component Ablation", h1_style))
    story.append(Paragraph(
        "CADENCE achieves decisive wins on specialized sensor and kinematic datasets over Hydra+MultiRocket: "
        "<b>SemgHandMovementCh2</b> (0.8713 vs 0.7717, <b>+9.96 pp</b>), <b>PigAirwayPressure</b> (0.8220 vs 0.7234, <b>+9.86 pp</b>), "
        "<b>InlineSkate</b> (0.6047 vs 0.5081, <b>+9.66 pp</b>), and <b>DistalPhalanxTW</b> (0.7746 vs 0.6969, <b>+7.77 pp</b>). "
        "CADENCE also directly surpasses HIVE-COTE 2.0 on 39 datasets (e.g. DistalPhalanxTW +7.29 pp, InlineSkate +5.92 pp, Earthquakes +4.99 pp). "
        "On Seed 0 (the exact original UCR split), CADENCE obtains 0.7482 on Earthquakes (identical to majority class rate) and 0.7050 on DistalPhalanxTW.", body_style))

    # Ablation Table
    ablation_data = [
        [Paragraph("Configuration", th_style), Paragraph("Component Modified", th_style), Paragraph("Mean Accuracy", th_style), Paragraph("Delta to Full (pp)", th_style)],
        [Paragraph("Branch A Only", td_style), Paragraph("MiniRocket + Ridge (No Interval Expert)", td_style), Paragraph("0.8724", td_center), Paragraph("-1.40 pp", td_center)],
        [Paragraph("Branch B Only", td_style), Paragraph("Hydra + MQ + FFT + ExtraTrees (No Conv Expert)", td_style), Paragraph("0.8729", td_center), Paragraph("-1.35 pp", td_center)],
        [Paragraph("Fixed 50/50 Soft Blend", td_style), Paragraph("Naive Averaging (No Dynamic Router)", td_style), Paragraph("0.8741", td_center), Paragraph("-1.23 pp", td_center)],
        [Paragraph("Static Routing (K &ge; 12)", td_style), Paragraph("Class-Count Heuristic Threshold", td_style), Paragraph("0.8813", td_center), Paragraph("-0.51 pp", td_center)],
        [Paragraph("<b>CADENCE (Full System)</b>", td_bold), Paragraph("<b>Confidence-Adaptive Meta-Router</b>", td_bold), Paragraph("<b>0.8864</b>", td_bold), Paragraph("<b>--</b>", td_bold)],
    ]
    t2 = Table(ablation_data, colWidths=[130, 220, 80, 80])
    t2.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.HexColor('#0F172A')),
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor('#0F172A')),
        ('LINEBELOW', (0,-1), (-1,-1), 1.0, colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.0),
        ('TOPPADDING', (0,0), (-1,-1), 2.0),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Controlled component ablation across all 109 datasets (30 resamples). MQ denotes MomentQuant.", caption_style))

    # Pareto & Ablation Plots
    if os.path.exists("paper/figures/fig4_pareto_render-1.png") and os.path.exists("paper/figures/fig5_ablation_render-1.png"):
        story.append(Spacer(1, 4))
        p_table = [
            [Image("paper/figures/fig4_pareto_render-1.png", width=245, height=145),
             Image("paper/figures/fig5_ablation_render-1.png", width=245, height=145)]
        ]
        t_img = Table(p_table, colWidths=[250, 250])
        t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(t_img)
        story.append(Paragraph("Figure 5: Left: Empirical Pareto frontier vs. cluster training runtime. Right: Systematic component ablations.", caption_style))

    # ---------------------------------------------------------
    # 6. ROUTING ANALYSIS & EFFICIENCY
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_routing", "6. Routing Analysis & Computational Efficiency", level=0))
    story.append(Paragraph("6. Routing Analysis and Computational Efficiency", h1_style))
    story.append(Paragraph(
        "<b>Exact Routing Distribution (N = 109 Datasets):</b> Across all 109 evaluated datasets, <b>82 datasets (75.2%)</b> trigger "
        "Regime II (Competitive Blending, e.g. Phoneme, ArrowHead, ECG200), <b>18 datasets (16.5%)</b> trigger Regime III (Pure Branch A Dominance, "
        "e.g. PigAirwayPressure, PigCVP, Fish), and <b>9 datasets (8.3%)</b> trigger Regime I (Pure Branch B Dominance, e.g. Semg, InlineSkate). "
        "In total, 27 datasets (24.8%) trigger dominance regimes where test inference skips one branch completely.", body_style))
    story.append(Paragraph(
        "<b>Pareto Efficiency:</b> CADENCE executes in an average of <b>17.53 seconds</b> per seed run. The entire 3,270-run benchmark completed in "
        "<b>15.92 CPU hours</b> (~7.5 hours wall-clock time via bidirectional execution on an Intel Core i7-7600U). Published cluster runtimes "
        "report TS-CHIEF at 1,016.87h (~32,685s per dataset), HIVE-COTE 1.0 at 427.18h (~13,730s), and HIVE-COTE 2.0 at 340.21h (~10,935s). "
        "CADENCE establishes a superior Pareto frontier, nearing HC2 accuracy while running orders of magnitude faster.", body_style))

    # ---------------------------------------------------------
    # 7. DISCUSSION, LIMITATIONS & CONCLUSION
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_conclusion", "7. Discussion, Limitations & Conclusion", level=0))
    story.append(Paragraph("7. Discussion, Limitations, and Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Discussion & Limitations:</b> CADENCE demonstrates that combining complementary representations through confidence-adaptive routing "
        "provides a scalable alternative to monolithic ensembles. Limitations include testing on univariate benchmarks, using a global margin threshold "
        "(&tau; = 0.08), discrete validation granularity on small training sets (N &le; 30), and relying on softmax calibration. "
        "On certain long-range contour outline and spectral sets (ShapesAll, EthanolLevel), HIVE-COTE 2.0 still holds an advantage.", body_style))
    story.append(Paragraph(
        "<b>Conclusion:</b> CADENCE provides a fast, CPU-native classifier achieving 0.8864 accuracy across the 109 UCR datasets. Ranking #2 among "
        "evaluated classifiers across the archive, CADENCE closes the gap to HIVE-COTE 2.0 to just 0.31 percentage points while running in seconds, "
        "offering a practical tool for machine learning researchers and practitioners.", body_style))

    # ---------------------------------------------------------
    # REFERENCES (With Clickable In-Text Targets)
    # ---------------------------------------------------------
    story.append(BookmarkFlowable("sec_refs", "References", level=0))
    story.append(Paragraph("References", h1_style))

    ref_items = [
        ("ref1", "[1] A. Dempster, F. Petitjean, and G. I. Webb, 'ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels,' Data Mining and Knowledge Discovery, vol. 34, no. 5, pp. 1454-1495, 2020. DOI: 10.1007/s10618-020-00701-z"),
        ("ref2", "[2] A. Dempster, D. F. Schmidt, and G. I. Webb, 'MiniRocket: A very fast (almost) deterministic transform for time series classification,' in Proc. 27th ACM SIGKDD, 2021, pp. 248-257. DOI: 10.1145/3447548.3467231"),
        ("ref3", "[3] A. Dempster, D. F. Schmidt, and G. I. Webb, 'Hydra: Competing convolutional kernels for fast and accurate time series classification,' Data Mining and Knowledge Discovery, vol. 37, no. 5, pp. 1779-1805, 2023. DOI: 10.1007/s10618-023-00939-3"),
        ("ref4", "[4] C. W. Tan, A. Dempster, C. Bergmeir, and G. I. Webb, 'MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification,' Data Mining and Knowledge Discovery, vol. 36, no. 5, pp. 1623-1646, 2022. DOI: 10.1007/s10618-022-00844-1"),
        ("ref5", "[5] M. Middlehurst, J. Large, M. Flynn, J. Lines, A. Bostrom, and A. Bagnall, 'HIVE-COTE 2.0: A new meta ensemble for time series classification,' Machine Learning, vol. 110, no. 11, pp. 3211-3243, 2021. DOI: 10.1007/s10994-021-06057-9"),
        ("ref6", "[6] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, 'The UCR time series archive,' IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, pp. 1293-1305, 2019. DOI: 10.1109/JAS.2019.1911747"),
        ("ref6b", "[6b] J. Lines, S. Taylor, and A. Bagnall, 'Time series classification with HIVE-COTE: The hierarchical vote collective of transformation-based ensembles,' ACM TKDD, vol. 12, no. 5, pp. 1-35, 2018. DOI: 10.1145/3182382"),
        ("ref7", "[7] A. Bagnall, J. Lines, A. Bostrom, J. Large, and E. Keogh, 'The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances,' Data Mining and Knowledge Discovery, vol. 31, no. 3, pp. 606-660, 2017. DOI: 10.1007/s10618-016-0483-9"),
        ("ref8", "[8] A. Shifaz, C. Pelletier, F. Petitjean, and G. I. Webb, 'TS-CHIEF: A scalable and accurate forest algorithm for time series classification,' Data Mining and Knowledge Discovery, vol. 34, no. 3, pp. 742-775, 2020. DOI: 10.1007/s10618-020-00679-8"),
        ("ref8b", "[8b] H. Ismail Fawaz et al., 'InceptionTime: Finding AlexNet for time series classification,' Data Mining and Knowledge Discovery, vol. 34, no. 6, pp. 1936-1962, 2020. DOI: 10.1007/s10618-020-00710-y"),
        ("ref9", "[9] E. A. Cornish and R. A. Fisher, 'Moments and cumulants in the specification of distributions,' Revue de l'Institut International de Statistique, vol. 5, no. 4, pp. 307-320, 1938. DOI: 10.2307/1400905"),
        ("ref10", "[10] P. Geurts, D. Ernst, and L. Wehenkel, 'Extremely randomized trees,' Machine Learning, vol. 63, no. 1, pp. 3-42, 2006. DOI: 10.1007/s10994-006-6226-1")
    ]

    for key, text in ref_items:
        ref_html = f'<a name="{key}"/>{text}'
        story.append(Paragraph(ref_html, ref_style))

    # Build Document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Publication PDF successfully compiled to {OUTPUT_PDF}!")

if __name__ == "__main__":
    build_pdf()
