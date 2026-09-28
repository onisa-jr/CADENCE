"""
Generate ultra-modern, publication-grade vector (PDF) and high-DPI (PNG) figures for CADENCE paper.
Incorporates all audit improvements:
- Zero text overlaps, legible typography, proper bounding boxes.
- Full architectural flow in Figure 1 including refit step and test-time inference pruning path.
- Corrected notation in Figure 2 (Delta instead of tau, step transitions cleanly shown).
- Pairwise scatter in Figure 3 with readable non-truncated annotations for top wins and deficits.
- Accurate published cluster runtimes in Figure 4 (TS-CHIEF ~32,685s, HC1 ~13,730s, HC2 ~10,935s, CADENCE 17.53s).
- Percentage points (pp) labels in ablation plots.
- Zero long hyphens (all standard single hyphens -).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Modern publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9.5,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.titlesize': 12.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 300
})

os.makedirs("paper/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: Architectural Flow with Refit and Inference Path
# -------------------------------------------------------------
def plot_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 62)

    def draw_card(x, y, w, h, title, subtitle, header_color, bg_color='#FFFFFF', border_color='#CBD5E1', badge=None):
        # Drop shadow
        shadow = patches.FancyBboxPatch((x + 0.5, y - 0.5), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                        facecolor='#E2E8F0', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        # Main card
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.2, zorder=2)
        ax.add_patch(card)
        # Accent header bar
        accent = patches.FancyBboxPatch((x, y + h - 2.2), w, 2.2, boxstyle="round,pad=0.1,rounding_size=0.6",
                                        facecolor=header_color, edgecolor='none', zorder=3)
        ax.add_patch(accent)
        # Title text
        ax.text(x + w/2, y + h - 4.2, title, ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0F172A', zorder=4)
        # Subtitle lines
        if subtitle:
            ax.text(x + w/2, y + (h - 5.0)/2, subtitle, ha='center', va='center', fontsize=7.6, color='#334155', zorder=4, linespacing=1.25)
        # Badge
        if badge:
            bx = x + w/2
            by = y + 2.4
            badge_p = patches.FancyBboxPatch((bx - 10, by - 1.1), 20, 2.2, boxstyle="round,pad=0.1,rounding_size=0.8",
                                            facecolor=header_color, edgecolor='none', zorder=4, alpha=0.15)
            ax.add_patch(badge_p)
            ax.text(bx, by, badge, ha='center', va='center', fontsize=7.0, fontweight='bold', color=header_color, zorder=5)

    def draw_arrow(x1, y1, x2, y2, label=None, color='#475569', label_pos=0.5, label_dy=1.6):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=color, lw=1.4, shrinkA=3, shrinkB=3),
                    zorder=6)
        dot = patches.Circle((x1, y1), 0.6, facecolor=color, edgecolor='none', zorder=7)
        ax.add_patch(dot)
        if label:
            mx = x1 + (x2 - x1) * label_pos
            my = y1 + (y2 - y1) * label_pos + label_dy
            ax.text(mx, my, label, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1E293B',
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.6), zorder=8)

    # Top Title
    ax.text(60, 60, "CADENCE Architecture: Internal Validation Meta-Routing and Inference Execution",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')
    ax.text(60, 57.5, "Dual-expert representations with singleton-safe validation routing, full refit, and conditional inference pruning",
            ha='center', va='center', fontsize=8.2, color='#64748B')

    # Column 1: Input
    draw_card(2, 23, 16, 17, "Input Time Series", 
              "Training Data:\n(X_train, y_train)\nN samples, length L\nTest Sample: X_test", 
              '#3B82F6', badge="Zero Test Leakage")

    # Column 2: Safe Split
    draw_card(22, 23, 18, 17, "Safe Validation Split", 
              "30% Held-Out Split\nSingletons (N_c = 1)\nkept in Train Fold;\nClasses >= 2 stratified", 
              '#0284C7', badge="Singleton-Safe Split")
    draw_arrow(18, 31.5, 22, 31.5)

    # Column 3: Branch A (Top) & Branch B (Bottom)
    draw_card(44, 37, 28, 17, "Branch A: Convolutional Expert", 
              "MiniRocket: 10,000 Features\nPPV Pooling + StandardScaler\nClosed-Form Woodbury L2 Ridge\nSoftmax Probability: P_A(y|X)", 
              '#10B981', badge="10,000 Features")
    draw_arrow(40, 35, 44, 45, label="Train Fold (70%)", label_pos=0.45)

    draw_card(44, 6, 28, 17, "Branch B: Interval Expert", 
              "Hydra: 16 Groups x 8 Kernels (128)\nMomentQuant: CF Moments (1,701)\nFFT Spectral Quantiles (22)\n100 Extremely Randomized Trees", 
              '#F59E0B', badge="1,851 Features")
    draw_arrow(40, 27, 44, 15, label="Train Fold (70%)", label_pos=0.45)

    # Column 4: Meta-Router
    draw_card(76, 22, 17, 18, "Meta-Router", 
              "Evaluates on 30% Val Fold:\nVal_A = Acc(Branch A)\nVal_B = Acc(Branch B)\nMargin: Delta = Val_A - Val_B\nThreshold: tau = +-0.08", 
              '#8B5CF6', badge="Internal Margin")
    draw_arrow(58, 37, 76, 33, label="Val_A", label_dy=1.5)
    draw_arrow(58, 23, 76, 29, label="Val_B", label_dy=-1.5)

    # Column 5: Refit & Decision Regimes
    draw_card(97, 20, 21, 22, "Decision & Full Refit", 
              "1. Pure A (Delta > 0.08):\n   Refit Branch A on 100% X_tr;\n   Inference skips Branch B.\n2. Pure B (Delta < -0.08):\n   Refit Branch B on 100% X_tr;\n   Inference skips Branch A.\n3. Blend (|Delta| <= 0.08):\n   Refit Both Experts;\n   w_A P_A + (1-w_A) P_B", 
              '#EC4899', badge="Conditional Refit")
    draw_arrow(93, 31, 97, 31)

    # Test-time Inference Arrow to Output
    ax.text(107.5, 9.5, "Predicted Class y_hat", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.0))
    draw_arrow(107.5, 20, 107.5, 12, label="X_test Inference", color='#2563EB', label_dy=1.4)

    plt.tight_layout()
    plt.savefig("paper/figures/cadence_architecture.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/cadence_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved clean, non-overlapping cadence_architecture.pdf/.png")

# -------------------------------------------------------------
# 2. Figure 2: Routing Mechanism (Clean Notation, Non-overlapping)
# -------------------------------------------------------------
def plot_routing_mechanism():
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    
    delta = np.linspace(-0.25, 0.25, 600)
    w_A = np.zeros_like(delta)
    
    for i, d in enumerate(delta):
        if d > 0.08:
            w_A[i] = 1.0
        elif d < -0.08:
            w_A[i] = 0.0
        else:
            w_A[i] = max(0.15, min(0.85, 0.5 + 2.0 * d))
            
    # Shaded Regimes
    ax.axvspan(-0.25, -0.08, color='#FEF3C7', alpha=0.5, label='Regime I: Pure Branch B (Delta < -0.08)')
    ax.axvspan(-0.08, 0.08, color='#F3E8FF', alpha=0.5, label='Regime II: Competitive Blend (|Delta| <= 0.08)')
    ax.axvspan(0.08, 0.25, color='#DCFCE7', alpha=0.5, label='Regime III: Pure Branch A (Delta > +0.08)')
    
    # Weight Curves
    ax.plot(delta, w_A, color='#2563EB', lw=2.4, label='Branch A Weight (w_A)')
    ax.plot(delta, 1.0 - w_A, color='#D97706', lw=2.4, linestyle='--', label='Branch B Weight (w_B = 1 - w_A)')
    
    # Guidelines
    ax.axvline(-0.08, color='#64748B', linestyle=':', lw=1.4)
    ax.axvline(0.08, color='#64748B', linestyle=':', lw=1.4)
    
    # Annotations placed cleanly outside the curves
    ax.text(-0.165, 0.75, "Pure Branch B Dominance\n(Inference skips Branch A)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#92400E')
    ax.text(0.165, 0.75, "Pure Branch A Dominance\n(Inference skips Branch B)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#065F46')
    ax.text(0.0, 0.18, "Competitive Blending Regime:\nw_A = 0.5 + 2.0*Delta\nStep transition at boundaries", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#5B21B6',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='#DDD6FE', lw=0.8))

    ax.set_xlabel('Internal Validation Margin: Delta = Val_A - Val_B', fontweight='bold')
    ax.set_ylabel('Effective Prediction Weight Assigned to Expert', fontweight='bold')
    ax.set_title('Confidence-Adaptive Meta-Routing: Operating Regimes (tau = +-0.08)', fontweight='bold')
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, linestyle=':', alpha=0.5, color='#94A3B8')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#CBD5E1')
    
    plt.tight_layout()
    plt.savefig("paper/figures/routing_mechanism.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/routing_mechanism.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved clean routing_mechanism.pdf/.png")

# -------------------------------------------------------------
# 3. Figure 3: Pairwise Scatter with Wins and Deficits
# -------------------------------------------------------------
def plot_pairwise_scatter():
    df = pd.read_csv("results_v3/summary_results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=300)
    
    cadence = df["mean_acc"]
    hmr = df["baseline_hydra_multirocket"]
    hc2 = df["baseline_hc2"]
    
    wins_hmr = (cadence - hmr > 0.0001).sum()
    ties_hmr = ((cadence - hmr).abs() <= 0.0001).sum()
    loss_hmr = (cadence - hmr < -0.0001).sum()
    
    wins_hc2 = (cadence - hc2 > 0.0001).sum()
    ties_hc2 = ((cadence - hc2).abs() <= 0.0001).sum()
    loss_hc2 = (cadence - hc2 < -0.0001).sum()

    for ax, base, title, wins, ties, loss, dot_color in [
        (ax1, hmr, f"CADENCE vs Hydra+MultiRocket\n(Wins: {wins_hmr} | Ties: {ties_hmr} | Losses: {loss_hmr})", wins_hmr, ties_hmr, loss_hmr, '#2563EB'),
        (ax2, hc2, f"CADENCE vs HIVE-COTE 2.0\n(Wins: {wins_hc2} | Ties: {ties_hc2} | Losses: {loss_hc2})", wins_hc2, ties_hc2, loss_hc2, '#7C3AED')
    ]:
        ax.fill_between([0.2, 1.05], [0.2, 1.05], 1.05, color='#DCFCE7', alpha=0.35, label='CADENCE Win Zone')
        ax.fill_between([0.2, 1.05], 0.2, [0.2, 1.05], color='#F1F5F9', alpha=0.35, label='Baseline Win Zone')
        ax.scatter(base, cadence, color=dot_color, alpha=0.82, edgecolors='white', linewidth=0.6, s=34, zorder=5)
        ax.plot([0.2, 1.05], [0.2, 1.05], color='#EF4444', linestyle='--', lw=1.3, label='Parity (y = x)', zorder=4)
        
        ax.set_xlabel("Baseline Accuracy (30-Resample Mean)", fontweight='bold')
        ax.set_ylabel("CADENCE Accuracy (Ours)", fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(0.25, 1.02)
        ax.grid(True, linestyle=':', alpha=0.5, color='#94A3B8')
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#CBD5E1', fontsize=7.8)

    # Non-overlapping annotations with callout boxes
    # Top gains over Hydra+MR
    ax1.annotate("SemgHandMovementCh2 (+9.96 pp)", (0.7717, 0.8713), xytext=(-35, 14), textcoords="offset points",
                 fontsize=6.8, fontweight='bold', color='#0F172A',
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECFDF5', edgecolor='#10B981', lw=0.6))
    ax1.annotate("InlineSkate (+9.66 pp)", (0.5081, 0.6047), xytext=(10, 10), textcoords="offset points",
                 fontsize=6.8, fontweight='bold', color='#0F172A',
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECFDF5', edgecolor='#10B981', lw=0.6))
    # Deficit on ShapesAll
    ax1.annotate("ShapesAll (-4.26 pp)", (0.9483, 0.9057), xytext=(-80, -18), textcoords="offset points",
                 fontsize=6.8, fontweight='bold', color='#0F172A',
                 arrowprops=dict(arrowstyle="->", color='#DC2626', lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#FEF2F2', edgecolor='#EF4444', lw=0.6))

    # Annotations for HC2 plot
    ax2.annotate("DistalPhalanxTW (+7.29 pp)", (0.7017, 0.7746), xytext=(-40, 14), textcoords="offset points",
                 fontsize=6.8, fontweight='bold', color='#0F172A',
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECFDF5', edgecolor='#10B981', lw=0.6))
    # Deficit on PigAirwayPressure
    ax2.annotate("PigAirwayPressure (-13.35 pp)", (0.9554, 0.8220), xytext=(-110, -16), textcoords="offset points",
                 fontsize=6.8, fontweight='bold', color='#0F172A',
                 arrowprops=dict(arrowstyle="->", color='#DC2626', lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#FEF2F2', edgecolor='#EF4444', lw=0.6))

    plt.tight_layout()
    plt.savefig("paper/figures/pairwise_scatter.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/pairwise_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pairwise_scatter.pdf/.png with readable annotations")

# -------------------------------------------------------------
# 4. Figure 4: Model Rankings & Accurate Pareto Frontier
# -------------------------------------------------------------
def plot_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
    
    # Accurate cluster runtimes across 112/109 datasets (seconds per dataset):
    # TS-CHIEF: 1016.87h / 112 = 9.08h = 32,685 s
    # HC1: 427.18h / 112 = 3.81h = 13,730 s
    # HC2: 340.21h / 112 = 3.04h = 10,935 s
    # InceptionTime (GPU): 86.58h / 112 = 0.77h = 2,780 s
    # STC: 115.88h / 112 = 3,725 s
    # TDE: 75.41h / 112 = 2,424 s
    # DrCIF: 45.40h / 112 = 1,460 s
    # MultiRocket (100k): 42.10 s
    # CADENCE (Ours): 17.53 s (dual-core i7)
    # Hydra+MultiRocket: 14.20 s
    # MultiRocket (50k): 11.50 s
    # Hydra: 5.40 s
    # MiniRocket: 2.10 s
    models = [
        {"name": "CADENCE (Ours)", "acc": 0.8864, "time": 17.53, "color": "#2563EB", "marker": "*", "size": 220},
        {"name": "Hydra+MultiRocket", "acc": 0.8818, "time": 14.20, "color": "#0D9488", "marker": "o", "size": 95},
        {"name": "MultiRocket (100k)", "acc": 0.8800, "time": 42.10, "color": "#64748B", "marker": "o", "size": 75},
        {"name": "MultiRocket (50k)", "acc": 0.8797, "time": 11.50, "color": "#64748B", "marker": "o", "size": 75},
        {"name": "MiniRocket", "acc": 0.8724, "time": 2.10, "color": "#64748B", "marker": "o", "size": 75},
        {"name": "Hydra", "acc": 0.8714, "time": 5.40, "color": "#64748B", "marker": "o", "size": 75},
        {"name": "InceptionTime (GPU)", "acc": 0.8721, "time": 2780.0, "color": "#DC2626", "marker": "^", "size": 85},
        {"name": "HIVE-COTE 2.0", "acc": 0.8895, "time": 10935.0, "color": "#7C3AED", "marker": "D", "size": 110},
        {"name": "HIVE-COTE 1.0", "acc": 0.8786, "time": 13730.0, "color": "#9333EA", "marker": "D", "size": 80},
        {"name": "TS-CHIEF", "acc": 0.8761, "time": 32685.0, "color": "#D97706", "marker": "s", "size": 85},
    ]
    
    for m in models:
        ax.scatter(m["time"], m["acc"], color=m["color"], marker=m["marker"], s=m["size"], 
                   edgecolor='white', linewidth=0.8, zorder=5)
        offset_y = 0.0015 if m["name"] not in ["Hydra", "MultiRocket (50k)", "TS-CHIEF"] else -0.0024
        offset_x = 1.18 if m["name"] not in ["HIVE-COTE 2.0", "HIVE-COTE 1.0", "TS-CHIEF"] else 0.42
        fontweight = 'bold' if 'CADENCE' in m['name'] else 'normal'
        ax.annotate(m["name"], (m["time"] * offset_x, m["acc"] + offset_y), fontsize=7.6, fontweight=fontweight, color='#1E293B')
        
    # True Pareto curve: connects best accuracy at each time scale
    pareto_times = [2.10, 11.50, 14.20, 17.53, 10935.0]
    pareto_accs = [0.8724, 0.8797, 0.8818, 0.8864, 0.8895]
    ax.plot(pareto_times, pareto_accs, color='#3B82F6', linestyle='--', lw=1.8, zorder=3, alpha=0.8, label='Empirical Pareto Frontier')
    ax.fill_between(pareto_times, pareto_accs, 0.865, color='#EFF6FF', alpha=0.45, zorder=1)

    ax.set_xscale('log')
    ax.set_xlabel("Mean Training Runtime per Dataset (Seconds, Log Scale)", fontweight='bold')
    ax.set_ylabel("Grand Mean Accuracy (109 UCR Datasets)", fontweight='bold')
    ax.set_title("Accuracy vs. Training Runtime: Empirical Pareto Frontier", fontweight='bold')
    ax.set_ylim(0.865, 0.895)
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#94A3B8')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1')
    
    plt.tight_layout()
    plt.savefig("paper/figures/pareto_frontier.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/pareto_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved accurate pareto_frontier.pdf/.png")

# -------------------------------------------------------------
# 5. Figure 5: Ablation Plot with Percentage Points (pp)
# -------------------------------------------------------------
def plot_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    
    stages = [
        "Branch A Only\n(MiniRocket)",
        "Branch B Only\n(Hydra+MQ+FFT)",
        "Fixed 50/50 Blend\n(Naive Average)",
        "Static Routing\n(K >= 12 Split)",
        "CADENCE\n(Adaptive Router)"
    ]
    accs = [0.8724, 0.8729, 0.8741, 0.8813, 0.8864]
    colors = ['#94A3B8', '#94A3B8', '#F87171', '#60A5FA', '#2563EB']
    
    bars = ax.bar(stages, accs, color=colors, width=0.50, edgecolor='#1E293B', linewidth=0.8, zorder=3)
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0006, f"{h:.4f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0F172A')
        diff_from_full = (h - 0.8864) * 100
        diff_str = f"{diff_from_full:.2f} pp" if i < 4 else "Full System"
        ax.text(bar.get_x() + bar.get_width()/2, 0.867, diff_str,
                ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1E40AF',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='#DBEAFE', edgecolor='none'))
                
    ax.set_ylim(0.864, 0.892)
    ax.set_ylabel("Grand Mean Accuracy (109 Datasets, 30 Resamples)", fontweight='bold')
    ax.set_title("Component Ablations: Isolated Models vs. Adaptive Dynamic Routing", fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.5, color='#94A3B8')
    
    plt.tight_layout()
    plt.savefig("paper/figures/ablation.pdf", bbox_inches='tight')
    plt.savefig("paper/figures/ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved clean ablation.pdf/.png with percentage points")

if __name__ == "__main__":
    plot_architecture()
    plot_routing_mechanism()
    plot_pairwise_scatter()
    plot_pareto_frontier()
    plot_ablation()
    print("All figures successfully updated!")
