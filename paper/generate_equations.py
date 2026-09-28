#!/usr/bin/env python3
"""
Publication-Grade LaTeX Equation Renderer for CADENCE Manuscript.
Generates line-by-line formatted, high-resolution mathematical cards
for Equations (1) through (6).
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use Computer Modern math font matching standard LaTeX documents
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'

OUT_DIR = "paper/figures/equations"
os.makedirs(OUT_DIR, exist_ok=True)

def render_card(lines, eq_num, filename, figsize, y_pos, fontsize=9.2):
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Modern card background with soft rounded border and subtle fill
    card = patches.FancyBboxPatch(
        (0.005, 0.04), 0.99, 0.92,
        boxstyle='round,pad=0.015,rounding_size=0.04',
        facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.9,
        transform=ax.transAxes
    )
    ax.add_patch(card)
    
    # Left accent indicator bar (royal blue)
    accent = patches.FancyBboxPatch(
        (0.006, 0.12), 0.007, 0.76,
        boxstyle='round,pad=0.002,rounding_size=0.004',
        facecolor='#2563EB', edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(accent)
    
    # Render lines
    for text, y in zip(lines, y_pos):
        ax.text(0.035, y, text, fontsize=fontsize, va='center', ha='left', color='#0F172A', transform=ax.transAxes)
        
    # Equation number badge on the right
    ax.text(0.965, 0.5, f"({eq_num})", fontsize=9.2, fontweight='bold', va='center', ha='right', color='#475569', transform=ax.transAxes)
    
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")

def generate_all_equations():
    # Equation 1: Convolution and PPV Pooling (2 lines)
    render_card([
        r"$\mathbf{Convolution:}\quad A_t = (X *_d \omega)_t = \sum_{j=0}^{8} \omega_j \cdot X_{t + j \cdot d}$",
        r"$\mathbf{PPV\;Pooling:}\quad \phi_{\mathrm{PPV}}(X, \omega, b) = \frac{1}{L - 8d} \sum_{t=1}^{L - 8d} \mathbf{1}(A_t > b)$"
    ], 1, "eq1.png", figsize=(7.2, 0.88), y_pos=[0.70, 0.30])

    # Equation 2: Woodbury Regularized Solver (2 lines)
    render_card([
        r"$\mathbf{Primal\;Form}\;(N \geq D_A):\quad W^* = (\widetilde{F}_A^\top \widetilde{F}_A + \alpha I_{D_A})^{-1} \widetilde{F}_A^\top Y$",
        r"$\mathbf{Dual\;Woodbury}\;(N < D_A):\quad W^* = \widetilde{F}_A^\top (\widetilde{F}_A \widetilde{F}_A^\top + \alpha I_N)^{-1} Y$"
    ], 2, "eq2.png", figsize=(7.2, 0.88), y_pos=[0.70, 0.30])

    # Equation 3: Softmax Probabilities and Linear Decision Scoring (2 lines)
    render_card([
        r"$\mathbf{Softmax\;Posterior:}\quad P_A(y = k \mid X) = \frac{\exp(f_{A, k}(X))}{\sum_{j=1}^K \exp(f_{A, j}(X))}$",
        r"$\mathbf{Linear\;Scoring:}\quad f_A(X) = W^{*\top} \widetilde{F}_A(X) + b_A$"
    ], 3, "eq3.png", figsize=(7.2, 0.88), y_pos=[0.70, 0.30])

    # Equation 4: Cornish-Fisher Quantile Expansion (2 lines)
    render_card([
        r"$\mathbf{Quantile\;Approximation:}\quad q_\alpha(S) = \hat{\mu} + \hat{\sigma} \cdot \Psi(z_\alpha, \hat{S}, \hat{K}), \quad \mathrm{where}\; z_\alpha = \Phi^{-1}(\alpha)$",
        r"$\mathbf{Asymptotic\;Expansion:}\quad \Psi(z_\alpha, \hat{S}, \hat{K}) = z_\alpha + \frac{\hat{S}}{6}(z_\alpha^2 - 1) + \frac{\hat{K}}{24}(z_\alpha^3 - 3z_\alpha) - \frac{\hat{S}^2}{36}(2z_\alpha^3 - 5z_\alpha)$"
    ], 4, "eq4.png", figsize=(7.2, 0.88), y_pos=[0.70, 0.30], fontsize=8.6)

    # Equation 5: Piecewise Meta-Routing Decision Rule (3 lines)
    render_card([
        r"$\mathbf{Regime\;III}\;(\Delta > \tau):\quad \hat{y}(X) = \arg\max_k P_A(y = k \mid X) \quad [\mathrm{Pure\;Branch\;A}]$",
        r"$\mathbf{Regime\;I}\;(\Delta < -\tau):\quad \hat{y}(X) = \arg\max_k P_B(y = k \mid X) \quad [\mathrm{Pure\;Branch\;B}]$",
        r"$\mathbf{Regime\;II}\;(|\Delta| \leq \tau):\quad \hat{y}(X) = \arg\max_k P_{\mathrm{blend}}(y = k \mid X) \quad [\mathrm{Confidence\;Blending}]$"
    ], 5, "eq5.png", figsize=(7.2, 1.18), y_pos=[0.78, 0.50, 0.22])

    # Equation 6: Soft Confidence Blending & Adaptive Blend Weight (2 lines)
    render_card([
        r"$\mathbf{Soft\;Blend:}\quad P_{\mathrm{blend}}(y = k \mid X) = w_A P_A(y = k \mid X) + (1 - w_A) P_B(y = k \mid X)$",
        r"$\mathbf{Adaptive\;Weight:}\quad w_A = \mathrm{clip}(0.5 + 2.0 \cdot \Delta, \; 0.15, \; 0.85)$"
    ], 6, "eq6.png", figsize=(7.2, 0.88), y_pos=[0.70, 0.30])

if __name__ == "__main__":
    generate_all_equations()
