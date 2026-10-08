"""
shap_plot_figure3.py
=====================
Builds the FINAL publication-ready six-panel Fig. 3 (global SHAP feature
importance) strictly from shap_global_values.csv.

SPECIFICATIONS:
- Panels:
    (a) CVD -- Mode A
    (b) CVD -- Mode B
    (c) Diabetes -- Mode A
    (d) Diabetes -- Mode B
    (e) Hypertension -- Mode A
    (f) Hypertension -- Mode B
- Bars sorted in descending mean absolute SHAP value with largest at the top.
- Actual numerical mean |SHAP| value displayed at the end of each bar with
  consistent 5-decimal-place precision across all panels (e.g. age = 0.06419,
  smoking_status = 0.01828).
- Independent x-axis scale per panel reflecting model-specific output spaces.
- Clean IEEE two-column publication style (7.16 in width, clean white background,
  restrained color, sans-serif typography, no decorative graphics/3D/gradients).
- Saves:
    fig3_global_shap_feature_importance.png  (300 DPI)
    fig3_global_shap_feature_importance.pdf
    fig3_global_shap_feature_importance.svg
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "Helvetica"]
plt.rcParams["svg.fonttype"] = "none"

NAVY_DARK = "#12283E"
BAR_COLOR = "#2E6E9E"   # Single consistent, publication-standard steel blue
SUBINK = "#5B6B7B"
INK = "#22303F"
SPINE_COLOR = "#B9C6D3"

CSV_PATH = "shap_global_values.csv"

PANEL_ORDER = [
    ("a", "CVD", "A"),
    ("b", "CVD", "B"),
    ("c", "Diabetes", "A"),
    ("d", "Diabetes", "B"),
    ("e", "Hypertension", "A"),
    ("f", "Hypertension", "B"),
]


def main():
    csv_file = Path(CSV_PATH)
    if not csv_file.exists():
        sys.exit(
            f"'{CSV_PATH}' not found. Run shap_compute.py first to generate the "
            f"verified SHAP feature-importance dataset."
        )

    df = pd.read_csv(csv_file)
    required_cols = {"target", "mode", "feature", "mean_abs_shap", "rank"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"'{CSV_PATH}' is missing required columns: {required_cols - set(df.columns)}")

    missing_panels = []
    for _, tgt, mode in PANEL_ORDER:
        sub = df[(df.target == tgt) & (df["mode"] == mode)]
        if len(sub) < 10:
            missing_panels.append(f"{tgt}-{mode} ({len(sub)}/10 rows)")
    if missing_panels:
        sys.exit(
            "Refusing to build figure: fewer than 10 rows for:\n  " +
            "\n  ".join(missing_panels)
        )

    # IEEE two-column standard width (7.16 inches)
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 8.4))

    for idx, (letter, tgt, mode) in enumerate(PANEL_ORDER):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]

        sub = df[(df.target == tgt) & (df["mode"] == mode)].sort_values("rank")

        # Largest at top -> reverse order for horizontal bar chart
        feats = sub["feature"].tolist()[::-1]
        vals = sub["mean_abs_shap"].tolist()[::-1]
        y_pos = list(range(len(feats)))

        ax.barh(y_pos, vals, height=0.62, color=BAR_COLOR, edgecolor="none", zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=6.8, color=INK)

        vmax = max(vals) if vals else 1.0
        # Independent x-axis scale per panel with headroom for numerical value labels
        ax.set_xlim(0, vmax * 1.30)

        # Consistent 5-decimal-place precision across all panels
        for yp, v in zip(y_pos, vals):
            ax.text(
                v + vmax * 0.02,
                yp,
                f"{v:.5f}",
                fontsize=6.2,
                color=INK,
                va="center",
                ha="left",
            )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)
        ax.tick_params(axis="both", length=2.5, labelsize=6.2, colors=SUBINK)
        ax.set_title(
            f"({letter}) {tgt} \u2014 Mode {mode}",
            fontsize=8.6,
            fontweight="bold",
            color=NAVY_DARK,
            pad=5,
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=6.4, color=SUBINK, labelpad=3)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.48, wspace=0.55)

    png_out = "fig3_global_shap_feature_importance.png"
    pdf_out = "fig3_global_shap_feature_importance.pdf"
    svg_out = "fig3_global_shap_feature_importance.svg"

    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated:")
    print(f"  - {png_out}")
    print(f"  - {pdf_out}")
    print(f"  - {svg_out}")


if __name__ == "__main__":
    main()