"""Save the baseline-vs-v1 test charts from the fitted regression.

Run from the repo root with the Python that has the scientific packages:

    python ml/charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regression import fit_regression

OUT_DIR = Path(__file__).resolve().parent / "charts"


def _style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_facecolor("white")


def bar_chart(path, labels, values, ylabel, title, value_format):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=["#8a8a8a", "#1a1a1a"], width=0.62)
    _style(ax)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    top = max(values)
    bottom = min(0, min(values))
    span = top - bottom if top != bottom else 1
    ax.set_ylim(bottom - span * 0.06, top + span * 0.2)
    for bar, value in zip(bars, values):
        # A near-zero bar has no interior, so the number sits just above the axis.
        label_y = value + span * 0.035 if value > span * 0.04 else span * 0.045
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=13,
            color="#111111",
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    fitted = fit_regression()
    test_fit = fitted["test_fit"]
    baseline_test = fitted["baseline_test"]
    n_test = len(fitted["test"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Baseline\n(training average)", "v1 regression"]
    error_path = OUT_DIR / "test_typical_error.png"
    r2_path = OUT_DIR / "test_r2.png"
    bar_chart(
        error_path,
        labels,
        [baseline_test["mae_days"], test_fit["mae_days"]],
        "Typical error (days)",
        f"Test filings (n = {n_test:,}): typical error",
        "{:.2f}",
    )
    bar_chart(
        r2_path,
        labels,
        [baseline_test["r2"], test_fit["r2"]],
        "R²",
        f"Test filings (n = {n_test:,}): R²",
        "{:.4f}",
    )
    print(f"Baseline test MAE {baseline_test['mae_days']:.4f}   v1 test MAE {test_fit['mae_days']:.4f}")
    print(f"Baseline test R²  {baseline_test['r2']:.4f}   v1 test R²  {test_fit['r2']:.4f}")
    print(error_path)
    print(r2_path)


if __name__ == "__main__":
    main()
