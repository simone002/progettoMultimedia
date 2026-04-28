from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# questo file si occupa di leggere i risultati della crossval subject-wise
#  (generati da crossval_subjectwise.py) e creare dei plot riassuntivi per 
# confronto modelli, trend per fold e confusions più frequenti.

BASE_DIR = Path(__file__).resolve().parent
CV_DIR = BASE_DIR / "output" / "classification_advanced" / "organized" / "02_cv_subjectwise"
SUMMARY_CSV = CV_DIR / "cv_subjectwise_summary.csv"
RESULTS_CSV = CV_DIR / "cv_subjectwise_results.csv"
CONFUSIONS_CSV = CV_DIR / "cv_top_confusions.csv"
PLOTS_DIR = CV_DIR / "plots"


def _format_model_name(name: str) -> str:
    return name.replace("_", " ").title()


def plot_cv_summary(summary_df: pd.DataFrame) -> Path:
    labels = [_format_model_name(m) for m in summary_df["model"].astype(str).tolist()]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        x,
        summary_df["accuracy_mean"],
        yerr=summary_df["accuracy_std"],
        capsize=5,
        color="#4C78A8",
        alpha=0.9,
    )
    axes[0].set_title("CV Subject-wise: Accuracy")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(
        x,
        summary_df["macro_f1_mean"],
        yerr=summary_df["macro_f1_std"],
        capsize=5,
        color="#F58518",
        alpha=0.9,
    )
    axes[1].set_title("CV Subject-wise: Macro-F1")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylim(0.6, 1.0)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = PLOTS_DIR / "cv_subjectwise_summary.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_fold_trends(results_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for model_name, model_df in results_df.groupby("model"):
        model_df = model_df.sort_values("fold")
        label = _format_model_name(str(model_name))

        axes[0].plot(model_df["fold"], model_df["accuracy"], marker="o", linewidth=2, label=label)
        axes[1].plot(model_df["fold"], model_df["macro_f1"], marker="o", linewidth=2, label=label)

    axes[0].set_title("Accuracy per fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.6, 0.9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title("Macro-F1 per fold")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.6, 0.85)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="best")

    plt.tight_layout()
    out_path = PLOTS_DIR / "cv_subjectwise_fold_trends.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_top_confusions(conf_df: pd.DataFrame, top_n: int = 15) -> Path:
    conf_df = conf_df.head(top_n).copy()
    conf_df["pair"] = conf_df["true"].astype(str) + "→" + conf_df["pred"].astype(str)
    conf_df = conf_df.iloc[::-1]

    fig_h = max(6, top_n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(conf_df["pair"], conf_df["count"], color="#54A24B", alpha=0.9)
    ax.set_title(f"Top-{top_n} confusioni (best model CV)")
    ax.set_xlabel("Conteggio")
    ax.set_ylabel("Vera → Predetta")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = PLOTS_DIR / "cv_subjectwise_top_confusions.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {SUMMARY_CSV}. Esegui prima crossval_subjectwise.py")
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {RESULTS_CSV}. Esegui prima crossval_subjectwise.py")
    if not CONFUSIONS_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {CONFUSIONS_CSV}. Esegui prima crossval_subjectwise.py")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(SUMMARY_CSV)
    results_df = pd.read_csv(RESULTS_CSV)
    conf_df = pd.read_csv(CONFUSIONS_CSV)

    p1 = plot_cv_summary(summary_df)
    p2 = plot_fold_trends(results_df)
    p3 = plot_top_confusions(conf_df, top_n=15)

    print("Plot generati:")
    print(f"- {p1}")
    print(f"- {p2}")
    print(f"- {p3}")


if __name__ == "__main__":
    main()
