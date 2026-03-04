from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
OUTPUT_ROOT = BASE_DIR / "output" / "classification_advanced" / "organized"
OUTPUT_DIR = OUTPUT_ROOT / "03_ablation"
SPLIT_CACHE_PATH = OUTPUT_ROOT / "05_cache_split" / "subject_holdout_split.json"
TARGET_COL = "lettera"
SUBJECT_COL = "soggetto"


BASE_CORE_COLS = ["punte", "incroci", "buchi", "componenti", "aspect_ratio", "densita_pixel"]
ENGINEERED_INTERACTION_COLS = [
    "complexity_index",
    "punte_x_buchi",
    "ink_compactness",
    "components_per_endpoint",
]
ADV_TOPOLOGY_COLS = ["endpoint_norm", "junction_norm", "euler_number", "hole_area_ratio", "pruning_removed_ratio"]
STROKE_GEOMETRY_COLS = ["avg_stroke_width", "skeleton_length", "skeleton_density"]
GLOBAL_SHAPE_COLS = [
    "solidity",
    "extent",
    "eccentricity",
    "orientation",
    "major_axis_length",
    "minor_axis_length",
    "axis_ratio",
]
SKELETON_BRANCH_COLS = ["n_branches", "branch_length_mean", "branch_length_max", "branch_length_std"]
SPATIAL_BALANCE_COLS = [
    "top_half_density",
    "bottom_half_density",
    "left_half_density",
    "right_half_density",
    "top_bottom_ratio",
    "left_right_ratio",
]
DIAGONAL_PROFILE_COLS = ["main_diag_density", "anti_diag_density", "diag_density_diff"]
ENDPOINT_POSITION_COLS = ["endpoint_y_mean_norm", "endpoint_x_mean_norm", "endpoint_y_std_norm", "endpoint_x_std_norm"]
SCANLINE_STRUCTURE_COLS = [
    "row_q25_runs",
    "row_q50_runs",
    "row_q75_runs",
    "row_q25_fill",
    "row_q50_fill",
    "row_q75_fill",
    "col_q25_runs",
    "col_q50_runs",
    "col_q75_runs",
    "col_q25_fill",
    "col_q50_fill",
    "col_q75_fill",
]
BORDER_CONTACT_COLS = ["top_border_contact", "bottom_border_contact", "left_border_contact", "right_border_contact"]
HOLE_POSITION_COLS = ["hole_centroid_y_norm", "hole_centroid_x_norm", "hole_centroid_spread"]
CONTOUR_DEPTH_COLS = [
    "top_depth_mean_norm",
    "top_depth_std_norm",
    "top_depth_range_norm",
    "bottom_depth_mean_norm",
    "bottom_depth_std_norm",
    "bottom_depth_range_norm",
    "top_center_depth_mean_norm",
    "bottom_center_depth_mean_norm",
]
ENDPOINT_DISTRIBUTION_COLS = [
    "endpoints_top_ratio",
    "endpoints_bottom_ratio",
    "endpoints_left_ratio",
    "endpoints_right_ratio",
    "endpoints_lower_quarter_ratio",
]
MIDLINE_RUN_COLS = [
    "mid_row_fg_runs",
    "mid_row_bg_runs",
    "mid_row_fg_fill",
    "mid_row_right_bg_tail_norm",
    "mid_col_fg_runs",
    "mid_col_fg_fill",
]
UPPER_LOWER_DENSITY_COLS = ["upper_third_density", "lower_third_density", "upper_lower_density_ratio"]


def _present(all_feature_cols: list[str], cols: list[str]) -> list[str]:
    all_set = set(all_feature_cols)
    return [c for c in cols if c in all_set]


def _present_by_prefix(all_feature_cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in all_feature_cols if c.startswith(prefixes)]


def build_feature_groups(all_feature_cols: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    candidates = [
        ("base_core", _present(all_feature_cols, BASE_CORE_COLS)),
        ("engineered_interactions", _present(all_feature_cols, ENGINEERED_INTERACTION_COLS)),
        ("advanced_topology", _present(all_feature_cols, ADV_TOPOLOGY_COLS)),
        ("stroke_geometry", _present(all_feature_cols, STROKE_GEOMETRY_COLS)),
        ("global_shape", _present(all_feature_cols, GLOBAL_SHAPE_COLS)),
        ("skeleton_branches", _present(all_feature_cols, SKELETON_BRANCH_COLS)),
        ("hu_moments", _present_by_prefix(all_feature_cols, ("hu_",))),
        ("zoning", _present_by_prefix(all_feature_cols, ("zone_",))),
        ("projection_profiles", _present_by_prefix(all_feature_cols, ("row_proj_", "col_proj_"))),
        ("spatial_balance", _present(all_feature_cols, SPATIAL_BALANCE_COLS)),
        ("transition_profiles", _present_by_prefix(all_feature_cols, ("row_trans_", "col_trans_"))),
        ("diagonal_profile", _present(all_feature_cols, DIAGONAL_PROFILE_COLS)),
        ("endpoint_position", _present(all_feature_cols, ENDPOINT_POSITION_COLS)),
        ("scanline_structure", _present(all_feature_cols, SCANLINE_STRUCTURE_COLS)),
        ("border_contact", _present(all_feature_cols, BORDER_CONTACT_COLS)),
        ("hole_position", _present(all_feature_cols, HOLE_POSITION_COLS)),
        ("contour_depth", _present(all_feature_cols, CONTOUR_DEPTH_COLS)),
        ("endpoint_distribution", _present(all_feature_cols, ENDPOINT_DISTRIBUTION_COLS)),
        ("midline_runs", _present(all_feature_cols, MIDLINE_RUN_COLS)),
        ("upper_lower_density", _present(all_feature_cols, UPPER_LOWER_DENSITY_COLS)),
    ]

    groups: dict[str, list[str]] = {}
    feature_to_group: dict[str, str] = {}
    used: set[str] = set()

    for group_name, group_cols in candidates:
        unique_cols = [c for c in group_cols if c not in used]
        if not unique_cols:
            continue
        groups[group_name] = unique_cols
        for col in unique_cols:
            feature_to_group[col] = group_name
        used.update(unique_cols)

    unassigned = [c for c in all_feature_cols if c not in used]
    if unassigned:
        groups["other_features"] = unassigned
        for col in unassigned:
            feature_to_group[col] = "other_features"

    return groups, feature_to_group


def split_by_subject(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    random_state: int = 42,
    cache_path: Path | None = None,
):
    if cache_path is not None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            train_subjects = set(map(str, payload.get("train_subjects", [])))
            test_subjects = set(map(str, payload.get("test_subjects", [])))
            if train_subjects and test_subjects:
                train_df = df[df[SUBJECT_COL].astype(str).isin(train_subjects)].copy()
                test_df = df[df[SUBJECT_COL].astype(str).isin(test_subjects)].copy()
                if not train_df.empty and not test_df.empty:
                    print(f"[split] loaded cached split from: {cache_path}")
                    return train_df, test_df, sorted(train_subjects), sorted(test_subjects)
        except Exception:
            pass

    subjects = np.array(sorted(df[SUBJECT_COL].astype(str).unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "test_ratio": test_ratio,
                    "random_state": random_state,
                    "train_subjects": sorted(train_subjects),
                    "test_subjects": sorted(test_subjects),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[split] saved split cache to: {cache_path}")

    train_df = df[df[SUBJECT_COL].astype(str).isin(train_subjects)].copy()
    test_df = df[df[SUBJECT_COL].astype(str).isin(test_subjects)].copy()
    return train_df, test_df, sorted(train_subjects), sorted(test_subjects)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    out["complexity_index"] = out["punte"] + 2.0 * out["incroci"] + out["buchi"]
    out["punte_x_buchi"] = out["punte"] * out["buchi"]
    out["ink_compactness"] = out["densita_pixel"] / (out["aspect_ratio"] + eps)
    out["components_per_endpoint"] = out["componenti"] / (out["punte"] + 1.0)
    return out


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COL, SUBJECT_COL, "file"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def build_xgb_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )


def evaluate_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_encoder: LabelEncoder,
):
    if not feature_cols:
        raise ValueError("Feature list vuota nella configurazione di ablation")

    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]

    y_train = train_df[TARGET_COL].astype(str)
    y_test = test_df[TARGET_COL].astype(str)

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_test_imp = imputer.transform(x_test)

    y_train_enc = label_encoder.transform(y_train)

    model = build_xgb_model()
    model.fit(x_train_imp, y_train_enc)

    y_pred_enc = model.predict(x_test_imp)
    y_pred = label_encoder.inverse_transform(np.asarray(y_pred_enc, dtype=int))

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    return accuracy, macro_f1


def save_delta_plot(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df[summary_df["removed_group"] != "none"].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("delta_macro_f1_vs_all", ascending=True)
    colors = ["#d62728" if value < 0 else "#2ca02c" for value in plot_df["delta_macro_f1_vs_all"]]

    fig_h = max(6, 0.35 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["removed_group"], plot_df["delta_macro_f1_vs_all"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Delta Macro-F1 vs all features")
    ax.set_ylabel("Gruppo rimosso")
    ax.set_title("Ablation XGBoost: effetto della rimozione di ciascun gruppo")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV}. Esegui prima analyze_letter.py")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = add_engineered_features(df)

    required = {TARGET_COL, SUBJECT_COL}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    all_feature_cols = get_numeric_feature_columns(df)
    if not all_feature_cols:
        raise ValueError("Nessuna feature numerica trovata per l'ablation study")

    feature_groups, feature_to_group = build_feature_groups(all_feature_cols)
    if not feature_groups:
        raise ValueError("Nessun gruppo di feature disponibile per l'ablation")

    train_df, test_df, train_subjects, test_subjects = split_by_subject(
        df,
        test_ratio=0.25,
        random_state=42,
        cache_path=SPLIT_CACHE_PATH,
    )

    classes = sorted(df[TARGET_COL].astype(str).unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    coverage_rows = []
    for col in all_feature_cols:
        coverage_rows.append({"feature": col, "group": feature_to_group.get(col, "unassigned")})
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["group", "feature"])
    coverage_path = OUTPUT_DIR / "ablation_feature_group_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)

    print("[Ablation] Baseline (all features)...")
    base_acc, base_f1 = evaluate_config(train_df, test_df, all_feature_cols, label_encoder)

    rows = [
        {
            "setting": "all_features",
            "removed_group": "none",
            "removed_columns": "",
            "n_features": len(all_feature_cols),
            "group_size_removed": 0,
            "accuracy": base_acc,
            "macro_f1": base_f1,
            "delta_accuracy_vs_all": 0.0,
            "delta_macro_f1_vs_all": 0.0,
            "macro_f1_relative_drop_pct": 0.0,
        }
    ]

    for group_name, group_cols in feature_groups.items():
        present_cols = [c for c in group_cols if c in all_feature_cols]
        if not present_cols:
            continue

        current_cols = [c for c in all_feature_cols if c not in present_cols]
        if not current_cols:
            continue

        print(f"[Ablation] Remove group: {group_name} ({present_cols})")
        acc, f1 = evaluate_config(train_df, test_df, current_cols, label_encoder)

        delta_acc = acc - base_acc
        delta_f1 = f1 - base_f1
        rel_drop_pct = 0.0 if base_f1 == 0 else float(100.0 * (base_f1 - f1) / abs(base_f1))

        rows.append(
            {
                "setting": f"no_{group_name}",
                "removed_group": group_name,
                "removed_columns": ", ".join(present_cols),
                "n_features": len(current_cols),
                "group_size_removed": len(present_cols),
                "accuracy": acc,
                "macro_f1": f1,
                "delta_accuracy_vs_all": delta_acc,
                "delta_macro_f1_vs_all": delta_f1,
                "macro_f1_relative_drop_pct": rel_drop_pct,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    losses_df = summary_df[summary_df["removed_group"] != "none"].copy()
    losses_df["importance_loss"] = base_f1 - losses_df["macro_f1"]
    losses_df = losses_df.sort_values("importance_loss", ascending=False)

    ranking_path = OUTPUT_DIR / "ablation_group_importance_ranked.csv"
    losses_df.to_csv(ranking_path, index=False)

    delta_plot_path = OUTPUT_DIR / "ablation_delta_macro_f1.png"
    save_delta_plot(summary_df, delta_plot_path)

    best_setting_row = summary_df.iloc[0]
    worst_removal_row = losses_df.iloc[0] if not losses_df.empty else None
    top_helpful_rows = losses_df.sort_values("delta_macro_f1_vs_all", ascending=False).head(3)
    top_harmful_rows = losses_df.sort_values("delta_macro_f1_vs_all", ascending=True).head(3)

    report = {
        "dataset": {
            "n_samples": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_subjects_train": int(len(train_subjects)),
            "n_subjects_test": int(len(test_subjects)),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        },
        "feature_space": {
            "n_total_features": int(len(all_feature_cols)),
            "n_groups": int(len(feature_groups)),
            "group_sizes": {group_name: int(len(group_cols)) for group_name, group_cols in feature_groups.items()},
            "coverage_csv": str(coverage_path),
        },
        "baseline_all_features": {
            "accuracy": base_acc,
            "macro_f1": base_f1,
            "n_features": len(all_feature_cols),
        },
        "best_setting": {
            "setting": str(best_setting_row["setting"]),
            "macro_f1": float(best_setting_row["macro_f1"]),
            "accuracy": float(best_setting_row["accuracy"]),
        },
        "top_harmful_removals": [
            {
                "removed_group": str(row["removed_group"]),
                "delta_macro_f1_vs_all": float(row["delta_macro_f1_vs_all"]),
                "delta_accuracy_vs_all": float(row["delta_accuracy_vs_all"]),
            }
            for _, row in top_harmful_rows.iterrows()
        ],
        "top_helpful_removals": [
            {
                "removed_group": str(row["removed_group"]),
                "delta_macro_f1_vs_all": float(row["delta_macro_f1_vs_all"]),
                "delta_accuracy_vs_all": float(row["delta_accuracy_vs_all"]),
            }
            for _, row in top_helpful_rows.iterrows()
        ],
        "most_important_removed_group": None
        if worst_removal_row is None
        else {
            "removed_group": str(worst_removal_row["removed_group"]),
            "removed_columns": str(worst_removal_row["removed_columns"]),
            "delta_macro_f1_vs_all": float(worst_removal_row["delta_macro_f1_vs_all"]),
            "importance_loss": float(worst_removal_row["importance_loss"]),
        },
        "outputs": {
            "summary_csv": str(summary_path),
            "ranking_csv": str(ranking_path),
            "report_json": str(OUTPUT_DIR / "ablation_report.json"),
            "delta_plot_png": str(delta_plot_path),
        },
    }

    report_path = OUTPUT_DIR / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Ablation Study (XGBoost) ---")
    print(summary_df.to_string(index=False))
    print(f"\nFeature coverage CSV: {coverage_path}")
    print(f"\nSummary CSV: {summary_path}")
    print(f"Ranking CSV: {ranking_path}")
    print(f"Delta plot: {delta_plot_path}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
