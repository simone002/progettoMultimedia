from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
OUTPUT_DIR = BASE_DIR / "output" / "classification_advanced" / "organized" / "03_ablation"
TARGET_COL = "lettera"
SUBJECT_COL = "soggetto"


ABLATED_GROUPS = {
    "advanced_topology": ["endpoint_norm", "junction_norm", "euler_number"],
    "stroke_geometry": ["avg_stroke_width", "skeleton_length", "skeleton_density"],
    "hole_structure": ["hole_area_ratio"],
    "pruning_signal": ["pruning_removed_ratio"],
    "global_shape": [
        "solidity",
        "extent",
        "eccentricity",
        "orientation",
        "major_axis_length",
        "minor_axis_length",
        "axis_ratio",
    ],
    "skeleton_branches": [
        "n_branches",
        "branch_length_mean",
        "branch_length_max",
        "branch_length_std",
    ],
    "hu_moments": ["hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7"],
}


def split_by_subject(df: pd.DataFrame, test_ratio: float = 0.25, random_state: int = 42):
    subjects = np.array(sorted(df[SUBJECT_COL].astype(str).unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

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

    train_df, test_df, train_subjects, test_subjects = split_by_subject(df, test_ratio=0.25, random_state=42)

    classes = sorted(df[TARGET_COL].astype(str).unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    print("[Ablation] Baseline (all features)...")
    base_acc, base_f1 = evaluate_config(train_df, test_df, all_feature_cols, label_encoder)

    rows = [
        {
            "setting": "all_features",
            "removed_group": "none",
            "removed_columns": "",
            "n_features": len(all_feature_cols),
            "accuracy": base_acc,
            "macro_f1": base_f1,
            "delta_macro_f1_vs_all": 0.0,
        }
    ]

    for group_name, group_cols in ABLATED_GROUPS.items():
        present_cols = [c for c in group_cols if c in all_feature_cols]
        if not present_cols:
            continue

        current_cols = [c for c in all_feature_cols if c not in present_cols]
        print(f"[Ablation] Remove group: {group_name} ({present_cols})")
        acc, f1 = evaluate_config(train_df, test_df, current_cols, label_encoder)

        rows.append(
            {
                "setting": f"no_{group_name}",
                "removed_group": group_name,
                "removed_columns": ", ".join(present_cols),
                "n_features": len(current_cols),
                "accuracy": acc,
                "macro_f1": f1,
                "delta_macro_f1_vs_all": f1 - base_f1,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    losses_df = summary_df[summary_df["removed_group"] != "none"].copy()
    losses_df["importance_loss"] = -losses_df["delta_macro_f1_vs_all"]
    losses_df = losses_df.sort_values("importance_loss", ascending=False)

    best_setting_row = summary_df.iloc[0]
    worst_removal_row = losses_df.iloc[0] if not losses_df.empty else None

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
        "most_important_removed_group": None
        if worst_removal_row is None
        else {
            "removed_group": str(worst_removal_row["removed_group"]),
            "removed_columns": str(worst_removal_row["removed_columns"]),
            "delta_macro_f1_vs_all": float(worst_removal_row["delta_macro_f1_vs_all"]),
            "importance_loss": float(worst_removal_row["importance_loss"]),
        },
    }

    report_path = OUTPUT_DIR / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Ablation Study (XGBoost) ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
