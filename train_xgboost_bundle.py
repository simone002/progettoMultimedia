from __future__ import annotations

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
DEFAULT_BUNDLE_PATH = BASE_DIR / "output" / "classification_advanced" / "organized" / "06_model_artifacts" / "xgboost_bundle.joblib"
DEFAULT_METADATA_PATH = BASE_DIR / "output" / "classification_advanced" / "organized" / "06_model_artifacts" / "xgboost_bundle_metadata.json"

TARGET_COL = "lettera"
SUBJECT_COL = "soggetto"


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


def train_bundle(features_csv: Path, bundle_path: Path, metadata_path: Path):
    if not features_csv.exists():
        raise FileNotFoundError(f"File feature non trovato: {features_csv}")

    df = pd.read_csv(features_csv)
    df = add_engineered_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Il CSV deve contenere la colonna '{TARGET_COL}'.")
    if SUBJECT_COL not in df.columns:
        raise ValueError(f"Il CSV deve contenere la colonna '{SUBJECT_COL}'.")

    feature_cols = get_numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("Nessuna feature numerica trovata nel CSV.")

    train_df, test_df, train_subjects, test_subjects = split_by_subject(df, test_ratio=0.25, random_state=42)

    labels = sorted(df[TARGET_COL].astype(str).unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    x_train = train_df[feature_cols]
    y_train = label_encoder.transform(train_df[TARGET_COL].astype(str))
    x_test = test_df[feature_cols]
    y_test = label_encoder.transform(test_df[TARGET_COL].astype(str))

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_test_imp = imputer.transform(x_test)

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=250,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(x_train_imp, y_train)

    y_pred = model.predict(x_test_imp)
    final_accuracy = float(accuracy_score(y_test, y_pred))

    bundle = {
        "model": model,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "source_features_csv": str(features_csv),
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "n_features": int(len(feature_cols)),
        "classes": labels,
        "accuracy": final_accuracy,
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    metadata = {
        "source_features_csv": str(features_csv),
        "bundle_path": str(bundle_path),
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "n_features": int(len(feature_cols)),
        "classes": labels,
        "feature_columns": feature_cols,
        "accuracy": final_accuracy,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[train] bundle salvato in: {bundle_path}")
    print(f"[train] metadata salvato in: {metadata_path}")
    print(f"[train] accuracy finale: {final_accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Allena XGBoost su holdout per soggetto, salva il bundle e stampa l'accuracy finale.")
    parser.add_argument("--features-csv", type=str, default=str(DEFAULT_FEATURES_CSV), help="CSV delle feature estratte.")
    parser.add_argument("--bundle-path", type=str, default=str(DEFAULT_BUNDLE_PATH), help="Path del bundle da salvare.")
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH), help="Path del JSON di metadata da salvare.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_bundle(
        features_csv=Path(args.features_csv).expanduser().resolve(),
        bundle_path=Path(args.bundle_path).expanduser().resolve(),
        metadata_path=Path(args.metadata_path).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
