from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
OUTPUT_ROOT = BASE_DIR / "output" / "classification_advanced" / "organized"
CV_DIR = OUTPUT_ROOT / "02_cv_subjectwise"
CACHE_DIR = OUTPUT_ROOT / "05_cache_split"
FOLDS_CACHE_PATH = CACHE_DIR / "subject_folds_5.json"

TARGET_COL = "lettera"
SUBJECT_COL = "soggetto"
N_SPLITS = 5


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


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        return result.returncode == 0 and "GPU" in (result.stdout or "")
    except Exception:
        return False


def resolve_gpu_preference() -> bool:
    mode = os.getenv("XGB_USE_GPU", "cpu").strip().lower()
    if mode in {"1", "true", "yes", "gpu", "cuda"}:
        return True
    if mode in {"0", "false", "no", "cpu"}:
        return False
    return _has_nvidia_gpu()


def build_xgb_pipeline(use_gpu: bool) -> Pipeline:
    xgb_kwargs = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 600,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": 1,
    }
    if use_gpu:
        xgb_kwargs["device"] = "cuda"

    tree_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return Pipeline(
        steps=[
            ("prep", tree_pipe),
            ("clf", XGBClassifier(**xgb_kwargs)),
        ]
    )


def build_models(use_gpu: bool) -> dict[str, Pipeline]:
    scaled_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    tree_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    svm_rbf = Pipeline(
        steps=[
            ("prep", scaled_pipe),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
        ]
    )

    extra_trees = Pipeline(
        steps=[
            ("prep", tree_pipe),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb = build_xgb_pipeline(use_gpu=use_gpu)

    return {
        "svm_rbf": svm_rbf,
        "extra_trees": extra_trees,
        "xgboost": xgb,
    }


def get_or_create_subject_folds(
    subjects: np.ndarray,
    n_splits: int,
    cache_path: Path,
    refresh_cache: bool = False,
) -> list[dict[str, list[str]]]:
    subject_set = set(map(str, subjects.tolist()))

    if cache_path.exists() and not refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        folds = payload.get("folds", [])
        valid = True
        for fold in folds:
            tr = set(map(str, fold.get("train_subjects", [])))
            te = set(map(str, fold.get("test_subjects", [])))
            if not tr or not te:
                valid = False
                break
            if (tr | te) != subject_set:
                valid = False
                break
            if tr & te:
                valid = False
                break
        if valid and len(folds) == n_splits:
            print(f"[cv] loaded cached subject folds from: {cache_path}")
            return folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in kf.split(subjects):
        train_subjects = sorted(map(str, subjects[train_idx].tolist()))
        test_subjects = sorted(map(str, subjects[test_idx].tolist()))
        folds.append({"train_subjects": train_subjects, "test_subjects": test_subjects})

    cache_path.write_text(
        json.dumps({"n_splits": n_splits, "folds": folds}, indent=2),
        encoding="utf-8",
    )
    print(f"[cv] saved subject folds cache to: {cache_path}")
    return folds


def aggregate_top_confusions(confusions: np.ndarray, labels: list[str], top_k: int = 15) -> pd.DataFrame:
    rows = []
    for i, true_lbl in enumerate(labels):
        row_sum = np.sum(confusions[i, :])
        for j, pred_lbl in enumerate(labels):
            if i == j:
                continue
            count = int(confusions[i, j])
            if count <= 0:
                continue
            rows.append(
                {
                    "true": true_lbl,
                    "pred": pred_lbl,
                    "count": count,
                    "row_error_rate": float(count / max(1, row_sum)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["true", "pred", "count", "row_error_rate"])

    return (
        pd.DataFrame(rows)
        .sort_values(["count", "row_error_rate"], ascending=[False, False])
        .head(top_k)
        .reset_index(drop=True)
    )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV}. Esegui prima analyze_letter.py")

    CV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = add_engineered_features(df)

    required = {TARGET_COL, SUBJECT_COL}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    feature_cols = get_numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("Nessuna feature numerica trovata")

    labels = sorted(df[TARGET_COL].astype(str).unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    subjects = np.array(sorted(df[SUBJECT_COL].astype(str).unique()))
    if len(subjects) < N_SPLITS:
        raise ValueError(f"Soggetti insufficienti per {N_SPLITS} fold: trovati {len(subjects)}")

    use_gpu = resolve_gpu_preference()
    refresh_folds = os.getenv("SPLIT_REFRESH", "0").strip().lower() in {"1", "true", "yes"}
    print(f"[xgboost] gpu_requested={use_gpu} (default=cpu, override with XGB_USE_GPU=cpu|gpu|auto)")
    models = build_models(use_gpu=use_gpu)

    folds = get_or_create_subject_folds(
        subjects,
        n_splits=N_SPLITS,
        cache_path=FOLDS_CACHE_PATH,
        refresh_cache=refresh_folds,
    )

    fold_rows = []
    agg_confusions = {name: np.zeros((len(labels), len(labels)), dtype=np.int64) for name in models.keys()}

    for fold_idx, fold in enumerate(folds, 1):
        train_subjects = set(map(str, fold["train_subjects"]))
        test_subjects = set(map(str, fold["test_subjects"]))

        train_df = df[df[SUBJECT_COL].astype(str).isin(train_subjects)].copy()
        test_df = df[df[SUBJECT_COL].astype(str).isin(test_subjects)].copy()

        x_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(str)
        x_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].astype(str)

        y_train_enc = label_encoder.transform(y_train)

        print(f"\n[Fold {fold_idx}/{N_SPLITS}] train_subjects={len(train_subjects)} test_subjects={len(test_subjects)}")

        for model_name, model in models.items():
            fit_target = y_train_enc if model_name == "xgboost" else y_train
            try:
                model.fit(x_train, fit_target)
            except Exception as exc:
                if model_name == "xgboost" and use_gpu:
                    print(f"  - [xgboost] GPU fallback CPU su fold {fold_idx}. Dettaglio: {exc}")
                    model = build_xgb_pipeline(use_gpu=False)
                    models[model_name] = model
                    model.fit(x_train, fit_target)
                else:
                    raise
            y_pred_raw = model.predict(x_test)
            if model_name == "xgboost":
                y_pred = label_encoder.inverse_transform(np.asarray(y_pred_raw, dtype=int))
            else:
                y_pred = np.asarray(y_pred_raw, dtype=str)

            acc = float(accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

            cm = confusion_matrix(y_test, y_pred, labels=labels)
            agg_confusions[model_name] += cm

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                }
            )

            print(f"  - {model_name}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )

    best_model = str(summary_df.iloc[0]["model"])
    top_confusions_df = aggregate_top_confusions(agg_confusions[best_model], labels, top_k=20)

    fold_path = CV_DIR / "cv_subjectwise_results.csv"
    summary_path = CV_DIR / "cv_subjectwise_summary.csv"
    confusions_path = CV_DIR / "cv_top_confusions.csv"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top_confusions_df.to_csv(confusions_path, index=False)

    report = {
        "n_splits": N_SPLITS,
        "n_subjects": int(len(subjects)),
        "n_samples": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "best_model": best_model,
        "cv_summary": summary_df.to_dict(orient="records"),
        "top_confusions_best_model": top_confusions_df.to_dict(orient="records"),
        "files": {
            "fold_results": str(fold_path),
            "summary": str(summary_path),
            "top_confusions": str(confusions_path),
        },
    }

    report_path = CV_DIR / "cv_subjectwise_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Subject-wise CV summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nBest model: {best_model}")
    print("\nTop confusions (best model):")
    if top_confusions_df.empty:
        print("(none)")
    else:
        print(top_confusions_df.to_string(index=False))

    print(f"\nSaved: {fold_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {confusions_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
