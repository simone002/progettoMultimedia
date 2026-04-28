from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier

# questo file contiene la logica per eseguire un'analisi di classificazione avanzata
#  sui dati estratti da analyze_letter.py, utilizzando modelli non-CNN come SVM, Extra Trees
#  e XGBoost, con un focus particolare sulla generalizzazione subject-wise.
# I risultati includono metriche di performance, confusion matrices,
#  feature importance e analisi dettagliate per coppie di lettere specifiche. 
# I risultati vengono salvati in CSV, JSON e immagini per una facile consultazione e confronto.

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
OUTPUT_ROOT = BASE_DIR / "output" / "classification_advanced" / "organized"
HOLDOUT_DIR = OUTPUT_ROOT / "01_holdout"
DIAG_DIR = OUTPUT_ROOT / "04_diagnostics"
CM_RAW_DIR = DIAG_DIR / "confusion_matrices" / "raw"
CM_NORM_DIR = DIAG_DIR / "confusion_matrices" / "normalized"
FI_DIR = DIAG_DIR / "feature_importance"
F1_DIR = DIAG_DIR / "f1_by_letter"
CACHE_DIR = OUTPUT_ROOT / "05_cache_split"

SPLIT_CACHE_PATH = CACHE_DIR / "subject_holdout_split.json"

BASE_FEATURE_COLS = [
    "punte",
    "incroci",
    "buchi",
    "componenti",
    "aspect_ratio",
    "densita_pixel",
]
TARGET_COL = "lettera"
SUBJECT_COL = "soggetto"
TARGET_CONFUSION_PAIRS = [("u", "n"), ("f", "t"), ("e", "a")]


def split_by_subject(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    random_state: int = 42,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
):
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        train_subjects = set(map(str, payload.get("train_subjects", [])))
        test_subjects = set(map(str, payload.get("test_subjects", [])))
        if train_subjects and test_subjects:
            train_df = df[df[SUBJECT_COL].astype(str).isin(train_subjects)].copy()
            test_df = df[df[SUBJECT_COL].astype(str).isin(test_subjects)].copy()
            print(f"[split] loaded cached split from: {cache_path}")
            return train_df, test_df, sorted(train_subjects), sorted(test_subjects)

    subjects = np.array(sorted(df[SUBJECT_COL].astype(str).unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = set(subjects[:n_test])
    train_subjects = set(subjects[n_test:])

    if cache_path is not None:
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


def build_models(full_feature_cols: list[str], use_gpu: bool):

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

    models = {
        "svm_rbf": svm_rbf,
        "extra_trees": extra_trees,
        "xgboost": xgb,
    }

    return models, full_feature_cols


def save_confusion_matrix(y_true, y_pred, class_labels, title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_normalized(y_true, y_pred, class_labels, title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Purples", colorbar=False, xticks_rotation=45, values_format=".2f")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_f1_by_class_plot(report: dict, class_labels: list[str], title: str, out_path: Path):
    f1_scores = [float(report.get(label, {}).get("f1-score", 0.0)) for label in class_labels]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(class_labels, f1_scores, color="teal", alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Lettera")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_xgb_feature_importance_plot(xgb_pipeline: Pipeline, feature_cols: list[str], out_path: Path, top_k: int = 20):
    clf = xgb_pipeline.named_steps["clf"]
    importances = np.asarray(clf.feature_importances_, dtype=float)
    if importances.size == 0:
        return

    names = np.asarray(feature_cols)
    n = min(len(names), importances.size)
    names = names[:n]
    importances = importances[:n]

    order = np.argsort(importances)[::-1][:top_k]
    top_names = names[order][::-1]
    top_vals = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_names, top_vals, color="darkorange", alpha=0.9)
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost Feature Importance (Top {len(top_names)})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def compute_target_pair_confusions(y_true, y_pred, target_pairs):
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    out = {}

    for a, b in target_pairs:
        mask_a = y_true_arr == a
        mask_b = y_true_arr == b

        support_a = int(np.sum(mask_a))
        support_b = int(np.sum(mask_b))
        a_to_b = int(np.sum(mask_a & (y_pred_arr == b)))
        b_to_a = int(np.sum(mask_b & (y_pred_arr == a)))

        out[f"{a}_vs_{b}"] = {
            "support_true_a": support_a,
            "support_true_b": support_b,
            "a_to_b": a_to_b,
            "b_to_a": b_to_a,
            "a_to_b_rate": float(a_to_b / max(1, support_a)),
            "b_to_a_rate": float(b_to_a / max(1, support_b)),
            "bidirectional_total": int(a_to_b + b_to_a),
        }

    return out


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV}. Esegui prima analyze_letter.py")

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    CM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CM_NORM_DIR.mkdir(parents=True, exist_ok=True)
    FI_DIR.mkdir(parents=True, exist_ok=True)
    F1_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required_cols = [*BASE_FEATURE_COLS, TARGET_COL, SUBJECT_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel CSV: {missing_cols}")

    df = add_engineered_features(df)
    use_gpu = resolve_gpu_preference()
    refresh_split = os.getenv("SPLIT_REFRESH", "0").strip().lower() in {"1", "true", "yes"}
    print(f"[xgboost] gpu_requested={use_gpu} (default=cpu, override with XGB_USE_GPU=cpu|gpu|auto)")

    all_feature_cols = get_numeric_feature_columns(df)
    models, all_feature_cols = build_models(all_feature_cols, use_gpu=use_gpu)

    train_df, test_df, train_subjects, test_subjects = split_by_subject(
        df,
        test_ratio=0.25,
        random_state=42,
        cache_path=SPLIT_CACHE_PATH,
        refresh_cache=refresh_split,
    )

    x_train = train_df[all_feature_cols]
    y_train = train_df[TARGET_COL].astype(str)
    x_test = test_df[all_feature_cols]
    y_test = test_df[TARGET_COL].astype(str)

    class_labels = sorted(df[TARGET_COL].astype(str).unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    y_train_encoded = label_encoder.transform(y_train)

    results = {
        "dataset": {
            "n_samples": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_subjects_train": int(len(train_subjects)),
            "n_subjects_test": int(len(test_subjects)),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "feature_columns": all_feature_cols,
            "target": TARGET_COL,
            "classes": class_labels,
        },
        "models": {},
    }

    summary_rows = []
    best_model = None
    best_macro_f1 = -1.0
    best_report = None
    best_y_pred = None

    for i, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] Training {model_name}...")

        # XGBoost richiede classi numeriche
        fit_target = y_train_encoded if model_name == "xgboost" else y_train
        try:
            model.fit(x_train, fit_target)
        except Exception as exc:
            if model_name == "xgboost" and use_gpu:
                print(f"[xgboost] GPU non disponibile o non supportata, fallback CPU. Dettaglio: {exc}")
                model = build_xgb_pipeline(use_gpu=False)
                models[model_name] = model
                model.fit(x_train, fit_target)
            else:
                raise
        y_pred_raw = model.predict(x_test)

        if model_name == "xgboost":
            y_pred = label_encoder.inverse_transform(np.asarray(y_pred_raw, dtype=int))
        else:
            y_pred = y_pred_raw

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        )

        cm_path = CM_RAW_DIR / f"confusion_matrix_{model_name}.png"
        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            class_labels=class_labels,
            title=f"Confusion Matrix - {model_name}",
            out_path=cm_path,
        )

        cm_norm_path = CM_NORM_DIR / f"confusion_matrix_{model_name}_normalized.png"
        save_confusion_matrix_normalized(
            y_true=y_test,
            y_pred=y_pred,
            class_labels=class_labels,
            title=f"Confusion Matrix Normalized - {model_name}",
            out_path=cm_norm_path,
        )

        if model_name == "xgboost":
            fi_path = FI_DIR / "xgboost_feature_importance_top20.png"
            save_xgb_feature_importance_plot(model, all_feature_cols, fi_path, top_k=20)

        results["models"][model_name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": report,
            "confusion_matrix_image": str(cm_path),
            "confusion_matrix_normalized_image": str(cm_norm_path),
            "target_pair_confusions": compute_target_pair_confusions(
                y_true=y_test,
                y_pred=y_pred,
                target_pairs=TARGET_CONFUSION_PAIRS,
            ),
        }

        summary_rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = model_name
            best_report = report
            best_y_pred = y_pred

        print(f"[{model_name}] accuracy={acc:.4f} | macro_f1={macro_f1:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_csv = HOLDOUT_DIR / "advanced_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    results["best_model"] = {
        "name": best_model,
        "macro_f1": best_macro_f1,
    }

    if best_report is not None and best_y_pred is not None:
        f1_plot_path = F1_DIR / f"f1_by_letter_{best_model}.png"
        save_f1_by_class_plot(
            report=best_report,
            class_labels=class_labels,
            title=f"F1 per lettera - {best_model}",
            out_path=f1_plot_path,
        )
        results["best_model"]["f1_by_letter_image"] = str(f1_plot_path)

    metrics_json = HOLDOUT_DIR / "advanced_metrics.json"
    metrics_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n--- Risultati modelli avanzati non-CNN ---")
    print(summary_df.to_string(index=False))
    print(f"\nMiglior modello: {best_model} (macro_f1={best_macro_f1:.4f})")
    print(f"Metriche complete: {metrics_json}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
