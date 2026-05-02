from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters, io, measure, morphology
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from extract_characters_cvl import ExtractionConfig, detect_boxes, preprocess
from src.morphology_logic import (
    average_stroke_width,
    border_contact_features,
    count_holes,
    contour_depth_features,
    diagonal_profile_features,
    euler_number,
    endpoint_distribution_features,
    endpoint_position_features,
    find_endpoints,
    find_junctions,
    global_shape_features,
    hole_area_ratio,
    hole_position_features,
    hu_moments_features,
    midline_run_features,
    projection_profile_features,
    pruning,
    scanline_structure_features,
    skeleton_branch_stats,
    skeleton_length,
    spatial_balance_features,
    transition_features,
    zoning_density_features,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "manoscritti"
DEFAULT_INPUT_IMAGE = DEFAULT_INPUT_DIR / "0002-1.tif"
FEATURES_CSV = BASE_DIR / "output" / "analysis" / "features_samples.csv"
PIPELINE_OUTPUT_DIR = BASE_DIR / "output" / "pipeline_demo"
MODEL_DIR = BASE_DIR / "output" / "classification_advanced" / "organized" / "06_model_artifacts"
MODEL_BUNDLE_PATH = MODEL_DIR / "xgboost_bundle.joblib"
DEMO_SUBJECT_DIRNAME = "soggettoNew"
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Bundle:
    model: XGBClassifier
    imputer: SimpleImputer
    label_encoder: LabelEncoder
    feature_cols: list[str]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    out["complexity_index"] = out["punte"] + 2.0 * out["incroci"] + out["buchi"]
    out["punte_x_buchi"] = out["punte"] * out["buchi"]
    out["ink_compactness"] = out["densita_pixel"] / (out["aspect_ratio"] + eps)
    out["components_per_endpoint"] = out["componenti"] / (out["punte"] + 1.0)
    return out


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"lettera", "soggetto", "file"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def load_image_any(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is not None:
        return bgr

    img = io.imread(str(image_path))
    if img.ndim == 2:
        return cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resolve_input_image(input_path: Path | None) -> Path:
    if input_path is None:
        if DEFAULT_INPUT_IMAGE.exists():
            return DEFAULT_INPUT_IMAGE
        if DEFAULT_INPUT_DIR.exists():
            candidates = sorted(
                [p for p in DEFAULT_INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
            )
            if candidates:
                return candidates[0]
        raise FileNotFoundError("Nessuna immagine trovata in data/manoscritti.")

    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        candidates = sorted(
            [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        )
        if not candidates:
            raise FileNotFoundError(f"Nessuna immagine supportata trovata in: {input_path}")
        return candidates[0]

    raise FileNotFoundError(f"Input non trovato: {input_path}")


def binarize_gray(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    try:
        thresh = filters.threshold_otsu(gray)
    except ValueError:
        thresh = float(np.mean(gray))
    return (gray < thresh).astype(bool)


def detect_and_crop_document(bgr: np.ndarray, cfg: ExtractionConfig):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = preprocess(gray, cfg)
    boxes = detect_boxes(binary, cfg)

    h_img, w_img = gray.shape[:2]
    crops = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        p = max(0, int(cfg.padding))
        x0 = max(0, x - p)
        y0 = max(0, y - p)
        x1 = min(w_img, x + w + p)
        y1 = min(h_img, y + h + p)
        crop_gray = gray[y0:y1, x0:x1]
        if crop_gray.size == 0:
            continue
        crops.append(
            {
                "index": idx,
                "bbox": (x, y, w, h),
                "gray": crop_gray,
                "binary": binarize_gray(crop_gray),
            }
        )

    boxed = bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return gray, binary, boxes, boxed, crops


def _bbox_aspect_ratio(binary: np.ndarray) -> float:
    coords = measure.regionprops(measure.label(binary.astype(int)))
    if not coords:
        return 0.0
    largest = max(coords, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    h = max(1, maxr - minr)
    w = max(1, maxc - minc)
    return float(w / h)


def _pixel_density(binary: np.ndarray) -> float:
    return float(binary.sum() / max(1, binary.size))


def extract_features_from_binary(binary: np.ndarray) -> dict[str, float]:
    skel = morphology.skeletonize(binary)
    clean_skel = pruning(skel, iterations=3)

    skeleton_len_raw = skeleton_length(skel)
    skeleton_len_clean = skeleton_length(clean_skel)
    removed_ratio = (skeleton_len_raw - skeleton_len_clean) / max(1, skeleton_len_raw)

    endpoint_mask = find_endpoints(clean_skel)
    endpoints = int(endpoint_mask.sum())
    junctions = int(find_junctions(clean_skel).sum())
    holes = int(count_holes(binary))
    components = int(measure.label(binary).max())
    aspect_ratio = _bbox_aspect_ratio(binary)
    density = _pixel_density(binary)
    euler = euler_number(binary)
    hole_ratio = hole_area_ratio(binary)
    stroke_w = average_stroke_width(binary, clean_skel)
    shape_feats = global_shape_features(binary)
    hu_feats = hu_moments_features(binary)
    branch_feats = skeleton_branch_stats(clean_skel)
    zoning_feats = zoning_density_features(binary, grid_rows=3, grid_cols=3)
    proj_feats = projection_profile_features(binary)
    balance_feats = spatial_balance_features(binary)
    trans_feats = transition_features(binary)
    diag_feats = diagonal_profile_features(binary)
    endpoint_pos_feats = endpoint_position_features(endpoint_mask)
    scanline_feats = scanline_structure_features(binary)
    border_feats = border_contact_features(binary)
    hole_pos_feats = hole_position_features(binary)
    contour_feats = contour_depth_features(binary)
    endpoint_dist_feats = endpoint_distribution_features(endpoint_mask)
    midline_feats = midline_run_features(binary)

    endpoint_norm = float(endpoints) / max(1, skeleton_len_clean)
    junction_norm = float(junctions) / max(1, skeleton_len_clean)
    skeleton_density = float(skeleton_len_clean) / max(1.0, float(binary.sum()))

    sample_row: dict[str, float] = {
        "punte": float(endpoints),
        "incroci": float(junctions),
        "buchi": float(holes),
        "componenti": float(components),
        "aspect_ratio": float(round(aspect_ratio, 6)),
        "densita_pixel": float(round(density, 6)),
        "euler_number": float(euler),
        "hole_area_ratio": float(round(hole_ratio, 6)),
        "avg_stroke_width": float(round(stroke_w, 6)),
        "skeleton_length": float(skeleton_len_clean),
        "endpoint_norm": float(round(endpoint_norm, 6)),
        "junction_norm": float(round(junction_norm, 6)),
        "skeleton_density": float(round(skeleton_density, 6)),
        "pruning_removed_ratio": float(round(removed_ratio, 6)),
        "solidity": float(round(shape_feats["solidity"], 6)),
        "extent": float(round(shape_feats["extent"], 6)),
        "eccentricity": float(round(shape_feats["eccentricity"], 6)),
        "orientation": float(round(shape_feats["orientation"], 6)),
        "major_axis_length": float(round(shape_feats["major_axis_length"], 6)),
        "minor_axis_length": float(round(shape_feats["minor_axis_length"], 6)),
        "axis_ratio": float(round(shape_feats["axis_ratio"], 6)),
        "n_branches": float(branch_feats["n_branches"]),
        "branch_length_mean": float(round(branch_feats["branch_length_mean"], 6)),
        "branch_length_max": float(round(branch_feats["branch_length_max"], 6)),
        "branch_length_std": float(round(branch_feats["branch_length_std"], 6)),
        "hu_1": float(round(hu_feats["hu_1"], 6)),
        "hu_2": float(round(hu_feats["hu_2"], 6)),
        "hu_3": float(round(hu_feats["hu_3"], 6)),
        "hu_4": float(round(hu_feats["hu_4"], 6)),
        "hu_5": float(round(hu_feats["hu_5"], 6)),
        "hu_6": float(round(hu_feats["hu_6"], 6)),
        "hu_7": float(round(hu_feats["hu_7"], 6)),
    }

    for block in (
        zoning_feats,
        proj_feats,
        balance_feats,
        trans_feats,
        diag_feats,
        endpoint_pos_feats,
        scanline_feats,
        border_feats,
        hole_pos_feats,
        contour_feats,
        endpoint_dist_feats,
        midline_feats,
    ):
        for key, value in block.items():
            sample_row[key] = float(round(float(value), 6))

    sample_row["complexity_index"] = float(sample_row["punte"] + 2.0 * sample_row["incroci"] + sample_row["buchi"])
    sample_row["punte_x_buchi"] = float(sample_row["punte"] * sample_row["buchi"])
    sample_row["ink_compactness"] = float(sample_row["densita_pixel"] / (sample_row["aspect_ratio"] + 1e-6))
    sample_row["components_per_endpoint"] = float(sample_row["componenti"] / (sample_row["punte"] + 1.0))

    return sample_row, skel, clean_skel, endpoint_mask


def load_bundle(bundle_path: Path) -> Bundle:
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle modello non trovato: {bundle_path}. Esegui prima train_xgboost_bundle.py"
        )

    payload = joblib.load(bundle_path)
    print(f"[model] bundle caricato da: {bundle_path}")
    return Bundle(
        model=payload["model"],
        imputer=payload["imputer"],
        label_encoder=payload["label_encoder"],
        feature_cols=list(payload["feature_cols"]),
    )


def classify_features(bundle: Bundle, feature_row: dict[str, float]) -> dict[str, object]:
    frame = pd.DataFrame([feature_row])
    for col in bundle.feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0

    x = frame[bundle.feature_cols]
    x_imp = bundle.imputer.transform(x)
    proba = bundle.model.predict_proba(x_imp)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(bundle.label_encoder.inverse_transform([pred_idx])[0])

    top_idx = np.argsort(proba)[::-1][:5]
    top_labels = bundle.label_encoder.inverse_transform(top_idx.astype(int))

    return {
        "pred_label": pred_label,
        "pred_prob": float(proba[pred_idx]),
        "top_labels": [str(lbl) for lbl in top_labels],
        "top_probs": [float(proba[i]) for i in top_idx],
        "probabilities": proba,
    }


def save_page_processing_figure(bgr: np.ndarray, gray: np.ndarray, binary: np.ndarray, boxed: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = np.asarray(axes).ravel()
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Documento originale")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Binarizzazione + pulizia")
    axes[3].imshow(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Bounding box caratteri")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_character_grid(crops: list[dict[str, object]], bundle: Bundle, base_out_dir: Path, items_per_page: int = 12):
    """Generate all multi-page character grids.
    
    Args:
        crops: List of character crop items
        bundle: XGBoost bundle for classification
        base_out_dir: Directory where page files will be saved
        items_per_page: Characters per page in grid (default 12 = 4x3)
    """
    if not crops:
        return
    
    base_out_dir.mkdir(parents=True, exist_ok=True)
    total_items = len(crops)
    ncols = 4
    nrows = int(np.ceil(items_per_page / ncols))
    
    # Calculate total pages needed
    total_pages = int(np.ceil(total_items / items_per_page))
    
    page_count = 0
    for page_idx in range(total_pages):
        start_idx = page_idx * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        items = crops[start_idx:end_idx]
        n = len(items)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        
        for ax in axes:
            ax.axis("off")
        
        for ax, item in zip(axes, items):
            gray = item["gray"]
            binary = item["binary"]
            features, _, _, _ = extract_features_from_binary(binary)
            pred = classify_features(bundle, features)
            ax.imshow(gray, cmap="gray")
            ax.set_title(f"#{item['index']} -> {pred['pred_label']} ({pred['pred_prob']:.2f})", fontsize=10)
            ax.axis("off")
        
        plt.tight_layout()
        page_num = page_idx + 1
        out_path = base_out_dir / f"02_extracted_characters_page_{page_num:03d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        page_count += 1
    
    print(f"[grid] generated {page_count} page(s) with {total_items} character(s)")


def save_extracted_crops(crops: list[dict[str, object]], bundle: Bundle, subject_new_dir: Path):
    subject_new_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in crops:
        features, _, _, _ = extract_features_from_binary(item["binary"])
        pred = classify_features(bundle, features)
        label_dir = subject_new_dir / pred["pred_label"]
        label_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"char_{int(item['index']):03d}_{pred['pred_label']}.png"
        out_path = label_dir / out_name
        cv2.imwrite(str(out_path), item["gray"])

        rows.append(
            {
                "char_index": item["index"],
                "predicted_label": pred["pred_label"],
                "predicted_prob": pred["pred_prob"],
                "saved_path": str(out_path),
                "top_labels": ", ".join(pred["top_labels"]),
                "top_probs": ", ".join([f"{p:.4f}" for p in pred["top_probs"]]),
            }
        )

    return pd.DataFrame(rows)


def save_selected_detail(item: dict[str, object], bundle: Bundle, out_path: Path):
    gray = item["gray"]
    binary = item["binary"]
    features, skel, clean_skel, endpoint_mask = extract_features_from_binary(binary)
    pred = classify_features(bundle, features)

    feature_lines = [
        f"Predizione: {pred['pred_label']} ({pred['pred_prob']:.3f})",
        f"Top-5: {', '.join([f'{lbl} {prob:.2f}' for lbl, prob in zip(pred['top_labels'], pred['top_probs'])])}",
        "",
        f"punte = {features['punte']:.0f}",
        f"incroci = {features['incroci']:.0f}",
        f"buchi = {features['buchi']:.0f}",
        f"aspect_ratio = {features['aspect_ratio']:.3f}",
        f"densita_pixel = {features['densita_pixel']:.3f}",
        f"avg_stroke_width = {features['avg_stroke_width']:.3f}",
        f"skeleton_length = {features['skeleton_length']:.0f}",
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Crop estratto")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(binary, cmap="gray")
    ax2.set_title("Crop binarizzato")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(clean_skel, cmap="gray")
    ax3.set_title("Scheletro pulito")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.text(0.02, 0.98, "\n".join(feature_lines), va="top", ha="left", fontsize=12, family="monospace")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_selected_details_batch(crops: list[dict[str, object]], bundle: Bundle, out_dir: Path):
    """Generate detail figures for all extracted characters, grouped by predicted letter."""
    out_dir.mkdir(parents=True, exist_ok=True)

    total_items = len(crops)
    for idx, item in enumerate(crops, start=1):
        features, _, _, _ = extract_features_from_binary(item["binary"])
        pred = classify_features(bundle, features)
        label_dir = out_dir / pred["pred_label"]
        label_dir.mkdir(parents=True, exist_ok=True)

        out_path = label_dir / f"04_character_detail_{idx:04d}_char_{int(item['index']):03d}_{pred['pred_label']}.png"
        save_selected_detail(item, bundle, out_path)

    print(f"[details] generated {total_items} character detail(s)")


def save_predictions_csv(crops: list[dict[str, object]], bundle: Bundle, out_path: Path):
    rows = []
    for item in crops:
        features, _, _, _ = extract_features_from_binary(item["binary"])
        pred = classify_features(bundle, features)
        rows.append(
            {
                "char_index": item["index"],
                "predicted_label": pred["pred_label"],
                "predicted_prob": pred["pred_prob"],
                "top_labels": ", ".join(pred["top_labels"]),
                "top_probs": ", ".join([f"{p:.4f}" for p in pred["top_probs"]]),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def create_demo_report(
    image_path: Path,
    output_dir: Path,
    bundle: Bundle,
    gray: np.ndarray,
    binary: np.ndarray,
    boxed: np.ndarray,
    crops: list[dict[str, object]],
    selected_index: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    page_fig = output_dir / "01_page_processing.png"
    grid_pages_dir = output_dir / "02_extracted_characters_pages"
    detail_fig = output_dir / "03_selected_character_detail.png"
    details_batch_dir = output_dir / "04_character_details_batch"
    predictions_csv = output_dir / "predictions.csv"
    report_md = output_dir / "report.md"
    subject_new_dir = output_dir / DEMO_SUBJECT_DIRNAME

    save_page_processing_figure(load_image_any(image_path), gray, binary, boxed, page_fig)
    save_character_grid(crops, bundle, grid_pages_dir, items_per_page=12)
    preds_df = save_extracted_crops(crops, bundle, subject_new_dir)
    preds_df.to_csv(predictions_csv, index=False)

    selected_item = None
    if crops:
        for item in crops:
            if int(item["index"]) == selected_index:
                selected_item = item
                break
        if selected_item is None:
            selected_item = crops[0]
        save_selected_detail(selected_item, bundle, detail_fig)
        # Generate details for ALL characters
        save_selected_details_batch(crops, bundle, details_batch_dir)

    # Count generated pages for report
    page_files = sorted(grid_pages_dir.glob("02_extracted_characters_page_*.png")) if grid_pages_dir.exists() else []
    pages_info = f"{len(page_files)} pages" if page_files else "no pages generated"
    
    # Count generated details for report
    detail_files = sorted(details_batch_dir.rglob("04_character_detail_*.png")) if details_batch_dir.exists() else []
    details_info = f"{len(detail_files)} examples in letter folders" if detail_files else "no examples generated"

    report_md.write_text(
        "\n".join(
            [
                "# Demo visivo pipeline",
                "",
                f"- Input: `{image_path}`",
                f"- Box rilevati: {len(crops)}",
                f"- Modello: XGBoost bundle ({bundle.label_encoder.classes_.size} classi)",
                "",
                "## Immagini",
                f"- [Documento / preprocessing]({page_fig.name})",
                f"- [Caratteri estratti - {pages_info}](02_extracted_characters_pages/)",
                f"- [Dettaglio carattere selezionato]({detail_fig.name})",
                f"- [Dettagli di tutti i caratteri - {details_info}](04_character_details_batch/)",
                "",
                "## Dati",
                f"- [Predizioni CSV]({predictions_csv.name})",
                f"- Cartella caratteri estratti: `{subject_new_dir.name}/<lettera>/...`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[report] immagini salvate in: {output_dir}")
    print(f"[report] markdown: {report_md}")
    print(f"[report] predictions: {predictions_csv}")
    print(f"[report] character grid pages: {pages_info}")
    print(f"[report] character details batch: {details_info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo visiva pura: carica un bundle XGBoost già allenato, classifica i caratteri estratti e li salva in soggettoNew/<classe>.",
    )
    parser.add_argument("--input", type=str, default=None, help="Immagine o cartella da processare. Default: primo TIFF in data/manoscritti.")
    parser.add_argument("--output", type=str, default=str(PIPELINE_OUTPUT_DIR), help="Cartella di output per report e immagini.")
    parser.add_argument("--char-index", type=int, default=1, help="Indice del carattere da evidenziare nel dettaglio.")
    parser.add_argument("--save-debug-extraction", action="store_true", help="Salva anche le immagini di debug dell'estrazione.")
    parser.add_argument("--bundle-path", type=str, default=str(MODEL_BUNDLE_PATH), help="Path del bundle XGBoost già allenato.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = resolve_input_image(Path(args.input).expanduser().resolve() if args.input else None)
    output_dir = Path(args.output).expanduser().resolve() / input_path.stem

    cfg = ExtractionConfig()
    bundle = load_bundle(Path(args.bundle_path).expanduser().resolve())

    bgr = load_image_any(input_path)
    gray, binary, boxes, boxed, crops = detect_and_crop_document(bgr, cfg)

    if args.save_debug_extraction:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{input_path.stem}_binary.png"), binary.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{input_path.stem}_boxed.png"), boxed)

    create_demo_report(
        image_path=input_path,
        output_dir=output_dir,
        bundle=bundle,
        gray=gray,
        binary=binary,
        boxed=boxed,
        crops=crops,
        selected_index=args.char_index,
    )

    print("\n=== Demo completata ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Caratteri estratti: {len(crops)}")


if __name__ == "__main__":
    main()