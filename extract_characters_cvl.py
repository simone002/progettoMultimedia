from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class ExtractionConfig:
    adaptive_block_size: int = 15
    adaptive_c: int = 8
    denoise_h: int = 30
    erode_kernel: int = 1
    dilate_kernel: int = 2
    opening_kernel: int = 1
    closing_kernel: int = 1
    min_area: int = 20
    min_width: int = 4
    min_height: int = 6
    min_aspect_ratio: float = 0.03
    max_aspect_ratio: float = 12.0
    max_width_ratio: float = 0.25
    max_height_ratio: float = 0.25
    filter_outliers: bool = True
    max_area_factor: float = 35.0
    max_height_factor: float = 5.5
    padding: int = 1
    resize_to: int | None = None


def _ensure_odd(value: int, fallback: int = 15) -> int:
    if value < 3:
        value = fallback
    if value % 2 == 0:
        value += 1
    return value


def preprocess(gray: np.ndarray, cfg: ExtractionConfig) -> np.ndarray:
    block_size = _ensure_odd(cfg.adaptive_block_size)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        cfg.adaptive_c,
    )

    bw = cv2.fastNlMeansDenoising(bw, None, cfg.denoise_h, 7, 21)

    erode_k = max(1, int(cfg.erode_kernel))
    dilate_k = max(1, int(cfg.dilate_kernel))
    kernel_erode = np.ones((erode_k, erode_k), np.uint8)
    kernel_dilate = np.ones((dilate_k, dilate_k), np.uint8)

    bw = cv2.erode(bw, kernel_erode, iterations=1)
    bw = cv2.dilate(bw, kernel_dilate, iterations=1)

    if cfg.opening_kernel > 1:
        k = np.ones((cfg.opening_kernel, cfg.opening_kernel), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    if cfg.closing_kernel > 1:
        k = np.ones((cfg.closing_kernel, cfg.closing_kernel), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    return bw


def sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes

    heights = np.asarray([h for _, _, _, h in boxes], dtype=float)
    y_tol = max(8.0, float(np.median(heights) * 0.6))

    lines: list[dict[str, object]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        _, y, _, h = box
        cy = y + h / 2.0

        assigned = False
        for line in lines:
            line_cy = float(line["cy"])
            if abs(cy - line_cy) <= y_tol:
                line_boxes = line["boxes"]
                assert isinstance(line_boxes, list)
                line_boxes.append(box)
                centers = [b[1] + b[3] / 2.0 for b in line_boxes]
                line["cy"] = float(np.mean(centers))
                assigned = True
                break

        if not assigned:
            lines.append({"cy": cy, "boxes": [box]})

    lines = sorted(lines, key=lambda x: float(x["cy"]))

    ordered: list[tuple[int, int, int, int]] = []
    for line in lines:
        line_boxes = line["boxes"]
        assert isinstance(line_boxes, list)
        ordered.extend(sorted(line_boxes, key=lambda b: b[0]))

    return ordered


def detect_boxes(binary_image: np.ndarray, cfg: ExtractionConfig) -> list[tuple[int, int, int, int]]:
    h_img, w_img = binary_image.shape[:2]
    max_w = max(cfg.min_width, int(w_img * cfg.max_width_ratio))
    max_h = max(cfg.min_height, int(h_img * cfg.max_height_ratio))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    boxes: list[tuple[int, int, int, int]] = []
    areas: list[int] = []
    heights: list[int] = []

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area < cfg.min_area:
            continue
        if w < cfg.min_width or h < cfg.min_height:
            continue
        if w > max_w or h > max_h:
            continue

        ar = float(w) / max(1.0, float(h))
        if ar < cfg.min_aspect_ratio or ar > cfg.max_aspect_ratio:
            continue

        boxes.append((x, y, w, h))
        areas.append(area)
        heights.append(h)

    if cfg.filter_outliers and len(boxes) >= 8:
        med_area = float(np.median(np.asarray(areas, dtype=float)))
        med_h = float(np.median(np.asarray(heights, dtype=float)))
        area_limit = max(float(cfg.min_area), med_area * cfg.max_area_factor)
        h_limit = max(float(cfg.min_height), med_h * cfg.max_height_factor)

        filtered: list[tuple[int, int, int, int]] = []
        for x, y, w, h in boxes:
            if (w * h) > area_limit:
                continue
            if h > h_limit:
                continue
            filtered.append((x, y, w, h))
        boxes = filtered

    return sort_boxes_reading_order(boxes)


def crop_characters(
    gray_image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    output_dir: Path,
    image_stem: str,
    cfg: ExtractionConfig,
) -> list[dict[str, int | str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h_img, w_img = gray_image.shape[:2]

    rows: list[dict[str, int | str]] = []
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        p = max(0, int(cfg.padding))
        x0 = max(0, x - p)
        y0 = max(0, y - p)
        x1 = min(w_img, x + w + p)
        y1 = min(h_img, y + h + p)

        char_img = gray_image[y0:y1, x0:x1]
        if char_img.size == 0:
            continue

        if cfg.resize_to is not None and cfg.resize_to > 0:
            char_img = cv2.resize(
                char_img,
                (cfg.resize_to, cfg.resize_to),
                interpolation=cv2.INTER_AREA,
            )

        out_name = f"{image_stem}_char_{index:03d}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), char_img)

        rows.append(
            {
                "char_index": index,
                "filename": out_name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    return rows


def save_debug_images(
    original_bgr: np.ndarray,
    binary_image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    output_dir: Path,
    image_stem: str,
):
    debug_dir = output_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    boxed = original_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(str(debug_dir / f"{image_stem}_boxed.png"), boxed)
    cv2.imwrite(str(debug_dir / f"{image_stem}_binary.png"), binary_image)


def process_single_image(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    cfg: ExtractionConfig,
    save_debug: bool,
) -> list[dict[str, str | int]]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"[WARN] Impossibile leggere: {image_path}")
        return []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = preprocess(gray, cfg)
    boxes = detect_boxes(binary, cfg)

    rel = image_path.relative_to(input_root)
    out_dir = output_root / rel.parent / rel.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    chars = crop_characters(gray, boxes, out_dir, rel.stem, cfg)

    if save_debug:
        save_debug_images(bgr, binary, boxes, out_dir, rel.stem)

    rows: list[dict[str, str | int]] = []
    for row in chars:
        rows.append(
            {
                "source_image": str(rel).replace("\\", "/"),
                "output_dir": str(out_dir.relative_to(output_root)).replace("\\", "/"),
                "char_index": int(row["char_index"]),
                "filename": str(row["filename"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "w": int(row["w"]),
                "h": int(row["h"]),
            }
        )

    print(f"[OK] {rel} -> {len(rows)} caratteri")
    return rows


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not input_path.is_dir():
        return []

    images = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    images.sort()
    return images


def write_manifest(rows: list[dict[str, str | int]], output_root: Path):
    manifest_path = output_root / "extraction_manifest.csv"
    fieldnames = ["source_image", "output_dir", "char_index", "filename", "x", "y", "w", "h"]

    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest salvato in: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrazione caratteri da immagini CVL (single o batch).",
    )
    parser.add_argument("--input", required=True, help="File immagine o cartella input")
    parser.add_argument("--output", required=True, help="Cartella output")
    parser.add_argument("--save-debug", action="store_true", help="Salva immagini binarie e con bounding box")
    parser.add_argument("--adaptive-block-size", type=int, default=15, help="Block size per adaptive threshold (dispari)")
    parser.add_argument("--adaptive-c", type=int, default=8, help="Costante C per adaptive threshold")
    parser.add_argument("--denoise-h", type=int, default=30, help="Forza denoising")
    parser.add_argument("--erode-kernel", type=int, default=1, help="Kernel size erode")
    parser.add_argument("--dilate-kernel", type=int, default=2, help="Kernel size dilate")
    parser.add_argument("--opening-kernel", type=int, default=1, help="Kernel opening (1 = disattivato)")
    parser.add_argument("--closing-kernel", type=int, default=1, help="Kernel closing (1 = disattivato)")
    parser.add_argument("--min-area", type=int, default=20, help="Area minima bounding box")
    parser.add_argument("--min-width", type=int, default=4, help="Larghezza minima bounding box")
    parser.add_argument("--min-height", type=int, default=6, help="Altezza minima bounding box")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.03, help="Aspect ratio minimo (w/h)")
    parser.add_argument("--max-aspect-ratio", type=float, default=12.0, help="Aspect ratio massimo (w/h)")
    parser.add_argument("--max-width-ratio", type=float, default=0.25, help="Larghezza massima bbox rispetto alla larghezza immagine")
    parser.add_argument("--max-height-ratio", type=float, default=0.25, help="Altezza massima bbox rispetto all'altezza immagine")
    parser.add_argument("--disable-outlier-filter", action="store_true", help="Disattiva il filtro outlier basato su mediane")
    parser.add_argument("--max-area-factor", type=float, default=35.0, help="Soglia area outlier = mediana_area * fattore")
    parser.add_argument("--max-height-factor", type=float, default=5.5, help="Soglia altezza outlier = mediana_h * fattore")
    parser.add_argument("--padding", type=int, default=1, help="Padding crop attorno ai caratteri")
    parser.add_argument("--resize-to", type=int, default=0, help="Ridimensiona ogni carattere a NxN (0 = disattivato)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input non trovato: {input_path}")

    cfg = ExtractionConfig(
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        denoise_h=args.denoise_h,
        erode_kernel=args.erode_kernel,
        dilate_kernel=args.dilate_kernel,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel,
        min_area=args.min_area,
        min_width=args.min_width,
        min_height=args.min_height,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        max_width_ratio=args.max_width_ratio,
        max_height_ratio=args.max_height_ratio,
        filter_outliers=not args.disable_outlier_filter,
        max_area_factor=args.max_area_factor,
        max_height_factor=args.max_height_factor,
        padding=args.padding,
        resize_to=None if args.resize_to <= 0 else args.resize_to,
    )

    images = collect_images(input_path)
    if not images:
        print("Nessuna immagine supportata trovata.")
        return

    input_root = input_path if input_path.is_dir() else input_path.parent

    all_rows: list[dict[str, str | int]] = []
    for image_path in images:
        rows = process_single_image(
            image_path=image_path,
            input_root=input_root,
            output_root=output_root,
            cfg=cfg,
            save_debug=args.save_debug,
        )
        all_rows.extend(rows)

    write_manifest(all_rows, output_root)

    print("\n=== Estrazione completata ===")
    print(f"Immagini processate: {len(images)}")
    print(f"Caratteri estratti: {len(all_rows)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
