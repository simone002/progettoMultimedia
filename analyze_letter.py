import os
from pathlib import Path
import pandas as pd
from src.preprocessing import load_and_binarize
from src.morphology_logic import (
    pruning,
    find_endpoints,
    find_junctions,
    count_holes,
    skeleton_length,
    euler_number,
    hole_area_ratio,
    average_stroke_width,
    global_shape_features,
    hu_moments_features,
    skeleton_branch_stats,
    zoning_density_features,
    projection_profile_features,
    spatial_balance_features,
    transition_features,
    diagonal_profile_features,
    endpoint_position_features,
    scanline_structure_features,
    border_contact_features,
    hole_position_features,
)
from skimage import morphology, measure

BASE_DIR = Path(__file__).resolve().parent
LETTERE_DIR = BASE_DIR / 'data' / 'lettere'
OUTPUT_DIR = BASE_DIR / 'output' / 'analysis'


def _get_letters(base_path: Path):
    return sorted([d.name for d in base_path.iterdir() if d.is_dir()])


def _connected_components(binary):
    labels = measure.label(binary)
    return int(labels.max())


def _bbox_aspect_ratio(binary):
    coords = measure.regionprops(measure.label(binary.astype(int)))
    if not coords:
        return 0.0
    # Usa la regione più grande (tratto principale)
    largest = max(coords, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    h = max(1, maxr - minr)
    w = max(1, maxc - minc)
    return float(w / h)


def _pixel_density(binary):
    return float(binary.sum() / binary.size)

def analyze_alphabet():
    results = []
    if not LETTERE_DIR.exists():
        raise FileNotFoundError(f"Percorso non trovato: {LETTERE_DIR}")

    # Trova tutti i soggetti disponibili
    subjects = sorted([d.name for d in LETTERE_DIR.iterdir() if d.is_dir()])
    print(f"Soggetti trovati: {len(subjects)}")
    print("Avvio analisi...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_errors = 0

    for subject_idx, subject in enumerate(subjects, 1):
        print(f"\n[{subject_idx}/{len(subjects)}] Elaborazione soggetto {subject}...")
        subject_path = LETTERE_DIR / subject
        letters = _get_letters(subject_path)
        
        for char in letters:
            char_path = subject_path / char
            if not char_path.exists():
                continue
            
            # Analizziamo tutti i file disponibili
            files = sorted(
                [f for f in os.listdir(char_path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
            )
            
            for f in files:
                path = char_path / f
                try:
                    binary = load_and_binarize(path)
                    
                    # Pulizia e Scheletrizzazione
                    skeleton = morphology.skeletonize(binary)
                    clean_skel = pruning(skeleton, iterations=3)

                    skeleton_len_raw = skeleton_length(skeleton)
                    skeleton_len_clean = skeleton_length(clean_skel)
                    removed_ratio = (
                        (skeleton_len_raw - skeleton_len_clean) / max(1, skeleton_len_raw)
                    )
                    
                    # Estrazione Features
                    endpoint_mask = find_endpoints(clean_skel)
                    endpoints = endpoint_mask.sum()
                    junctions = find_junctions(clean_skel).sum()
                    holes = count_holes(binary)
                    components = _connected_components(binary)
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

                    endpoint_norm = float(endpoints) / max(1, skeleton_len_clean)
                    junction_norm = float(junctions) / max(1, skeleton_len_clean)
                    skeleton_density = float(skeleton_len_clean) / max(1.0, float(binary.sum()))
                    
                    sample_row = {
                        'soggetto': subject,
                        'lettera': char,
                        'file': f,
                        'punte': endpoints,
                        'incroci': junctions,
                        'buchi': holes,
                        'componenti': components,
                        'aspect_ratio': round(aspect_ratio, 4),
                        'densita_pixel': round(density, 4),
                        'euler_number': euler,
                        'hole_area_ratio': round(hole_ratio, 4),
                        'avg_stroke_width': round(stroke_w, 4),
                        'skeleton_length': skeleton_len_clean,
                        'endpoint_norm': round(endpoint_norm, 6),
                        'junction_norm': round(junction_norm, 6),
                        'skeleton_density': round(skeleton_density, 6),
                        'pruning_removed_ratio': round(removed_ratio, 6),
                        'solidity': round(shape_feats['solidity'], 6),
                        'extent': round(shape_feats['extent'], 6),
                        'eccentricity': round(shape_feats['eccentricity'], 6),
                        'orientation': round(shape_feats['orientation'], 6),
                        'major_axis_length': round(shape_feats['major_axis_length'], 6),
                        'minor_axis_length': round(shape_feats['minor_axis_length'], 6),
                        'axis_ratio': round(shape_feats['axis_ratio'], 6),
                        'n_branches': branch_feats['n_branches'],
                        'branch_length_mean': round(branch_feats['branch_length_mean'], 6),
                        'branch_length_max': round(branch_feats['branch_length_max'], 6),
                        'branch_length_std': round(branch_feats['branch_length_std'], 6),
                        'hu_1': round(hu_feats['hu_1'], 6),
                        'hu_2': round(hu_feats['hu_2'], 6),
                        'hu_3': round(hu_feats['hu_3'], 6),
                        'hu_4': round(hu_feats['hu_4'], 6),
                        'hu_5': round(hu_feats['hu_5'], 6),
                        'hu_6': round(hu_feats['hu_6'], 6),
                        'hu_7': round(hu_feats['hu_7'], 6),
                    }

                    for k, v in zoning_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in proj_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in balance_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in trans_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in diag_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in endpoint_pos_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in scanline_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in border_feats.items():
                        sample_row[k] = round(float(v), 6)

                    for k, v in hole_pos_feats.items():
                        sample_row[k] = round(float(v), 6)

                    results.append(sample_row)
                    total_processed += 1
                except Exception as e:
                    print(f"  Errore su {subject}/{char}/{f}: {e}")
                    total_errors += 1
                    continue
    
    print(f"\n\n=== Elaborazione completata ===")
    print(f"Campioni elaborati con successo: {total_processed}")
    print(f"Errori: {total_errors}")

    # Creazione tabella riassuntiva
    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("Nessun campione trovato per l'analisi.")

    print(f"\nCampioni analizzati: {len(df)}")
    print(f"Lettere uniche: {df['lettera'].nunique()}")
    print(f"Soggetti unici: {df['soggetto'].nunique()}")

    summary = (
        df.drop(columns=['soggetto', 'file'])
          .groupby('lettera')
          .mean(numeric_only=True)
          .round(3)
    )

    # Aggiungi anche analisi per soggetto
    summary_per_subject = (
        df.drop(columns=['file'])
          .groupby(['soggetto', 'lettera'])
          .mean(numeric_only=True)
          .round(3)
    )

    samples_out = OUTPUT_DIR / 'features_samples.csv'
    summary_out = OUTPUT_DIR / 'features_summary.csv'
    summary_subject_out = OUTPUT_DIR / 'features_summary_per_subject.csv'
    
    df.to_csv(samples_out, index=False)
    summary.to_csv(summary_out)
    summary_per_subject.to_csv(summary_subject_out)

    print("\n--- Statistiche Medie per Lettera ---")
    print(summary)
    print(f"\nCSV campioni salvato in: {samples_out}")
    print(f"CSV summary salvato in: {summary_out}")
    print(f"CSV summary per soggetto salvato in: {summary_subject_out}")
    return summary

if __name__ == "__main__":
    analyze_alphabet()