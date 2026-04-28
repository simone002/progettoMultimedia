# Il "cuore": Hit-or-Miss, Pruning e Ricostruzione

import numpy as np
from skimage import morphology, measure
from scipy.ndimage import binary_hit_or_miss, distance_transform_edt

# questo file contiene le funzioni principali per l'analisi morfologica:

def get_endpoint_kernels():
    """Ritorna gli 8 kernel per trovare i pixel terminali (punte)."""
    # Kernel per un pixel che ha un solo vicino (N, S, E, W + diagonali)
    '''
    Analisi delle Punte: Usando gli 8 kernel nella Hit-or-Miss,
    abbiamo insegnato al computer a riconoscere dove finisce una linea.
    '''
    kernels = [
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]), # Punta in alto
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]), # Punta in basso
        np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]), # Punta a sinistra
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]), # Punta a destra
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]), # Diagonale 1
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]), # Diagonale 2
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]), # Diagonale 3
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # Diagonale 4
    ]
    return kernels

def find_endpoints(skeleton):
    """Trova tutte le terminazioni dello scheletro usando la Hit-or-Miss."""
    endpoints = np.zeros_like(skeleton)
    kernels = get_endpoint_kernels()
    
    for k in kernels:
        # Applichiamo la Hit-or-Miss: cerca dove il kernel coincide perfettamente
        endpoints |= binary_hit_or_miss(skeleton, structure1=k)
    return endpoints

def pruning(skeleton, iterations=5):
    """Elimina i rami spuri (peli) corti dallo scheletro."""

    '''
    La Potatura (Pruning): Eliminando iterativamente queste punte, "accorciamo" i rami.
    I rami di rumore (molto corti) spariscono, mentre le lettere (tratti lunghi)
    rimangono quasi intatte.
    '''
    pruned = skeleton.copy()
    for _ in range(iterations):
        endpoints = find_endpoints(pruned)
        # Rimuoviamo i punti terminali trovati
        pruned[endpoints] = 0
    return pruned

def morphological_reconstruction(marker, mask):
    """Esegue la ricostruzione geodesica per recuperare le parti connesse."""

    '''
    Il Restauro: La ricostruzione usa lo scheletro pulito per "riempire" di nuovo solo
    le lettere dell'immagine originale, cancellando per sempre lo sporco non connesso.
    '''
    return morphology.reconstruction(marker, mask, method='dilation')

def get_junction_kernels():
    """Ritorna kernel per trovare le biforcazioni (punti di incrocio)."""
    # Esempi di configurazioni a T o a X
    kernels = [
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), # Croce a X
    ]
    return kernels

def find_junctions(skeleton):
    """Trova i punti di intersezione nello scheletro."""
    junctions = np.zeros_like(skeleton)
    kernels = get_junction_kernels()
    for k in kernels:
        junctions |= binary_hit_or_miss(skeleton, structure1=k)
    return junctions

def count_holes(binary_image):
    """Conta i buchi chiusi (es. interno della 'o', 'b', 'p')."""
    # Usiamo l'etichettatura dei componenti connessi sullo sfondo
    inverted = ~binary_image
    labels = measure.label(inverted)
    # Sottraiamo 1 perché il bordo esterno conta come un componente
    return max(0, labels.max() - 1)


def skeleton_length(skeleton):
    """Conta i pixel dello scheletro (proxy della lunghezza del tratto)."""
    return int(np.sum(skeleton))


def euler_number(binary_image):
    """Calcola il numero di Euler: componenti - buchi."""
    return int(measure.euler_number(binary_image, connectivity=2))


def hole_area_ratio(binary_image):
    """Rapporto area buchi / area foreground."""
    labels_bg = measure.label(~binary_image)
    if labels_bg.max() == 0:
        return 0.0

    border_labels = set(np.unique(labels_bg[0, :]))
    border_labels |= set(np.unique(labels_bg[-1, :]))
    border_labels |= set(np.unique(labels_bg[:, 0]))
    border_labels |= set(np.unique(labels_bg[:, -1]))

    hole_mask = np.zeros_like(binary_image, dtype=bool)
    for label_id in range(1, labels_bg.max() + 1):
        if label_id not in border_labels:
            hole_mask |= labels_bg == label_id

    hole_area = float(np.sum(hole_mask))
    foreground_area = float(np.sum(binary_image))
    if foreground_area <= 0:
        return 0.0
    return hole_area / foreground_area


def average_stroke_width(binary_image, skeleton):
    """Stima la larghezza media del tratto usando distance transform sul foreground."""
    if np.sum(skeleton) == 0:
        return 0.0
    dist = distance_transform_edt(binary_image)
    widths = 2.0 * dist[skeleton]
    if widths.size == 0:
        return 0.0
    return float(np.mean(widths))


def global_shape_features(binary_image):
    """Feature geometriche della componente principale."""
    labeled = measure.label(binary_image.astype(bool))
    regions = measure.regionprops(labeled)
    if not regions:
        return {
            "solidity": 0.0,
            "extent": 0.0,
            "eccentricity": 0.0,
            "orientation": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
            "axis_ratio": 0.0,
        }

    region = max(regions, key=lambda r: r.area)
    major = float(region.axis_major_length)
    minor = float(region.axis_minor_length)
    axis_ratio = major / max(1e-6, minor)

    return {
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "eccentricity": float(region.eccentricity),
        "orientation": float(region.orientation),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "axis_ratio": float(axis_ratio),
    }


def hu_moments_features(binary_image):
    """Momenti di Hu (log-compressi) sulla maschera binaria."""
    image = binary_image.astype(float)
    if np.sum(image) == 0:
        return {f"hu_{i+1}": 0.0 for i in range(7)}

    moments = measure.moments(image)
    center_r = moments[1, 0] / max(1e-12, moments[0, 0])
    center_c = moments[0, 1] / max(1e-12, moments[0, 0])

    central = measure.moments_central(image, center=(center_r, center_c))
    normalized = measure.moments_normalized(central)
    hu = measure.moments_hu(normalized)

    out = {}
    for i, value in enumerate(hu, 1):
        signed_log = np.sign(value) * np.log10(abs(value) + 1e-30)
        out[f"hu_{i}"] = float(signed_log)
    return out


def skeleton_branch_stats(skeleton):
    """Statistiche dei rami rimuovendo i nodi di giunzione."""
    skel = skeleton.astype(bool)
    if np.sum(skel) == 0:
        return {
            "n_branches": 0,
            "branch_length_mean": 0.0,
            "branch_length_max": 0.0,
            "branch_length_std": 0.0,
        }

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    from scipy.ndimage import convolve

    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    junction_mask = skel & (neighbor_count >= 3)

    segment_mask = skel & (~junction_mask)
    segment_labels = measure.label(segment_mask)
    if segment_labels.max() == 0:
        length = float(np.sum(skel))
        return {
            "n_branches": 1 if length > 0 else 0,
            "branch_length_mean": length,
            "branch_length_max": length,
            "branch_length_std": 0.0,
        }

    lengths = []
    for idx in range(1, segment_labels.max() + 1):
        lengths.append(float(np.sum(segment_labels == idx)))

    lengths_arr = np.asarray(lengths, dtype=float)
    return {
        "n_branches": int(len(lengths)),
        "branch_length_mean": float(np.mean(lengths_arr)),
        "branch_length_max": float(np.max(lengths_arr)),
        "branch_length_std": float(np.std(lengths_arr)),
    }


def zoning_density_features(binary_image, grid_rows=3, grid_cols=3):
    """Densità di foreground per celle di una griglia (zoning)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    row_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

    out = {}
    for r in range(grid_rows):
        for c in range(grid_cols):
            r0, r1 = row_edges[r], row_edges[r + 1]
            c0, c1 = col_edges[c], col_edges[c + 1]
            cell = img[r0:r1, c0:c1]
            density = float(np.mean(cell)) if cell.size > 0 else 0.0
            out[f"zone_r{r}_c{c}_density"] = density
    return out


def projection_profile_features(binary_image):
    """Statistiche sui profili di proiezione orizzontale/verticale."""
    img = binary_image.astype(bool)
    h, w = img.shape

    def _shape_stats(x: np.ndarray, prefix: str):
        if x.size == 0:
            return {
                f"{prefix}_q25": 0.0,
                f"{prefix}_q50": 0.0,
                f"{prefix}_q75": 0.0,
                f"{prefix}_skew": 0.0,
                f"{prefix}_kurtosis": 0.0,
            }

        mean = float(np.mean(x))
        std = float(np.std(x))
        if std < 1e-12:
            skew = 0.0
            kurt = 0.0
        else:
            centered = x - mean
            z = centered / std
            skew = float(np.mean(z**3))
            kurt = float(np.mean(z**4))

        return {
            f"{prefix}_q25": float(np.quantile(x, 0.25)),
            f"{prefix}_q50": float(np.quantile(x, 0.50)),
            f"{prefix}_q75": float(np.quantile(x, 0.75)),
            f"{prefix}_skew": skew,
            f"{prefix}_kurtosis": kurt,
        }

    if h == 0 or w == 0:
        return {
            "row_proj_mean": 0.0,
            "row_proj_std": 0.0,
            "row_proj_max": 0.0,
            "row_proj_argmax_norm": 0.0,
            "row_nonzero_ratio": 0.0,
            "row_proj_q25": 0.0,
            "row_proj_q50": 0.0,
            "row_proj_q75": 0.0,
            "row_proj_skew": 0.0,
            "row_proj_kurtosis": 0.0,
            "col_proj_mean": 0.0,
            "col_proj_std": 0.0,
            "col_proj_max": 0.0,
            "col_proj_argmax_norm": 0.0,
            "col_nonzero_ratio": 0.0,
            "col_proj_q25": 0.0,
            "col_proj_q50": 0.0,
            "col_proj_q75": 0.0,
            "col_proj_skew": 0.0,
            "col_proj_kurtosis": 0.0,
        }

    row_proj = np.sum(img, axis=1).astype(float) / max(1.0, float(w))
    col_proj = np.sum(img, axis=0).astype(float) / max(1.0, float(h))

    row_argmax = int(np.argmax(row_proj)) if row_proj.size else 0
    col_argmax = int(np.argmax(col_proj)) if col_proj.size else 0

    out = {
        "row_proj_mean": float(np.mean(row_proj)) if row_proj.size else 0.0,
        "row_proj_std": float(np.std(row_proj)) if row_proj.size else 0.0,
        "row_proj_max": float(np.max(row_proj)) if row_proj.size else 0.0,
        "row_proj_argmax_norm": float(row_argmax / max(1, h - 1)),
        "row_nonzero_ratio": float(np.mean(row_proj > 0)) if row_proj.size else 0.0,
        "col_proj_mean": float(np.mean(col_proj)) if col_proj.size else 0.0,
        "col_proj_std": float(np.std(col_proj)) if col_proj.size else 0.0,
        "col_proj_max": float(np.max(col_proj)) if col_proj.size else 0.0,
        "col_proj_argmax_norm": float(col_argmax / max(1, w - 1)),
        "col_nonzero_ratio": float(np.mean(col_proj > 0)) if col_proj.size else 0.0,
    }

    out.update(_shape_stats(row_proj, "row_proj"))
    out.update(_shape_stats(col_proj, "col_proj"))
    return out


def spatial_balance_features(binary_image):
    """Bilanciamento di inchiostro tra metà dell'immagine."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "top_half_density": 0.0,
            "bottom_half_density": 0.0,
            "left_half_density": 0.0,
            "right_half_density": 0.0,
            "top_bottom_ratio": 0.0,
            "left_right_ratio": 0.0,
        }

    mid_h = h // 2
    mid_w = w // 2
    top = img[:mid_h, :]
    bottom = img[mid_h:, :]
    left = img[:, :mid_w]
    right = img[:, mid_w:]

    top_d = float(np.mean(top)) if top.size else 0.0
    bottom_d = float(np.mean(bottom)) if bottom.size else 0.0
    left_d = float(np.mean(left)) if left.size else 0.0
    right_d = float(np.mean(right)) if right.size else 0.0

    return {
        "top_half_density": top_d,
        "bottom_half_density": bottom_d,
        "left_half_density": left_d,
        "right_half_density": right_d,
        "top_bottom_ratio": top_d / max(1e-6, bottom_d),
        "left_right_ratio": left_d / max(1e-6, right_d),
    }


def transition_features(binary_image):
    """Statistiche sulle transizioni 0/1 per righe e colonne."""
    img = binary_image.astype(np.uint8)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "row_trans_mean": 0.0,
            "row_trans_std": 0.0,
            "row_trans_max": 0.0,
            "col_trans_mean": 0.0,
            "col_trans_std": 0.0,
            "col_trans_max": 0.0,
        }

    row_transitions = np.sum(np.abs(np.diff(img, axis=1)), axis=1)
    col_transitions = np.sum(np.abs(np.diff(img, axis=0)), axis=0)

    return {
        "row_trans_mean": float(np.mean(row_transitions)) if row_transitions.size else 0.0,
        "row_trans_std": float(np.std(row_transitions)) if row_transitions.size else 0.0,
        "row_trans_max": float(np.max(row_transitions)) if row_transitions.size else 0.0,
        "col_trans_mean": float(np.mean(col_transitions)) if col_transitions.size else 0.0,
        "col_trans_std": float(np.std(col_transitions)) if col_transitions.size else 0.0,
        "col_trans_max": float(np.max(col_transitions)) if col_transitions.size else 0.0,
    }


def diagonal_profile_features(binary_image):
    """Densità su diagonale principale e secondaria (utile per forme tipo z/x)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "main_diag_density": 0.0,
            "anti_diag_density": 0.0,
            "diag_density_diff": 0.0,
        }

    n = min(h, w)
    if n <= 0:
        return {
            "main_diag_density": 0.0,
            "anti_diag_density": 0.0,
            "diag_density_diff": 0.0,
        }

    row_idx = np.linspace(0, h - 1, n).astype(int)
    col_main = np.linspace(0, w - 1, n).astype(int)
    col_anti = np.linspace(w - 1, 0, n).astype(int)

    main_d = float(np.mean(img[row_idx, col_main]))
    anti_d = float(np.mean(img[row_idx, col_anti]))
    return {
        "main_diag_density": main_d,
        "anti_diag_density": anti_d,
        "diag_density_diff": main_d - anti_d,
    }


def endpoint_position_features(endpoint_mask):
    """Posizione media normalizzata delle terminazioni dello scheletro."""
    m = endpoint_mask.astype(bool)
    h, w = m.shape
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return {
            "endpoint_y_mean_norm": 0.0,
            "endpoint_x_mean_norm": 0.0,
            "endpoint_y_std_norm": 0.0,
            "endpoint_x_std_norm": 0.0,
        }

    return {
        "endpoint_y_mean_norm": float(np.mean(ys) / max(1, h - 1)),
        "endpoint_x_mean_norm": float(np.mean(xs) / max(1, w - 1)),
        "endpoint_y_std_norm": float(np.std(ys) / max(1, h - 1)),
        "endpoint_x_std_norm": float(np.std(xs) / max(1, w - 1)),
    }


def _count_foreground_runs_1d(arr_1d: np.ndarray) -> int:
    """Conta i segmenti contigui di foreground in un array 1D booleano."""
    a = arr_1d.astype(np.uint8)
    if a.size == 0:
        return 0
    padded = np.pad(a, (1, 1), constant_values=0)
    starts = np.sum((padded[1:-1] == 1) & (padded[:-2] == 0))
    return int(starts)


def scanline_structure_features(binary_image):
    """Feature strutturali su righe/colonne a quartili (utile per lettere confuse)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "row_q25_runs": 0.0,
            "row_q50_runs": 0.0,
            "row_q75_runs": 0.0,
            "row_q25_fill": 0.0,
            "row_q50_fill": 0.0,
            "row_q75_fill": 0.0,
            "col_q25_runs": 0.0,
            "col_q50_runs": 0.0,
            "col_q75_runs": 0.0,
            "col_q25_fill": 0.0,
            "col_q50_fill": 0.0,
            "col_q75_fill": 0.0,
        }

    rq = [int(round((h - 1) * q)) for q in (0.25, 0.50, 0.75)]
    cq = [int(round((w - 1) * q)) for q in (0.25, 0.50, 0.75)]

    row_vals = []
    for r in rq:
        line = img[r, :]
        row_vals.append((_count_foreground_runs_1d(line), float(np.mean(line))))

    col_vals = []
    for c in cq:
        line = img[:, c]
        col_vals.append((_count_foreground_runs_1d(line), float(np.mean(line))))

    return {
        "row_q25_runs": float(row_vals[0][0]),
        "row_q50_runs": float(row_vals[1][0]),
        "row_q75_runs": float(row_vals[2][0]),
        "row_q25_fill": row_vals[0][1],
        "row_q50_fill": row_vals[1][1],
        "row_q75_fill": row_vals[2][1],
        "col_q25_runs": float(col_vals[0][0]),
        "col_q50_runs": float(col_vals[1][0]),
        "col_q75_runs": float(col_vals[2][0]),
        "col_q25_fill": col_vals[0][1],
        "col_q50_fill": col_vals[1][1],
        "col_q75_fill": col_vals[2][1],
    }


def border_contact_features(binary_image):
    """Quanto il tratto tocca i bordi (normalizzato)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "top_border_contact": 0.0,
            "bottom_border_contact": 0.0,
            "left_border_contact": 0.0,
            "right_border_contact": 0.0,
        }

    top = float(np.mean(img[0, :])) if w > 0 else 0.0
    bottom = float(np.mean(img[-1, :])) if w > 0 else 0.0
    left = float(np.mean(img[:, 0])) if h > 0 else 0.0
    right = float(np.mean(img[:, -1])) if h > 0 else 0.0

    return {
        "top_border_contact": top,
        "bottom_border_contact": bottom,
        "left_border_contact": left,
        "right_border_contact": right,
    }


def hole_position_features(binary_image):
    """Posizione media normalizzata delle cavità interne (se presenti)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "hole_centroid_y_norm": 0.0,
            "hole_centroid_x_norm": 0.0,
            "hole_centroid_spread": 0.0,
        }

    labels_bg = measure.label(~img)
    if labels_bg.max() == 0:
        return {
            "hole_centroid_y_norm": 0.0,
            "hole_centroid_x_norm": 0.0,
            "hole_centroid_spread": 0.0,
        }

    border_labels = set(np.unique(labels_bg[0, :]))
    border_labels |= set(np.unique(labels_bg[-1, :]))
    border_labels |= set(np.unique(labels_bg[:, 0]))
    border_labels |= set(np.unique(labels_bg[:, -1]))

    hole_mask = np.zeros_like(img, dtype=bool)
    for lid in range(1, labels_bg.max() + 1):
        if lid not in border_labels:
            hole_mask |= labels_bg == lid

    ys, xs = np.where(hole_mask)
    if ys.size == 0:
        return {
            "hole_centroid_y_norm": 0.0,
            "hole_centroid_x_norm": 0.0,
            "hole_centroid_spread": 0.0,
        }

    y_norm = float(np.mean(ys) / max(1, h - 1))
    x_norm = float(np.mean(xs) / max(1, w - 1))
    spread = float((np.std(ys) / max(1, h - 1)) + (np.std(xs) / max(1, w - 1)))

    return {
        "hole_centroid_y_norm": y_norm,
        "hole_centroid_x_norm": x_norm,
        "hole_centroid_spread": spread,
    }


def contour_depth_features(binary_image):
    """Profili top/bottom per colonna: utili su archi/valle (es. u/n, f/t)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "top_depth_mean_norm": 0.0,
            "top_depth_std_norm": 0.0,
            "top_depth_range_norm": 0.0,
            "bottom_depth_mean_norm": 0.0,
            "bottom_depth_std_norm": 0.0,
            "bottom_depth_range_norm": 0.0,
            "top_center_depth_mean_norm": 0.0,
            "bottom_center_depth_mean_norm": 0.0,
        }

    top_depths = []
    bottom_depths = []
    top_center = []
    bottom_center = []

    c0 = int(round(w * 0.33))
    c1 = int(round(w * 0.67))
    if c1 <= c0:
        c0, c1 = 0, w

    for c in range(w):
        ys = np.where(img[:, c])[0]
        if ys.size == 0:
            continue

        top_d = float(ys.min())
        bottom_d = float((h - 1) - ys.max())
        top_depths.append(top_d)
        bottom_depths.append(bottom_d)

        if c0 <= c < c1:
            top_center.append(top_d)
            bottom_center.append(bottom_d)

    if not top_depths:
        return {
            "top_depth_mean_norm": 0.0,
            "top_depth_std_norm": 0.0,
            "top_depth_range_norm": 0.0,
            "bottom_depth_mean_norm": 0.0,
            "bottom_depth_std_norm": 0.0,
            "bottom_depth_range_norm": 0.0,
            "top_center_depth_mean_norm": 0.0,
            "bottom_center_depth_mean_norm": 0.0,
        }

    top_arr = np.asarray(top_depths, dtype=float)
    bottom_arr = np.asarray(bottom_depths, dtype=float)
    denom = max(1.0, float(h - 1))

    top_center_mean = float(np.mean(top_center)) if top_center else float(np.mean(top_arr))
    bottom_center_mean = float(np.mean(bottom_center)) if bottom_center else float(np.mean(bottom_arr))

    return {
        "top_depth_mean_norm": float(np.mean(top_arr) / denom),
        "top_depth_std_norm": float(np.std(top_arr) / denom),
        "top_depth_range_norm": float((np.max(top_arr) - np.min(top_arr)) / denom),
        "bottom_depth_mean_norm": float(np.mean(bottom_arr) / denom),
        "bottom_depth_std_norm": float(np.std(bottom_arr) / denom),
        "bottom_depth_range_norm": float((np.max(bottom_arr) - np.min(bottom_arr)) / denom),
        "top_center_depth_mean_norm": float(top_center_mean / denom),
        "bottom_center_depth_mean_norm": float(bottom_center_mean / denom),
    }


def endpoint_distribution_features(endpoint_mask):
    """Distribuzione spaziale delle terminazioni (utile su f/t e u/n)."""
    m = endpoint_mask.astype(bool)
    h, w = m.shape
    ys, xs = np.where(m)

    if ys.size == 0:
        return {
            "endpoints_top_ratio": 0.0,
            "endpoints_bottom_ratio": 0.0,
            "endpoints_left_ratio": 0.0,
            "endpoints_right_ratio": 0.0,
            "endpoints_lower_quarter_ratio": 0.0,
        }

    top_ratio = float(np.mean(ys < (h * 0.5)))
    bottom_ratio = float(np.mean(ys >= (h * 0.5)))
    left_ratio = float(np.mean(xs < (w * 0.5)))
    right_ratio = float(np.mean(xs >= (w * 0.5)))
    lower_quarter_ratio = float(np.mean(ys >= (h * 0.75)))

    return {
        "endpoints_top_ratio": top_ratio,
        "endpoints_bottom_ratio": bottom_ratio,
        "endpoints_left_ratio": left_ratio,
        "endpoints_right_ratio": right_ratio,
        "endpoints_lower_quarter_ratio": lower_quarter_ratio,
    }


def _count_runs_1d(arr_1d: np.ndarray, value: int) -> int:
    a = (arr_1d.astype(np.uint8) == np.uint8(value)).astype(np.uint8)
    if a.size == 0:
        return 0
    padded = np.pad(a, (1, 1), constant_values=0)
    starts = np.sum((padded[1:-1] == 1) & (padded[:-2] == 0))
    return int(starts)


def midline_run_features(binary_image):
    """Pattern su riga/colonna mediana e apertura lato destro (utile su e/a, f/t)."""
    img = binary_image.astype(bool)
    h, w = img.shape
    if h == 0 or w == 0:
        return {
            "mid_row_fg_runs": 0.0,
            "mid_row_bg_runs": 0.0,
            "mid_row_fg_fill": 0.0,
            "mid_row_right_bg_tail_norm": 0.0,
            "mid_col_fg_runs": 0.0,
            "mid_col_fg_fill": 0.0,
            "upper_third_density": 0.0,
            "lower_third_density": 0.0,
            "upper_lower_density_ratio": 0.0,
        }

    mid_r = h // 2
    mid_c = w // 2
    row = img[mid_r, :]
    col = img[:, mid_c]

    fg_runs_row = _count_runs_1d(row.astype(np.uint8), value=1)
    bg_runs_row = _count_runs_1d(row.astype(np.uint8), value=0)
    fg_runs_col = _count_runs_1d(col.astype(np.uint8), value=1)

    right_tail = 0
    for val in row[::-1]:
        if val:
            break
        right_tail += 1

    t0, t1 = 0, max(1, int(round(h / 3.0)))
    b0, b1 = min(h, int(round(2 * h / 3.0))), h
    upper_density = float(np.mean(img[t0:t1, :])) if t1 > t0 else 0.0
    lower_density = float(np.mean(img[b0:b1, :])) if b1 > b0 else 0.0
    raw_ratio = float((upper_density + 1e-4) / (lower_density + 1e-4))
    stable_ratio = float(np.clip(raw_ratio, 0.0, 10.0))

    return {
        "mid_row_fg_runs": float(fg_runs_row),
        "mid_row_bg_runs": float(bg_runs_row),
        "mid_row_fg_fill": float(np.mean(row)),
        "mid_row_right_bg_tail_norm": float(right_tail / max(1, w)),
        "mid_col_fg_runs": float(fg_runs_col),
        "mid_col_fg_fill": float(np.mean(col)),
        "upper_third_density": upper_density,
        "lower_third_density": lower_density,
        "upper_lower_density_ratio": stable_ratio,
    }