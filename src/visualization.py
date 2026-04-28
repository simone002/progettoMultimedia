# Funzioni per creare i grafici e confronti
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#questo file contiene funzioni per creare grafici e confronti tra le lettere basati 
# sui dati estratti da analyze_letter.py.

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_CSV = BASE_DIR / 'output' / 'analysis' / 'features_summary.csv'
SAMPLES_CSV = BASE_DIR / 'output' / 'analysis' / 'features_samples.csv'
PLOTS_DIR = BASE_DIR / 'output' / 'analysis' / 'plots'
NUMERIC_COLS = ['punte', 'incroci', 'buchi', 'componenti', 'aspect_ratio', 'densita_pixel']


def plot_feature_space():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {SUMMARY_CSV}. Esegui prima analyze_letter.py"
        )

    df = pd.read_csv(SUMMARY_CSV)

    plt.figure(figsize=(12, 8))
    plt.scatter(df['punte'], df['buchi'], color='teal', s=100)

    # Aggiungiamo le etichette per ogni punto
    for _, row in df.iterrows():
        plt.annotate(
            row['lettera'],
            (row['punte'] + 0.05, row['buchi'] + 0.05),
            fontsize=12,
            fontweight='bold'
        )

    plt.title("Distribuzione delle Lettere nel Feature Space (Morfologia)")
    plt.xlabel("Numero Medio di Punte (Endpoints)")
    plt.ylabel("Numero Medio di Buchi (Holes)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out = PLOTS_DIR / 'feature_space_scatter.png'
    plt.savefig(out, dpi=150)
    plt.show()


def plot_correlation_heatmap():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {SUMMARY_CSV}. Esegui prima analyze_letter.py"
        )

    df = pd.read_csv(SUMMARY_CSV)
    numeric_cols = [c for c in df.columns if c != 'lettera' and pd.api.types.is_numeric_dtype(df[c])]

    feature_groups = {
        'base_core': [
            'punte', 'incroci', 'buchi', 'componenti', 'aspect_ratio', 'densita_pixel'
        ],
        'topology_adv': [
            'euler_number', 'hole_area_ratio', 'endpoint_norm', 'junction_norm', 'pruning_removed_ratio'
        ],
        'stroke_geom': ['avg_stroke_width', 'skeleton_length', 'skeleton_density'],
        'global_shape': [
            'solidity', 'extent', 'eccentricity', 'orientation', 'major_axis_length', 'minor_axis_length', 'axis_ratio'
        ],
        'branches': ['n_branches', 'branch_length_mean', 'branch_length_max', 'branch_length_std'],
        'hu_moments': [c for c in numeric_cols if c.startswith('hu_')],
        'zoning': [c for c in numeric_cols if c.startswith('zone_')],
        'proj_row': [c for c in numeric_cols if c.startswith('row_proj_')],
        'proj_col': [c for c in numeric_cols if c.startswith('col_proj_')],
    }

    grouped = {g: [c for c in cols if c in numeric_cols] for g, cols in feature_groups.items()}
    grouped = {g: cols for g, cols in grouped.items() if cols}

    corr = df[numeric_cols].corr().abs()
    group_names = list(grouped.keys())
    group_matrix = np.zeros((len(group_names), len(group_names)), dtype=float)

    for i, g1 in enumerate(group_names):
        cols1 = grouped[g1]
        for j, g2 in enumerate(group_names):
            cols2 = grouped[g2]
            block = corr.loc[cols1, cols2].values
            group_matrix[i, j] = float(np.mean(block)) if block.size else 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(group_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(group_names)))
    ax.set_yticks(np.arange(len(group_names)))
    ax.set_xticklabels(group_names, rotation=35, ha='right')
    ax.set_yticklabels(group_names)

    for i in range(len(group_names)):
        for j in range(len(group_names)):
            ax.text(j, i, f"{group_matrix[i, j]:.2f}", ha='center', va='center', fontsize=9, color='white')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Correlazione media assoluta tra gruppi di feature')
    plt.tight_layout()
    out = PLOTS_DIR / 'feature_correlation_heatmap.png'
    plt.savefig(out, dpi=150)
    plt.show()


def plot_boxplots_by_letter():
    if not SAMPLES_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {SAMPLES_CSV}. Esegui prima analyze_letter.py"
        )

    df = pd.read_csv(SAMPLES_CSV)
    letters = sorted(df['lettera'].unique())

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(NUMERIC_COLS):
        data_by_letter = [df.loc[df['lettera'] == l, feature].values for l in letters]
        axes[idx].boxplot(data_by_letter, tick_labels=letters, showfliers=True)
        axes[idx].set_title(f'Distribuzione {feature} per lettera')
        axes[idx].set_xlabel('Lettera')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / 'feature_boxplots_by_letter.png'
    plt.savefig(out, dpi=150)
    plt.show()


def plot_pca_2d():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {SUMMARY_CSV}. Esegui prima analyze_letter.py"
        )

    df = pd.read_csv(SUMMARY_CSV)
    pca_cols = [c for c in df.columns if c != 'lettera' and pd.api.types.is_numeric_dtype(df[c])]
    X = df[pca_cols].to_numpy(dtype=float)

    # Standardizzazione per evitare che feature con scale diverse dominino la PCA
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    # PCA via decomposizione agli autovettori della covarianza
    cov = np.cov(Xs, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W = eigvecs[:, :2]
    Z = Xs @ W

    explained = eigvals / eigvals.sum()
    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100

    plt.figure(figsize=(12, 8))
    plt.scatter(Z[:, 0], Z[:, 1], color='darkorange', s=110)

    for i, row in df.iterrows():
        plt.annotate(
            row['lettera'],
            (Z[i, 0] + 0.03, Z[i, 1] + 0.03),
            fontsize=11,
            fontweight='bold'
        )

    plt.title('Proiezione PCA 2D delle lettere (feature morfologiche)')
    plt.xlabel(f'PC1 ({pc1_var:.1f}% varianza spiegata)')
    plt.ylabel(f'PC2 ({pc2_var:.1f}% varianza spiegata)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = PLOTS_DIR / 'feature_space_pca_2d.png'
    plt.savefig(out, dpi=150)
    plt.show()


def run_all_plots():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap()
    plot_pca_2d()
    print(f"Heatmap salvata in: {PLOTS_DIR / 'feature_correlation_heatmap.png'}")
    print(f"PCA salvata in: {PLOTS_DIR / 'feature_space_pca_2d.png'}")


if __name__ == '__main__':
    run_all_plots()