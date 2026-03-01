import os
from pathlib import Path
from src.preprocessing import load_and_binarize
from src.morphology_logic import pruning, morphological_reconstruction
from skimage import morphology, io
import matplotlib.pyplot as plt

# 1. Configurazione percorsi
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / 'data' / 'manoscritti' / '0002-1.tif'
OUTPUT_DIR = BASE_DIR / 'output' / 'reconstructed'

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_pipeline():
    print(f"--- Inizio elaborazione su: {INPUT_FILE} ---")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")
    
    # STEP 1: Binarizzazione
    binary = load_and_binarize(str(INPUT_FILE))
    
    # STEP 2: Scheletrizzazione (Thinning)
    skeleton = morphology.skeletonize(binary)
    
    # STEP 3: Pruning (Eliminiamo il rumore dello scheletro)
    # Rimuoviamo i 'rametti' corti causati dalle imperfezioni della carta
    pruned_skeleton = pruning(skeleton, iterations=10)
    
    # STEP 4: Ricostruzione Geodesica
    # Usiamo lo scheletro pulito come 'seme' per riprendere la forma originale
    final_result = morphological_reconstruction(pruned_skeleton, binary)
    
    # 3. Visualizzazione e salvataggio
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(binary, cmap='gray')
    axes[0].set_title("Originale (Binarizzato)")
    
    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title("Scheletro Grezzo (con Rumore)")
    
    axes[2].imshow(final_result, cmap='gray')
    axes[2].set_title("Risultato Finale (Pulito)")
    
    plt.tight_layout()
    plt.show()
    
    # Salva il risultato
    io.imsave(str(OUTPUT_DIR / '0002-1_restored.png'), final_result.astype('uint8')*255)
    print("Elaborazione completata!")

if __name__ == "__main__":
    run_pipeline()