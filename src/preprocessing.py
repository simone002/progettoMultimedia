# Funzioni per binarizzazione e pulizia iniziale
import numpy as np
from skimage import io, color, filters, util

def load_and_binarize(image_path, invert=True):
    """Carica un'immagine, la converte in scala di grigi e applica Otsu."""
    img = io.imread(image_path)
    if len(img.shape) == 3: # Se è a colori
        img = color.rgb2gray(img)
    
    # Calcolo della soglia ottimale con Otsu
    thresh = filters.threshold_otsu(img)
    binary = img < thresh if invert else img > thresh
    
    return util.img_as_bool(binary)