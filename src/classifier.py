def predict_letter(new_features, training_data):
    """
    new_features: lista [punte, incroci, buchi] della nuova immagine
    training_data: la tabella delle medie che hai appena postato
    """
    best_match = None
    min_distance = float('inf')
    
    for _, row in training_data.iterrows():
        # Calcoliamo la distanza euclidea tra le feature della nuova lettera e le medie
        dist = np.sqrt((new_features[0] - row['punte'])**2 + 
                       (new_features[1] - row['incroci'])**2 + 
                       (new_features[2] - row['buchi'])**2)
        
        if dist < min_distance:
            min_distance = dist
            best_match = row['lettera']
            
    return best_match