# Progetto: Analisi morfologica di scrittura manoscritta

## Descrizione generale
Questo progetto nasce con l’obiettivo di analizzare immagini di lettere manoscritte attraverso tecniche di elaborazione morfologica.  
L’idea centrale è trasformare immagini rumorose in rappresentazioni strutturali più pulite, così da poter descrivere ogni lettera con caratteristiche numeriche semplici ma significative.

In questa fase il lavoro è focalizzato sull’**analisi**: estrazione e studio di feature morfologiche, confronto tra lettere e visualizzazione delle differenze nel feature space.

---

## Obiettivi del progetto
1. Pulire e normalizzare immagini manoscritte.
2. Ridurre il rumore mantenendo la forma utile dei caratteri.
3. Estrarre descrittori interpretabili della geometria delle lettere.
4. Confrontare statisticamente le lettere dell’alfabeto.
5. Preparare una base solida per la futura classificazione automatica.

---

## Dataset e organizzazione dati
Il progetto utilizza due viste principali del dataset:

- **Pagine complete manoscritte** (cartella `data/manoscritti`), utili per testare la pipeline di pulizia su immagini reali complesse.
- **Lettere segmentate per soggetto** (cartella `data/lettere/<soggetto>/<lettera>`), utili per l’analisi quantitativa per classe.
**Importante**: l'analisi è stata estesa a **tutti i 20 soggetti disponibili**, non solo uno. Questo garantisce:
- generalizzazione del modello (non si limita alla calligrafia di una persona);
- cattura della variabilità inter-soggetto;
- possibilità di fare split train/test **per soggetto** (no data leakage);
- dataset più robusto per futura classificazione.
La struttura è pensata per supportare sia test qualitativi (immagine singola) sia analisi statistica su più esempi.

---

## Architettura logica
Il progetto è organizzato in moduli separati, ognuno con una responsabilità chiara:

- [src/preprocessing.py](src/preprocessing.py): caricamento immagine, conversione in scala di grigi, binarizzazione con soglia di Otsu.
- [src/morphology_logic.py](src/morphology_logic.py): funzioni morfologiche principali (endpoints, junctions, pruning, ricostruzione, hole counting).
- [main.py](main.py): pipeline end-to-end su una pagina manoscritta.
- [analyze_letter.py](analyze_letter.py): analisi statistica delle feature per lettera.
- [src/visualization.py](src/visualization.py): visualizzazione della distribuzione delle lettere nel piano delle feature.

Questa separazione rende il progetto facilmente estendibile e più semplice da mantenere.

---

## Flusso di elaborazione
Il flusso implementato è il seguente:

1. **Caricamento immagine** e verifica del path.
2. **Binarizzazione** con Otsu per separare tratto e sfondo.
3. **Scheletrizzazione** per ridurre i tratti alla loro struttura essenziale.
4. **Pruning iterativo** per rimuovere ramificazioni spurie dovute al rumore.
5. **Ricostruzione morfologica** per recuperare la forma coerente del carattere.
6. **Estrazione feature** morfologiche.
7. **Aggregazione statistica** per lettera e visualizzazione finale.

---

## Feature analizzate
Le feature attualmente usate sono:

- **Punte (endpoints)**: quante terminazioni di tratto sono presenti.
- **Incroci (junctions)**: punti di biforcazione/intersezione.
- **Buchi (holes)**: regioni interne chiuse della lettera.
- **Componenti connesse**: regioni separate del tratto.
- **Aspect ratio**: rapporto larghezza/altezza della bounding box della componente principale.
- **Densità pixel**: frazione di pixel attivi rispetto all’area dell’immagine.

In aggiunta, sono state introdotte feature morfologiche avanzate per aumentare la capacità discriminativa:

- **Euler number** (`euler_number`): differenza tra componenti e buchi, sintetizza topologia globale.
- **Hole area ratio** (`hole_area_ratio`): quota di area occupata da cavità interne.
- **Average stroke width** (`avg_stroke_width`): stima dello spessore medio tramite distance transform.
- **Skeleton length** (`skeleton_length`): lunghezza totale del tratto scheletrizzato.
- **Endpoint/Junction norm** (`endpoint_norm`, `junction_norm`): endpoint/incroci normalizzati per lunghezza scheletro.
- **Skeleton density** (`skeleton_density`): densità dello scheletro rispetto all’immagine.
- **Pruning removed ratio** (`pruning_removed_ratio`): frazione di pixel rimossi dal pruning.

Ulteriore estensione (ultima iterazione) con descrittori di forma e invarianti:

- **Global shape**: `solidity`, `extent`, `eccentricity`, `orientation`, `major_axis_length`, `minor_axis_length`, `axis_ratio`.
- **Skeleton branches**: `n_branches`, `branch_length_mean`, `branch_length_max`, `branch_length_std`.
- **Hu moments**: `hu_1` ... `hu_7` (invarianti a rotazione/scala/traslazione, in forma log-compressa).
- **Zoning density (3x3)**: `zone_r*_c*_density` per catturare la distribuzione spaziale locale del tratto.
- **Projection profiles**: statistiche dei profili orizzontali/verticali (`row_proj_*`, `col_proj_*`) per rappresentare la forma lungo assi principali.

Estensione mirata alle classi più ambigue (ultima iterazione):

- **Spatial balance**: `top_half_density`, `bottom_half_density`, `left_half_density`, `right_half_density`, `top_bottom_ratio`, `left_right_ratio`.
- **Transition features**: `row_trans_*`, `col_trans_*` (numero/statistiche di transizioni 0↔1 su righe/colonne).
- **Diagonal profile**: `main_diag_density`, `anti_diag_density`, `diag_density_diff`.
- **Endpoint position**: `endpoint_*_mean_norm`, `endpoint_*_std_norm` (posizione media/dispersione delle terminazioni).

Queste misure permettono una descrizione compatta, interpretabile e più robusta delle forme.

---

## Stato attuale dell’implementazione
Attualmente il progetto consente di:

- eseguire una pipeline completa su immagine manoscritta,
- calcolare statistiche medie per lettera,
- salvare automaticamente i risultati in CSV,
- analizzare automaticamente tutte le lettere disponibili nella cartella del soggetto,
- visualizzare la separazione tra lettere con grafici analitici mirati (heatmap per gruppi + PCA 2D).

Sono già stati risolti i principali problemi tecnici incontrati (path, estensioni, dipendenze TIFF, import mancanti), rendendo l’ambiente stabile per il lavoro di analisi.

---

## Tecnologie utilizzate
- Python
- NumPy
- scikit-image
- SciPy
- Pandas
- Matplotlib
- OpenCV (presente tra dipendenze)
- imagecodecs (per TIFF compressi)

Dipendenze centralizzate in [requirements.txt](requirements.txt).

---

## Output prodotti dalla fase di analisi
L’analisi genera automaticamente:

- [output/analysis/features_samples.csv](output/analysis/features_samples.csv): feature a livello campione (include colonna `soggetto`).
- [output/analysis/features_summary.csv](output/analysis/features_summary.csv): medie per lettera (tutti i soggetti aggregati).
- [output/analysis/features_summary_per_subject.csv](output/analysis/features_summary_per_subject.csv): medie per combinazione soggetto-lettera (utile per analisi di variabilità).
- Grafici in [output/analysis/plots](output/analysis/plots):
	- `feature_correlation_heatmap.png`
	- `feature_space_pca_2d.png`

---

## Cosa indica ogni plot

### 1) Heatmap di correlazione delle feature (per gruppi)
Visualizza la correlazione media assoluta tra gruppi di feature (base, topology, shape, Hu, zoning, projection).

Interpretazione:
- valori alti: gruppi fortemente ridondanti/informativamente vicini;
- valori bassi: gruppi complementari.

Serve per capire ridondanze informative tra blocchi di descrittori e guidare la selezione feature.

### 2) PCA 2D
Proietta tutte le feature in due componenti principali (`PC1`, `PC2`) mantenendo la massima varianza possibile.

Interpretazione:
- permette una vista globale della separabilità multi-feature;
- la percentuale in asse (`PC1`, `PC2`) indica quanta informazione è mantenuta;
- cluster separati in PCA suggeriscono potenziale buona classificabilità futura.

---

## Metodologia sperimentale
La metodologia adottata è stata progettata per rispettare gli obiettivi del progetto: analizzare la struttura topologica dei caratteri manoscritti e restaurare i tratti utili in presenza di rumore.

### Fase 1 — Preprocessing
Ogni immagine viene caricata, convertita in scala di grigi (se necessario) e binarizzata con soglia di Otsu. Questo passaggio produce una rappresentazione booleana uniforme, adatta agli operatori morfologici successivi.

### Fase 2 — Analisi topologica con Hit-or-Miss
Sul risultato scheletrizzato vengono applicati kernel dedicati per riconoscere configurazioni locali specifiche:
- endpoint (terminazioni),
- junction (biforcazioni/incroci).

Questa fase consente di trasformare la forma del carattere in descrittori strutturali interpretabili.

### Fase 3 — Skeletonization e Pruning
La skeletonization riduce il tratto alla sua “spina dorsale” topologica. Successivamente, il pruning elimina iterativamente ramificazioni corte e spurie, tipicamente introdotte dal rumore di acquisizione o dalla texture della carta.

### Fase 4 — Ricostruzione morfologica
Per evitare perdita di informazione utile, il marker pulito ottenuto dal pruning viene usato in ricostruzione geodesica rispetto alla maschera binaria originale. In questo modo si recuperano componenti connesse coerenti col tratto reale, mantenendo il filtraggio del rumore.

### Fase 5 — Estrazione feature e aggregazione
Per ciascun campione vengono estratte feature topologiche, geometriche e invarianti (`punte`, `incroci`, `buchi`, `componenti`, `aspect_ratio`, `densita_pixel`, blocchi `global_shape`, `skeleton_branches`, `hu_moments`, `zoning_density`, `projection_profiles`, `spatial_balance`, `transitions`, `diagonal_profile`, `endpoint_position`).
I risultati sono poi aggregati:
- per lettera,
- per soggetto-lettera,
così da studiare sia il comportamento medio sia la variabilità tra scrittori.

---

## Validazione sperimentale
La validazione è stata impostata su due livelli complementari.

### 1) Validazione algoritmica su caratteri isolati
Su dataset di lettere segmentate è stata verificata la capacità del sistema di:
- identificare pattern topologici plausibili (endpoint, junction),
- distinguere lettere con diversa complessità morfologica,
- mantenere coerenza delle statistiche medie per classe.

L’analisi è stata estesa da un singolo soggetto a tutti i 20 soggetti disponibili, riducendo il rischio di bias legato alla singola calligrafia.

### 2) Validazione qualitativa su manoscritti reali
Su pagine manoscritte complete è stata valutata la pipeline di restauro (binarizzazione → scheletro → pruning → ricostruzione) con confronto visivo tra output intermedi e risultato finale.

Obiettivo della verifica qualitativa: confermare che il sistema rimuova rumore mantenendo la connettività principale dei caratteri.

### Evidenze ottenute
- Feature estratte con comportamento coerente alla morfologia attesa delle lettere.
- Variabilità inter-soggetto osservabile e quantificabile.
- Separabilità parziale delle classi nei grafici (scatter/PCA), con cluster ben distinti e aree di sovrapposizione realistiche.

### Limiti attuali
- Validazione ancora principalmente descrittiva/interpretativa.
- Le confusion matrix mostrano ancora sovrapposizioni tra lettere graficamente simili.

---

## Stato rispetto alla proposta iniziale
Rispetto all’idea progettuale iniziale, i tre blocchi richiesti risultano implementati:

1. **Hit-or-Miss Transform**: implementato per estrarre endpoint e junction.
2. **Skeletonization + Pruning**: implementati e usati sia su lettere isolate sia su manoscritti reali.
3. **Ricostruzione morfologica**: implementata e integrata nella pipeline di restauro.

Sono inoltre già presenti dataset, script di analisi e visualizzazioni utili alla discussione sperimentale.

---

## Prossimi passi (chiusura progetto)
Per completare formalmente la parte sperimentale, i passi consigliati sono:

1. definire uno split train/test **per soggetto**,
2. addestrare un classificatore baseline sulle feature estratte,
3. riportare metriche quantitative (accuracy, macro-F1, confusion matrix),
4. confrontare l’effetto del pruning/ricostruzione sulle performance.

Con questi step, il progetto passa da analisi morfologica avanzata a validazione quantitativa completa.

---

## Conclusioni
Il progetto ha raggiunto l’obiettivo principale di applicare tecniche avanzate di Morfologia Matematica all’analisi e al restauro di caratteri manoscritti, andando oltre gli operatori base di erosione e dilatazione.

In particolare, sono stati implementati e validati:
- la trasformata Hit-or-Miss per l’estrazione di feature topologiche (endpoint e junction),
- la pipeline skeletonization + pruning per la pulizia strutturale dei tratti,
- la ricostruzione morfologica per il recupero della connettività rilevante.

L’uso combinato di dataset controllato (lettere isolate) e dati reali degradati (manoscritti completi) ha permesso di verificare la robustezza della soluzione sia in condizioni semplificate sia in scenari realistici.

I risultati ottenuti mostrano che le feature estratte sono informative, interpretabili e adatte a descrivere differenze tra classi, mantenendo al tempo stesso la variabilità naturale tra soggetti.

Nel complesso, il lavoro fornisce una base metodologica solida per il passaggio a una fase di classificazione automatica supervisionata.

---

## Sviluppi futuri
Gli sviluppi più naturali del progetto sono:

1. introduzione di una fase di classificazione con split train/test per soggetto;
2. confronto tra modelli baseline (SVM, Random Forest, kNN);
3. misurazione quantitativa dell’impatto di pruning e ricostruzione sulle performance;
4. estensione delle feature con descrittori statistici e invarianti di forma;
5. analisi degli errori con confusion matrix e studio delle coppie di lettere più ambigue.

Questi passi completerebbero la transizione da una pipeline di analisi morfologica avanzata a un sistema completo di riconoscimento.

---

## Classificazione advanced non-CNN (split per soggetto)
La validazione quantitativa è ora concentrata su [train_noncnn_advanced.py](train_noncnn_advanced.py), usando solo modelli non-CNN e rimuovendo esplicitamente `stacking`.

### Setup sperimentale
- Input: [output/analysis/features_samples.csv](output/analysis/features_samples.csv)
- Feature: tutte le feature numeriche morfologiche (base + avanzate + feature ingegnerizzate)
- Target: `lettera`
- Split: per soggetto (15 soggetti train, 5 soggetti test)
- Campioni: 14521 totali (11610 train, 2911 test)
- Modelli: `svm_rbf`, `extra_trees`, `xgboost`

### Risultati principali aggiornati
- `xgboost`: Accuracy = **0.7183**, Macro-F1 = **0.7119** (migliore)
- `svm_rbf`: Accuracy = 0.7042, Macro-F1 = 0.7020
- `extra_trees`: Accuracy = 0.7032, Macro-F1 = 0.6871

L’aumento della Macro-F1 rispetto alle iterazioni iniziali conferma che il miglioramento deriva soprattutto dal **feature engineering morfologico**, inclusa la nuova informazione spaziale/strutturale introdotta dai blocchi mirati.

### Cross-validation per soggetto (5 fold)
Per una stima più robusta della generalizzazione tra soggetti è stata eseguita anche CV subject-wise a 5 fold.

- `xgboost`: Accuracy = **0.7676 ± 0.0246**, Macro-F1 = **0.7236 ± 0.0269** (migliore)
- `svm_rbf`: Accuracy = 0.7372 ± 0.0427, Macro-F1 = 0.7039 ± 0.0351
- `extra_trees`: Accuracy = 0.7472 ± 0.0265, Macro-F1 = 0.6898 ± 0.0306

Le confusioni più frequenti nel best model (`xgboost`) restano coerenti con l’analisi qualitativa: `u→n`, `f→t`, `u→v`, `e→a`, `n→u`.

### Focus su lettere più deboli (dopo miglioramento mirato)
Con le feature targeted, le classi più difficili migliorano in modo sensibile. Le 5 F1 più basse attuali sono:
- `n`: 0.541
- `u`: 0.559
- `z`: 0.590
- `x`: 0.615
- `k`: 0.625

Per confronto qualitativo, nelle iterazioni precedenti alcune di queste classi erano sensibilmente più basse (es. `z` ~0.35).

### Artefatti prodotti
- Holdout metrics JSON: [output/classification_advanced/organized/01_holdout/advanced_metrics.json](output/classification_advanced/organized/01_holdout/advanced_metrics.json)
- Holdout summary CSV: [output/classification_advanced/organized/01_holdout/advanced_summary.csv](output/classification_advanced/organized/01_holdout/advanced_summary.csv)
- CV summary CSV: [output/classification_advanced/organized/02_cv_subjectwise/cv_subjectwise_summary.csv](output/classification_advanced/organized/02_cv_subjectwise/cv_subjectwise_summary.csv)
- Top confusion pairs (CV): [output/classification_advanced/organized/02_cv_subjectwise/cv_top_confusions.csv](output/classification_advanced/organized/02_cv_subjectwise/cv_top_confusions.csv)
- Confusion matrix:
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_xgboost.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_xgboost.png)
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_svm_rbf.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_svm_rbf.png)
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_extra_trees.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/raw/confusion_matrix_extra_trees.png)
- Confusion matrix normalizzate:
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_xgboost_normalized.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_xgboost_normalized.png)
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_svm_rbf_normalized.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_svm_rbf_normalized.png)
	- [output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_extra_trees_normalized.png](output/classification_advanced/organized/04_diagnostics/confusion_matrices/normalized/confusion_matrix_extra_trees_normalized.png)
- F1 per lettera (best model):
	- [output/classification_advanced/organized/04_diagnostics/f1_by_letter/f1_by_letter_xgboost.png](output/classification_advanced/organized/04_diagnostics/f1_by_letter/f1_by_letter_xgboost.png)
	- [output/classification_advanced/organized/04_diagnostics/f1_by_letter/f1_by_letter_sorted_after_targeted.csv](output/classification_advanced/organized/04_diagnostics/f1_by_letter/f1_by_letter_sorted_after_targeted.csv)
- Importanza feature (XGBoost):
	- [output/classification_advanced/organized/04_diagnostics/feature_importance/xgboost_feature_importance_top20.png](output/classification_advanced/organized/04_diagnostics/feature_importance/xgboost_feature_importance_top20.png)

## Ablation study morfologico
Per misurare il contributo reale dei descrittori è stato eseguito [ablation_noncnn_morphology.py](ablation_noncnn_morphology.py) con modello `xgboost` fisso e rimozione di gruppi di feature.

Nota: i risultati sotto sono relativi al run di ablation precedente all'aggiunta dei blocchi `zoning_density` e `projection_profiles`.

### Risultati ablation (delta Macro-F1 vs tutte le feature)
- Rimozione `global_shape` (`solidity`, `extent`, `eccentricity`, `orientation`, `major_axis_length`, `minor_axis_length`, `axis_ratio`): **-0.0497**
- Rimozione `hu_moments` (`hu_1` ... `hu_7`): **-0.0313**
- Rimozione `stroke_geometry` (`avg_stroke_width`, `skeleton_length`, `skeleton_density`): **-0.0126**
- Rimozione `skeleton_branches` (`n_branches`, `branch_length_mean`, `branch_length_max`, `branch_length_std`): **-0.0069**
- Rimozione `advanced_topology` (`endpoint_norm`, `junction_norm`, `euler_number`): **+0.0001**
- Rimozione `hole_structure` (`hole_area_ratio`): **+0.0016**
- Rimozione `pruning_signal` (`pruning_removed_ratio`): **+0.0103**

### Interpretazione ablation
- **Contributo principale**: `global_shape` e `hu_moments` sono i gruppi più informativi nella configurazione attuale.
- **Contributo secondario**: `stroke_geometry` e `skeleton_branches` aggiungono informazione utile ma meno dominante.
- **Possibile ridondanza/rumore**: i delta positivi su `pruning_signal`, `hole_structure` e `advanced_topology` indicano che, in questa configurazione, questi gruppi non stanno aiutando il modello XGBoost e possono essere candidati a feature selection.

### Apporto complessivo delle nuove feature
L’introduzione progressiva delle feature avanzate e poi delle feature mirate porta il sistema da una macro-F1 nell’ordine di ~0.33 alle versioni intermedie (~0.48, ~0.64), fino all’assetto attuale ~0.70, con netto miglioramento della generalizzazione su soggetti non visti.

### Artefatti ablation
- Summary CSV: [output/classification_advanced/organized/03_ablation/ablation_summary.csv](output/classification_advanced/organized/03_ablation/ablation_summary.csv)
- Report JSON: [output/classification_advanced/organized/03_ablation/ablation_report.json](output/classification_advanced/organized/03_ablation/ablation_report.json)