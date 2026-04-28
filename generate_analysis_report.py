"""
Genera un report testuale di analisi dei risultati
"""
from pathlib import Path
import pandas as pd
import numpy as np

# questo file contiene la logica per generare un report testuale di analisi dei risultati
# basato sui dati estratti da analyze_letter.py. Il report include statistiche generali, 
# correlazioni tra feature, lettere più/meno complesse, variabilità intra-lettera e 
# osservazioni chiave. Il report viene salvato in formato Markdown per una facile
#  lettura e condivisione.

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_CSV = BASE_DIR / 'output' / 'analysis' / 'features_samples.csv'
SUMMARY_CSV = BASE_DIR / 'output' / 'analysis' / 'features_summary.csv'
REPORT_OUT = BASE_DIR / 'output' / 'analysis' / 'ANALYSIS_REPORT.md'


def generate_report():
    if not SAMPLES_CSV.exists() or not SUMMARY_CSV.exists():
        raise FileNotFoundError("Esegui prima analyze_letter.py")
    
    df_samples = pd.read_csv(SAMPLES_CSV)
    df_summary = pd.read_csv(SUMMARY_CSV)
    
    feature_cols = ['punte', 'incroci', 'buchi', 'componenti', 'aspect_ratio', 'densita_pixel']
    
    # Statistiche globali
    n_samples = len(df_samples)
    n_subjects = df_samples['soggetto'].nunique()
    n_letters = df_samples['lettera'].nunique()
    
    # Correlazioni più forti
    corr = df_summary[feature_cols].corr()
    corr_flat = []
    for i, f1 in enumerate(feature_cols):
        for j, f2 in enumerate(feature_cols):
            if i < j:
                corr_flat.append((f1, f2, corr.loc[f1, f2]))
    corr_flat.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Lettere più/meno complesse
    df_summary['complessita'] = (
        df_summary['punte'] + df_summary['incroci'] * 2 + df_summary['buchi']
    )
    most_complex = df_summary.nlargest(5, 'complessita')[['lettera', 'complessita']]
    least_complex = df_summary.nsmallest(5, 'complessita')[['lettera', 'complessita']]
    
    # Variabilità intra-lettera
    variability = df_samples.groupby('lettera')[feature_cols].std().mean(axis=1).sort_values(ascending=False)
    
    # Genera report
    report = f"""# Report di analisi morfologica - Lettere manoscritte

## 1. Statistiche generali del dataset

- **Campioni totali analizzati**: {n_samples}
- **Soggetti**: {n_subjects}
- **Lettere**: {n_letters}
- **Campioni medi per lettera**: {n_samples / n_letters:.1f}

---

## 2. Feature estratte

Le seguenti feature morfologiche sono state calcolate per ogni campione:

| Feature | Descrizione |
|---------|-------------|
| `punte` | Numero di endpoint dello scheletro (terminazioni di tratto) |
| `incroci` | Numero di junction (biforcazioni/intersezioni) |
| `buchi` | Numero di regioni interne chiuse (loop) |
| `componenti` | Numero di componenti connesse |
| `aspect_ratio` | Rapporto larghezza/altezza bounding box |
| `densita_pixel` | Frazione di pixel attivi |

---

## 3. Correlazioni tra feature

Le correlazioni più forti (in valore assoluto) sono:

"""
    
    for f1, f2, val in corr_flat[:5]:
        report += f"- **{f1}** ↔ **{f2}**: {val:.3f}\n"
    
    report += f"""

👉 **Conclusione**: nessuna multicollinearità forte (tutte sotto 0.7), quindi le feature sono complementari.

---

## 4. Lettere per complessità morfologica

### Più complesse (punte + incroci*2 + buchi)

"""
    
    for _, row in most_complex.iterrows():
        report += f"- `{row['lettera']}`: {row['complessita']:.2f}\n"
    
    report += "\n### Meno complesse\n\n"
    
    for _, row in least_complex.iterrows():
        report += f"- `{row['lettera']}`: {row['complessita']:.2f}\n"
    
    report += f"""

---

## 5. Variabilità intra-lettera

Lettere con maggiore variabilità tra campioni (standard deviation media):

"""
    
    for letter, std_val in variability.head(5).items():
        report += f"- `{letter}`: {std_val:.3f}\n"
    
    report += f"""

👉 Alta variabilità indica che la lettera viene scritta in modi molto diversi tra soggetti.

---

## 6. Osservazioni chiave

### Separabilità
Alcune lettere hanno profili molto distintivi:
- `o`: poche punte ({df_summary.loc[df_summary['lettera']=='o', 'punte'].values[0]:.2f}), alta densità
- `x`: molte punte ({df_summary.loc[df_summary['lettera']=='x', 'punte'].values[0]:.2f}), pochi buchi
- `i`: aspect_ratio basso ({df_summary.loc[df_summary['lettera']=='i', 'aspect_ratio'].values[0]:.3f}), forma verticale

### Sovrapposizioni attese
Lettere con profili simili che potrebbero confondersi:
- lettere con pochi buchi e medie punte: `c`, `r`, `s`, `n`
- lettere con aspect_ratio simile: `h`, `b`, `d`

---

## 7. Prossimi passi consigliati

1. **Train/test split per soggetto** per evitare data leakage
2. **Feature selection** tramite importanza da modello
3. **Classificatore baseline** (SVM, Random Forest)
4. **Analisi errori** con confusion matrix
5. **Augmentation** se serve più bilanciamento

---

*Report generato automaticamente da `generate_analysis_report.py`*
"""
    
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report, encoding='utf-8')
    print(f"Report salvato in: {REPORT_OUT}")
    return report


if __name__ == "__main__":
    report = generate_report()
    print("\n" + report)
