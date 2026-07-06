# 🚲 ML Algorithm Comparison: Bike Sharing Demand

Dieses Projekt vergleicht 6 verschiedene Machine-Learning-Modelle (von simplen baselines bis zu Tree-basierten Boosting-Algorithmen), um die Fahrrad-Nachfrage (Zähldaten) vorherzusagen.

## 🏆 Leaderboard (Ergebnisse)
Tree-basierte Modelle dominieren diesen Datensatz klar, da sie komplexe, nicht-lineare Muster (wie Rush-Hour-Zeiten an Werktagen) besser erfassen können als klassische GLMs.

| Platz | Modell | Poisson Deviance | RMSE |
|-------|--------|------------------|------|
| 🥇 | **Random Forest** | 47.59 | 122.71 |
| 🥈 | **LightGBM (Poisson)** | 49.53 | 124.79 |
| 🥉 | **XGBoost (Poisson)** | 49.75 | 124.92 |
| 4. | **Negative Binomial GLM** | 137.96 | 192.21 |
| 5. | **Linear Baseline (Ridge)** | 138.70 | 186.75 |
| 6. | **Poisson GLM** | 142.21 | 191.14 |

## 🚀 Quickstart

1. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt

python compare_models.py

(Die grafischen Auswertungen (Actual vs. Predicted) landen automatisch im Ordner results/plots/)