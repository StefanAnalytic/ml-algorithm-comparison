<div align="center">

# 🚲 Bike Sharing Demand Prediction
**Machine Learning Algorithm Comparison & GLM Analysis**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-1793D1.svg?style=for-the-badge&logo=xgboost&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-ff69b4.svg?style=for-the-badge)](#)

</div>

---

> **Executive Summary:** Dieses Projekt vergleicht 6 verschiedene Machine-Learning-Modelle – von simplen statistischen Baselines bis hin zu State-of-the-Art Tree-Boosting-Algorithmen –, um die Fahrrad-Nachfrage (Zähldaten) vorherzusagen.

## 🏆 Leaderboard

Tree-basierte Modelle dominieren diesen Datensatz klar. Sie erfassen komplexe, nicht-lineare Verhaltensmuster (wie exakte Rush-Hour-Zeiten an Werktagen) deutlich präziser als klassische verallgemeinerte lineare Modelle (GLMs).

| Rank | Model Architecture | Poisson Deviance 📉 | RMSE 🎯 |
| :---: | :--- | :---: | :---: |
| 🥇 | **Random Forest** | `47.59` | `122.71` |
| 🥈 | **LightGBM** *(Poisson)* | `49.53` | `124.79` |
| 🥉 | **XGBoost** *(Poisson)* | `49.75` | `124.92` |
| 4️⃣ | **Negative Binomial GLM** | `137.96` | `192.21` |
| 5️⃣ | **Linear Baseline** *(Ridge)* | `138.70` | `186.75` |
| 6️⃣ | **Poisson GLM** | `142.21` | `191.14` |

*💡 **Lesehilfe:** Ein niedrigerer Wert bedeutet eine bessere Vorhersagegenauigkeit. Die Boosting-Algorithmen wurden intern auf die Poisson-Verteilung optimiert.*

---

## 🚀 Quickstart

Bringe die gesamte Pipeline lokal zum Laufen:

### 1. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 2. Modell-Battle starten

```bash
python compare_models.py
```

📊 **Ergebnisse:**  
Die grafischen Auswertungen (Actual vs. Predicted Scatter-Plots und Zeitverläufe) werden nach dem Run automatisch im Ordner `results/plots/` generiert.
