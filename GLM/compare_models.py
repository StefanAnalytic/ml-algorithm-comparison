import os
import pandas as pd
from utils.preprocessing import DataProcessor
from utils.metrics import evaluate_model
from utils.plots import plot_model_results

# Alle Modelle importieren
from models.linear_model import LinearBaseline
from models.poisson_glm import PoissonGLM
from models.negbin_glm import NegativeBinomialGLM
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel

def main():
    print("🚲 Starte das ultimative Model-Battle...\n")
    
    # 1. Daten laden
    print("1️⃣ Lade und verarbeite Daten...")
    processor = DataProcessor(data_path="data/train.csv")
    X_train, X_test, y_train, y_test = processor.get_train_test_split()
    print(f"✅ Daten bereit! Trainingsset: {X_train.shape[0]} Reihen, Testset: {X_test.shape[0]} Reihen.\n")
    
    # 2. Modelle initialisieren
    models = {
        "1_Linear_Baseline": LinearBaseline(),
        "2_Poisson_GLM": PoissonGLM(),
        "3_Negative_Binomial_GLM": NegativeBinomialGLM(alpha=1.0),
        "4_Random_Forest": RandomForestModel(),
        "5_XGBoost_Poisson": XGBoostModel(),
        "6_LightGBM_Poisson": LightGBMModel()
    }
    
    # Verzeichnis für CSV erstellen
    os.makedirs("results", exist_ok=True)
    all_metrics = []
    
    # 3. Training & Evaluation Schleife
    print("2️⃣ Starte Training und Visualisierung...")
    for name, model in models.items():
        print(f"   ⏳ Trainiere {name}...")
        
        # Trainieren
        model.fit(X_train, y_train)
        
        # Vorhersagen
        predictions = model.predict(X_test)
        
        # Evaluieren
        metrics = evaluate_model(name, y_test, predictions)
        all_metrics.append(metrics)
        
        # Grafik generieren und speichern
        plot_model_results(name, y_test, predictions, output_dir="results/plots")
        print(f"   📈 Plot gespeichert: results/plots/{name}_plot.png")
        
    # 4. Ergebnisse auswerten
    print("\n3️⃣ Auswertung abgeschlossen! Hier ist das Leaderboard:")
    results_df = pd.DataFrame(all_metrics)
    
    # Sortieren nach Poisson Deviance (kleinster Wert = bestes Modell)
    results_df = results_df.sort_values(by="Poisson_Dev")
    
    print("\n" + "="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")
    
    # CSV speichern
    results_df.to_csv("results/metrics.csv", index=False)
    print("💾 Ergebnisse wurden in 'results/metrics.csv' gespeichert.")

if __name__ == "__main__":
    main()