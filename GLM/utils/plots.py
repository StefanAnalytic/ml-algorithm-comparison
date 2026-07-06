import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_model_results(model_name, y_true, y_pred, output_dir="results/plots"):
    """Erstellt einen Dual-Plot für jedes Modell und speichert ihn als PNG."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Ein großes Bild mit zwei Grafiken nebeneinander
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Grafik 1: Zeitverlauf (Ausschnitt von 300 Stunden) ---
    subset = 300
    ax1.plot(y_true[:subset], label="Tatsächlich (Realität)", color="#2ecc71", alpha=0.7, linewidth=2)
    ax1.plot(y_pred[:subset], label=f"Vorhersage ({model_name})", color="#e74c3c", linestyle="--", linewidth=2)
    ax1.set_title(f"{model_name}: Zeitverlauf (300 Stunden)", fontsize=14, pad=10)
    ax1.set_xlabel("Zeit (Stunden im Test-Set)", fontsize=12)
    ax1.set_ylabel("Fahrrad-Ausleihen", fontsize=12)
    ax1.legend()
    
    # --- Grafik 2: Scatter-Plot (Vorhersagegenauigkeit) ---
    # Je näher die Punkte an der roten gestrichelten Linie sind, desto perfekter das Modell
    ax2.scatter(y_true, y_pred, alpha=0.3, color="#3498db", s=15)
    max_val = max(max(y_true), max(y_pred))
    ax2.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Perfekte Vorhersage")
    
    # Horizontale Linie bei 0 (um zu zeigen, ob Modelle unlogische negative Werte ausspucken)
    ax2.axhline(0, color='black', linewidth=1)
    
    ax2.set_title(f"{model_name}: Tatsächlich vs. Vorhersage", fontsize=14, pad=10)
    ax2.set_xlabel("Tatsächliche Ausleihen", fontsize=12)
    ax2.set_ylabel("Vorhergesagte Ausleihen", fontsize=12)
    ax2.legend()
    
    # Speichern
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{model_name}_plot.png")
    plt.savefig(filepath, dpi=300)
    plt.close()