import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_stability(output_dir="checkpoints"):
    print("\n[Phase 7: Visualisation] Génération du graphique de l'entraînement...")

    # Les résultats d'entraînement sur les 5 seeds (récupérés de vos logs précédents)
    training_data = [
        {"seed": "Seed 10", "mean_final_reward": 0.5980, "mean_agreement": 86.83},
        {"seed": "Seed 20", "mean_final_reward": 0.6009, "mean_agreement": 89.33},
        {"seed": "Seed 30", "mean_final_reward": 0.6107, "mean_agreement": 87.83},
        {"seed": "Seed 40", "mean_final_reward": 0.5988, "mean_agreement": 88.00},
        {"seed": "Seed 50", "mean_final_reward": 0.6011, "mean_agreement": 87.83}
    ]
    
    df = pd.DataFrame(training_data)
    
    # Configuration du style
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Axe 1 : Graphique à barres pour la Récompense (Reward)
    color_reward = "royalblue"
    sns.barplot(data=df, x="seed", y="mean_final_reward", color=color_reward, alpha=0.8, ax=ax1)
    ax1.set_ylabel("Récompense Finale Moyenne (Reward)", color=color_reward, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Initialisation (Seeds)", fontsize=12)
    ax1.set_ylim(0.58, 0.62)  # Zoom sur les valeurs pour voir la faible variance
    ax1.tick_params(axis='y', labelcolor=color_reward)
    
    # Ajouter les valeurs exactes sur les barres
    for index, row in df.iterrows():
        ax1.text(index, row.mean_final_reward + 0.001, f'{row.mean_final_reward:.4f}', color='black', ha="center", fontsize=10)

    # Axe 2 : Ligne pour le Taux d'Accord (Agreement)
    ax2 = ax1.twinx()
    color_agree = "darkorange"
    # On utilise l'index numérique pour x dans sns.lineplot quand on superpose sur un barplot
    sns.lineplot(data=df, x=df.index, y="mean_agreement", color=color_agree, marker="o", linewidth=3, markersize=10, ax=ax2, label="Taux d'accord (%)")
    ax2.set_ylabel("Taux d'Accord des Agents (%)", color=color_agree, fontsize=12, fontweight="bold")
    ax2.set_ylim(80, 100)
    ax2.tick_params(axis='y', labelcolor=color_agree)
    ax2.grid(False) # Désactiver la grille secondaire pour plus de clarté
    
    # Légende et Titre
    plt.title("Stabilité de l'Entraînement Multi-Agent sur 5 Seeds", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    
    # Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "training_stability_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Graphique d'entraînement sauvegardé avec succès dans : {save_path}\n")

if __name__ == "__main__":
    plot_training_stability()