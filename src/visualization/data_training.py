import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Définition des dossiers et fichiers
output_dir = "checkpoints"
json_filepath = os.path.join(output_dir, "train_summary_full.json")
image_filepath = os.path.join(output_dir, "training_metrics.png")

# Création du dossier si nécessaire
os.makedirs(output_dir, exist_ok=True)

# 2. Lecture des données depuis le fichier JSON
with open(json_filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraction des métriques pour la première Seed (Seed 10)
seed_data = data[0]
reward_history = seed_data["reward_history"]
agreement_history = seed_data["agreement_history"]

steps_reward = np.arange(1, len(reward_history) + 1)
steps_agreement = np.arange(1, len(agreement_history) + 1)

# Lissage via moyenne mobile (fenêtre de 20 pas)
window = 20
reward_ma = (
    pd.Series(reward_history).rolling(window=window, min_periods=1).mean()
)
agreement_ma = (
    pd.Series(agreement_history).rolling(window=window, min_periods=1).mean()
)

# 3. Création de la figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Graphique 1 : Reward
axes[0].plot(
    steps_reward,
    reward_history,
    alpha=0.25,
    color="#1f77b4",
    label="Reward (Brut)",
)
axes[0].plot(
    steps_reward,
    reward_ma,
    color="#1f77b4",
    linewidth=2,
    label=f"Moyenne mobile ({window})",
)
axes[0].axhline(
    y=seed_data["mean_final_reward"],
    color="red",
    linestyle="--",
    label=f'Moyenne finale ({seed_data["mean_final_reward"]:.3f})',
)
axes[0].set_title(f'Seed {seed_data["seed"]}: Reward History')
axes[0].set_xlabel("Étape")
axes[0].set_ylabel("Reward")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(loc="lower right")

# Graphique 2 : Agreement
axes[1].plot(
    steps_agreement,
    agreement_history,
    alpha=0.25,
    color="#ff7f0e",
    label="Agreement (Brut)",
)
axes[1].plot(
    steps_agreement,
    agreement_ma,
    color="#ff7f0e",
    linewidth=2,
    label=f"Moyenne mobile ({window})",
)
axes[1].axhline(
    y=seed_data["mean_agreement"],
    color="green",
    linestyle="--",
    label=f'Moyenne finale ({seed_data["mean_agreement"]:.3f})',
)
axes[1].set_title(f'Seed {seed_data["seed"]}: Agreement History')
axes[1].set_xlabel("Étape")
axes[1].set_ylabel("Agreement")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(loc="lower right")

plt.tight_layout()

# 4. Sauvegarde de l'image dans le dossier output_dir="checkpoints"
plt.savefig(image_filepath)
plt.close()

print(f"Graphique sauvegardé dans : {image_filepath}")