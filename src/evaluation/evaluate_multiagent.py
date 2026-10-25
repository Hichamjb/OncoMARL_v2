import ast
import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ajouter le chemin racine pour importer demo.py
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
from demo import OncoMARLInferenceEngine


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON personnalisé pour convertir en sécurité les types NumPy/Pandas."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def run_evaluation(
    data_path="data/processed/test_patients.csv",
    model_path="checkpoints/marl_model.pkl",
    output_path="checkpoints/test_evaluation_summary.json",
):
    print(
        f"\n[Phase 8: Evaluation] Chargement des données de test depuis : {data_path}"
    )

    if not os.path.exists(data_path):
        print(f"[Erreur] Le fichier de test est introuvable à {data_path}")
        return

    engine = OncoMARLInferenceEngine(model_path)
    if not engine.is_loaded:
        print(
            "[Erreur] Le modèle n'est pas trouvé. Veuillez d'abord entraîner le modèle."
        )
        return

    df = pd.read_csv(data_path)
    total_states = len(df)
    agreements = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    per_example_results = []

    print(
        f"[Evaluation] Lancement de l'inférence sur {total_states} états de patients...\n"
    )

    for idx, row in tqdm(
        df.iterrows(), total=total_states, desc="Évaluation en cours"
    ):
        state_data = row["state"]
        if isinstance(state_data, str):
            state_vector = ast.literal_eval(state_data)
        else:
            state_vector = state_data

        # Exécution de la prédiction
        result = engine.predict(state_vector)

        voted_action_id = int(result["voted_action_id"])
        is_unanimous = bool(result["is_unanimous"])
        action_name = engine.action_names.get(
            voted_action_id, f"Action_{voted_action_id}"
        )

        if is_unanimous:
            agreements += 1

        action_counts[voted_action_id] = (
            action_counts.get(voted_action_id, 0) + 1
        )

        # Extraction de l'identifiant patient s'il existe dans le CSV
        patient_id = row.get(
            "patient_id", row.get("id", f"example_{idx}")
        )

        # Construction du résultat individuel
        example_entry = {
            "example_index": idx,
            "patient_id": patient_id,
            "voted_action_id": voted_action_id,
            "voted_action_name": action_name,
            "is_unanimous": is_unanimous,
            "prediction_details": result,
        }
        per_example_results.append(example_entry)

    agreement_rate = agreements / total_states

    # Préparation du dictionnaire global incluant les détails par exemple
    report_data = {
        "dataset_evaluated": data_path,
        "total_states_evaluated": total_states,
        "multi_agent_agreement_rate": round(agreement_rate, 4),
        "action_selection_distribution": {},
        "per_example_results": per_example_results,
    }

    # Formatage de la distribution des actions
    print("\n" + "=" * 50)
    print("             RÉSULTATS DE L'ÉVALUATION            ")
    print("=" * 50)
    print(f" Nombre total d'états évalués : {total_states}")
    print(f" Taux d'accord (Agreement Rate) : {agreement_rate:.2%}")
    print("\n Distribution des actions :")

    for action_id, count in action_counts.items():
        action_name = engine.action_names.get(
            action_id, f"Action_{action_id}"
        )
        percentage = count / total_states

        report_data["action_selection_distribution"][action_name] = {
            "action_id": action_id,
            "count": count,
            "percentage": round(percentage, 4),
        }

        print(
            f"  • {action_name} (ID {action_id}): {count} états ({percentage:.2%})"
        )
    print("=" * 50 + "\n")

    # Sauvegarde sécurisée du rapport complet dans le fichier JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, cls=NumpyEncoder)

    print(f"✅ Résultats sauvegardés avec succès dans : {output_path}")


if __name__ == "__main__":
    run_evaluation()