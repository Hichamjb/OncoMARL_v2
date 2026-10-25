import pandas as pd
import ast
from demo import OncoMARLInferenceEngine

# 1. Charger le modèle et le dataset
engine = OncoMARLInferenceEngine("checkpoints/marl_model.pkl")
df = pd.read_csv("data/processed/test_patients.csv")

correct_count = 0
incorrect_count = 0

print("⏳ Évaluation en cours, veuillez patienter...")

# 2. Boucler sur chaque patient du dataset
for index, row in df.iterrows():
    # Convertir l'état en liste s'il est en format texte
    state_data = row['state']
    if isinstance(state_data, str):
        state = ast.literal_eval(state_data)
    else:
        state = state_data
        
    reference_action = int(row['action'])
    
    # 3. Obtenir la prédiction du modèle
    result = engine.predict(state)
    predicted_action = result['voted_action_id']
    
    # 4. Comparer
    if predicted_action == reference_action:
        correct_count += 1
    else:
        incorrect_count += 1

# 5. Afficher le résultat final
total = correct_count + incorrect_count
accuracy = (correct_count / total) * 100

print("\n" + "="*40)
print(f"✅ PRÉDICTIONS CORRECTES   : {correct_count}")
print(f"❌ PRÉDICTIONS INCORRECTES : {incorrect_count}")
print("="*40)
print(f"📊 TOTAL ÉVALUÉ            : {total} états")
print(f"🎯 PRÉCISION GLOBALE       : {accuracy:.2f} %")
print("="*40)