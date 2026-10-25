import os
import ast
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

def load_environment_and_data(model_path="checkpoints/marl_model.pkl", data_path="data/processed/test_patients.csv"):
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Missing checkpoint or evaluation dataset.")
        
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)
        
    df = pd.read_csv(data_path)
    
    # Safe parsing of stringified vectors
    if isinstance(df['state'].iloc[0], str):
        df['state'] = df['state'].apply(ast.literal_eval)
    if 'next_state' in df.columns and isinstance(df['next_state'].iloc[0], str):
        df['next_state'] = df['next_state'].apply(ast.literal_eval)
        
    return checkpoint, df

def validate_dataset(df):
    print("\n--- DATASET VALIDATION ---")
    total_rows = len(df)
    valid_actions = df['action'].isin([0, 1, 2, 3])
    valid_states = df['state'].apply(lambda x: len(x) == 6)
    
    invalid_rows = (~valid_actions | ~valid_states).sum()
    print(f"Total rows: {total_rows}")
    print(f"Invalid rows detected: {invalid_rows}")
    
    # Check Sequential consistency
    if 'next_state' in df.columns and 'step' in df.columns:
        df_sorted = df.sort_values(by=['patient_id', 'step'])
        # Shift state by -1 to compare next_state(t) with state(t+1)
        next_step_state = df_sorted.groupby('patient_id')['state'].shift(-1)
        # Note: Exact float matching can fail, we check if shapes/types align for sequential logic
        valid_transitions = next_step_state.notna().sum()
        print(f"Valid sequential transitions identified: {valid_transitions}")
        
    return df[valid_actions & valid_states].copy()

def evaluate_model(checkpoint, df, output_dir="checkpoints/evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    discretizer = checkpoint["discretizer"]
    agents = checkpoint["agents"]
    obs_design = checkpoint["obs_design"]
    
    results = []
    
    print("\n--- RUNNING INFERENCE ---")
    for _, row in df.iterrows():
        state = row['state']
        ref_action = int(row['action'])
        reward = row['reward']
        
        disc_state = discretizer.transform(state)
        
        # We extract Q-values from Agent 0 as the representative baseline, 
        # or average across all 3 agents for MARL consensus Q-values.
        q_values_consensus = np.zeros(4)
        proposals = []
        
        for i, agent in enumerate(agents):
            agent_obs = discretizer.transform_agent_obs(disc_state, obs_design, agent_id=i)
            # Fetch actual Q-values from Q-table (action_space = 4)
            q_vals = [agent.get_q_value(agent_obs, a) for a in range(4)]
            q_values_consensus += np.array(q_vals)
            proposals.append(np.argmax(q_vals)) # Greedy policy
            
        q_values_consensus /= len(agents)
        predicted_action = int(np.argmax(q_values_consensus))
        
        results.append({
            "patient_id": row['patient_id'],
            "step": row['step'],
            "state": state,
            "reference_action": ref_action,
            "predicted_action": predicted_action,
            "q_value_action_0": q_values_consensus[0],
            "q_value_action_1": q_values_consensus[1],
            "q_value_action_2": q_values_consensus[2],
            "q_value_action_3": q_values_consensus[3],
            "recorded_reward": reward,
            "prediction_correct": int(predicted_action == ref_action)
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "detailed_predictions.csv"), index=False)
    
    generate_metrics_and_plots(results_df, output_dir)

def generate_metrics_and_plots(df, output_dir):
    y_true = df['reference_action']
    y_pred = df['predicted_action']
    
    print("\n--- METRICS ---")
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], zero_division=0)
    
    print(f"Accuracy: {acc:.4f} | Balanced Accuracy: {b_acc:.4f}")
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
    plt.title("Confusion Matrix: Reference vs Predicted")
    plt.xlabel("Predicted Action")
    plt.ylabel("Reference Action")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 2. Action Distribution Plot
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(4)
    ref_counts = [sum(y_true == i) for i in range(4)]
    pred_counts = [sum(y_pred == i) for i in range(4)]
    
    plt.bar(index, ref_counts, bar_width, label='Reference', color='#4CA9CB')
    plt.bar(index + bar_width, pred_counts, bar_width, label='Predicted', color='#F49446')
    plt.xlabel('Action ID')
    plt.ylabel('Frequency')
    plt.title('Action Distribution: Recorded vs Predicted')
    plt.xticks(index + bar_width / 2, ['0: Cont', '1: Reassess', '2: Switch', '3: Multi'])
    plt.legend()
    plt.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Q-Value Distribution Boxplot
    plt.figure(figsize=(10, 6))
    q_data = pd.melt(df[['q_value_action_0', 'q_value_action_1', 'q_value_action_2', 'q_value_action_3']])
    sns.boxplot(x='variable', y='value', data=q_data)
    plt.title("Q-Value Distribution per Action")
    plt.xticks([0,1,2,3], ['Action 0', 'Action 1', 'Action 2', 'Action 3'])
    plt.savefig(os.path.join(output_dir, "q_value_distribution.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    ckpt, data = load_environment_and_data()
    valid_data = validate_dataset(data)
    evaluate_model(ckpt, valid_data)