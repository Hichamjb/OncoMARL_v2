import os
import json
import pickle
import numpy as np
import pandas as pd
from src.agents.double_q_agent import IndependentDoubleQAgent
from src.coordinator.majority_voting import MajorityVotingCoordinator
from src.environment.oncomarl_env import OncoMARLEnv
from src.state_discretizer import StateDiscretizer


def train_pipeline(
    num_episodes=500,
    seeds=[10, 20, 30, 40, 50],
    obs_design="full",
    output_dir="checkpoints",
):
    """
    Entraîne le système multi-agent (MARL) sur plusieurs seeds et sauvegarde 
    le modèle ainsi que l'historique de l'apprentissage.
    """
    print(f"\n[Training] Démarrage de l'entraînement (Episodes: {num_episodes}, Design: {obs_design})")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Préparation du discrétiseur sur l'ensemble d'entraînement uniquement
    train_df = pd.read_csv("data/processed/train_patients.csv")
    discretizer = StateDiscretizer()
    discretizer.fit(train_df)
    print(f"[Training] Discrétisation configurée à partir de {len(train_df)} transitions.")

    best_agents = None
    best_reward = -float("inf")
    seed_results = []

    # 2. Boucle d'entraînement sur différentes initialisations (seeds)
    for seed in seeds:
        print(f"  -> Entraînement en cours pour Seed {seed}...")
        env = OncoMARLEnv(seed=seed)
        coordinator = MajorityVotingCoordinator()
        
        # Initialisation de 3 agents Double Q-Learning indépendants
        agents = [IndependentDoubleQAgent(agent_id=i) for i in range(3)]

        episode_rewards = []
        agreements = []

        # Boucle des épisodes
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            disc_state = discretizer.transform(obs)
            ep_reward = 0.0
            ep_agreed_steps = 0

            # Simulation des 12 visites du patient
            for step in range(12):
                proposals = []
                
                # Étape A: Chaque agent propose une action basée sur son observation
                for i, agent in enumerate(agents):
                    agent_obs = discretizer.transform_agent_obs(
                        disc_state, obs_design=obs_design, agent_id=i
                    )
                    proposals.append(agent.select_action(agent_obs, eval_mode=False))

                # Étape B: Le coordinateur tranche par vote majoritaire
                voted_action, is_unanimous, is_tie, _ = coordinator.coordinate(
                    proposals, eval_mode=False
                )
                if is_unanimous:
                    ep_agreed_steps += 1

                # Étape C: L'environnement exécute l'action votée
                next_obs, reward, done, _ = env.step(voted_action)
                next_disc_state = discretizer.transform(next_obs)

                # Étape D: Mise à jour (Apprentissage) de chaque agent
                for i, agent in enumerate(agents):
                    agent_obs = discretizer.transform_agent_obs(
                        disc_state, obs_design=obs_design, agent_id=i
                    )
                    next_agent_obs = discretizer.transform_agent_obs(
                        next_disc_state, obs_design=obs_design, agent_id=i
                    )
                    
                    # Mise à jour de la table Q avec la récompense commune
                    agent.update(agent_obs, voted_action, reward, next_agent_obs, done)
                    
                    # Décroissance de l'epsilon (réduction de l'exploration)
                    agent.decay_epsilon()

                disc_state = next_disc_state
                ep_reward += reward

            # Enregistrement des performances de l'épisode (Loss/Gain)
            episode_rewards.append(ep_reward)
            agreements.append(ep_agreed_steps / 12.0)

        # 3. Fin de l'entraînement pour ce seed : Collecte des statistiques
        mean_final_reward = float(np.mean(episode_rewards[-50:]))
        
        # SAUVEGARDE COMPLÈTE DE L'HISTORIQUE (Nouveau)
        seed_results.append({
            "seed": seed,
            "mean_final_reward": mean_final_reward,
            "mean_agreement": float(np.mean(agreements[-50:])),
            "reward_history": episode_rewards,      # L'historique complet des récompenses par épisode
            "agreement_history": agreements         # L'historique complet du taux d'accord par épisode
        })

        # Sauvegarder le meilleur modèle pour l'inférence
        if mean_final_reward > best_reward:
            best_reward = mean_final_reward
            best_agents = agents

    # 4. Sauvegarde des artefacts d'entraînement
    
    # A. Sauvegarde du modèle (Checkpoint PKL)
    checkpoint_path = os.path.join(output_dir, "marl_model.pkl")
    checkpoint_data = {
        "discretizer": discretizer,
        "agents": best_agents,
        "obs_design": obs_design,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    # B. Sauvegarde des résultats au format JSON (Incluant l'historique)
    summary_path = os.path.join(output_dir, f"train_summary_{obs_design}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(seed_results, f, indent=2)

    print(f"\n[Training Complete]")
    print(f"✅ Modèle sauvegardé dans       : '{checkpoint_path}'")
    print(f"✅ Historique sauvegardé dans : '{summary_path}'")
    
    return seed_results


if __name__ == "__main__":
    # Lancer l'entraînement par défaut (500 épisodes avec toutes les observations)
    train_pipeline(num_episodes=500, obs_design="full")