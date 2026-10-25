import pickle
from src.coordinator.majority_voting import MajorityVotingCoordinator

# VÉRIFIEZ QUE CETTE LIGNE EST BIEN PRÉSENTE :
class OncoMARLInferenceEngine:
    def __init__(self, model_path="checkpoints/marl_model.pkl"):
        self.is_loaded = False
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.discretizer = data["discretizer"]
            self.agents = data["agents"]
            self.obs_design = data["obs_design"]
            self.coordinator = MajorityVotingCoordinator()
            self.action_names = {
                0: "CONTINUE_CURRENT_PATHWAY",
                1: "REASSESS_CURRENT_PATHWAY",
                2: "SWITCH_PATHWAY",
                3: "MULTIDISCIPLINARY_REASSESSMENT"
            }
            self.is_loaded = True
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")

    # VÉRIFIEZ L'INDENTATION (les espaces) DE CETTE FONCTION
    def predict(self, obs):
        disc_state = self.discretizer.transform(obs)
        proposals = []
        
        for i, agent in enumerate(self.agents):
            agent_obs = self.discretizer.transform_agent_obs(disc_state, self.obs_design, agent_id=i)
            action = agent.select_action(agent_obs, eval_mode=True)
            proposals.append(action)

        voted_action, is_unanimous, is_tie, _ = self.coordinator.coordinate(proposals, eval_mode=True)
        
        return {
            "voted_action_id": voted_action,
            "voted_action_name": self.action_names[voted_action],
            "is_unanimous": is_unanimous,
            "is_tie": is_tie,
            "agent_actions": proposals 
        }