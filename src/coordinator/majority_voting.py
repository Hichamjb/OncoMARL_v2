import numpy as np


class MajorityVotingCoordinator:

  def __init__(self, action_dim=4):
    self.action_dim = action_dim

  def coordinate(self, agent_proposals, eval_mode=True):
    counts = np.bincount(agent_proposals, minlength=self.action_dim)
    max_votes = np.max(counts)
    candidates = np.where(counts == max_votes)[0]

    is_unanimous = bool(max_votes == len(agent_proposals))
    is_tie = bool(len(candidates) > 1)

    # حسم التعادل بطريقة حتمية في التقييم (اختيار أصغر فهرس)
    if eval_mode:
      final_action = int(candidates[0])
    else:
      final_action = int(np.random.choice(candidates))

    return final_action, is_unanimous, is_tie, counts