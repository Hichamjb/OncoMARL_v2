import numpy as np


class IndependentDoubleQAgent:

  def __init__(
      self,
      agent_id: int,
      action_dim: int = 4,
      lr: float = 0.05,
      gamma: float = 0.95,
      initial_epsilon: float = 1.0,
      epsilon_decay: float = 0.992,
      min_epsilon: float = 0.05,
  ):
    self.agent_id = agent_id
    self.action_dim = action_dim
    self.lr = lr
    self.gamma = gamma
    self.epsilon = initial_epsilon
    self.epsilon_decay = epsilon_decay
    self.min_epsilon = min_epsilon

    self.Q_A = {}
    self.Q_B = {}

  def get_q_values(self, obs):
    q_a = self.Q_A.get(obs, np.zeros(self.action_dim))
    q_b = self.Q_B.get(obs, np.zeros(self.action_dim))
    return (q_a + q_b) / 2.0

  def select_action(self, obs, eval_mode=False):
    if not eval_mode and np.random.rand() < self.epsilon:
      return int(np.random.randint(self.action_dim))
    q_avg = self.get_q_values(obs)
    return int(np.argmax(q_avg))

  def update(self, obs, action, reward, next_obs, done):
    """تحديث Double Q-Learning مع إيقاف التقدير المستقبلي عند النهاية (1 - done)."""
    q_a = self.Q_A.get(obs, np.zeros(self.action_dim)).copy()
    q_b = self.Q_B.get(obs, np.zeros(self.action_dim)).copy()

    if np.random.rand() < 0.5:
      q_a_next = self.Q_A.get(next_obs, np.zeros(self.action_dim))
      best_a = int(np.argmax(q_a_next))
      q_b_next = self.Q_B.get(next_obs, np.zeros(self.action_dim))

      target = reward + self.gamma * (1.0 - float(done)) * q_b_next[best_a]
      q_a[action] += self.lr * (target - q_a[action])
      self.Q_A[obs] = q_a
    else:
      q_b_next = self.Q_B.get(next_obs, np.zeros(self.action_dim))
      best_a = int(np.argmax(q_b_next))
      q_a_next = self.Q_A.get(next_obs, np.zeros(self.action_dim))

      target = reward + self.gamma * (1.0 - float(done)) * q_a_next[best_a]
      q_b[action] += self.lr * (target - q_b[action])
      self.Q_B[obs] = q_b

  def decay_epsilon(self):
    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)