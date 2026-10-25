import numpy as np


class OncoMARLEnv:

  def __init__(self, max_visits=12, seed=42):
    self.max_visits = max_visits
    self.rng = np.random.default_rng(seed)
    self.reset()

  def reset(self, seed=None):
    if seed is not None:
      self.rng = np.random.default_rng(seed)

    self.step_idx = 0
    self.tb = float(self.rng.uniform(0.40, 0.80))
    self.tox = float(self.rng.uniform(0.10, 0.30))
    self.unc = float(self.rng.uniform(0.20, 0.60))
    self.prog = 0.0
    self.prev_act = -1.0

    return self._get_obs()

  def _get_obs(self):
    visit_norm = self.step_idx / float(self.max_visits)
    return np.array(
        [self.tb, self.tox, self.unc, self.prog, visit_norm, self.prev_act],
        dtype=np.float32,
    )

  def step(self, action: int):
    # ديناميكيات الانتقال حسب الإجراء
    if action == 0:  # CONTINUE
      dtb = self.rng.normal(+0.015, 0.005)
      dtox = self.rng.normal(-0.010, 0.004)
      dunc = self.rng.normal(-0.010, 0.005)
    elif action == 1:  # REASSESS
      dtb = self.rng.normal(-0.035, 0.010)
      dtox = self.rng.normal(-0.020, 0.005)
      dunc = self.rng.normal(-0.050, 0.010)
    elif action == 2:  # SWITCH
      dtb = self.rng.normal(-0.025, 0.012)
      dtox = self.rng.normal(+0.045, 0.010)
      dunc = self.rng.normal(-0.020, 0.008)
    else:  # MULTIDISCIPLINARY
      dtb = self.rng.normal(+0.002, 0.004)
      dtox = self.rng.normal(-0.015, 0.005)
      dunc = self.rng.normal(-0.150, 0.020)

    next_tb = float(np.clip(self.tb + dtb, 0.0, 1.0))
    next_tox = float(np.clip(self.tox + dtox, 0.0, 1.0))
    next_unc = float(np.clip(self.unc + dunc, 0.0, 1.0))

    next_prog = 1.0 if (next_tb > 0.85 or next_tox > 0.80) else 0.0

    reward = -(
        (next_tb - self.tb)
        + (next_tox - self.tox)
        + 0.10 * (1.0 if action == 2 else 0.0)
        + 0.05 * (1.0 if action == 3 else 0.0)
    ) - 0.20 * next_prog

    self.step_idx += 1
    self.tb, self.tox, self.unc, self.prog, self.prev_act = (
        next_tb,
        next_tox,
        next_unc,
        next_prog,
        float(action),
    )

    done = self.step_idx >= self.max_visits
    next_obs = self._get_obs()

    return next_obs, float(reward), done, {}