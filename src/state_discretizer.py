import ast
import numpy as np
import pandas as pd


class StateDiscretizer:

  def __init__(
      self, n_bins_tb=4, n_bins_tox=4, n_bins_unc=3, n_bins_visit=4
  ):
    self.n_bins_tb = n_bins_tb
    self.n_bins_tox = n_bins_tox
    self.n_bins_unc = n_bins_unc
    self.n_bins_visit = n_bins_visit
    self.bins = {}

  def fit(self, train_df: pd.DataFrame):
    """Fit quantile boundaries on training data (supports both explicit and vectorized columns)."""
    if "current_tumor_burden" in train_df.columns:
      tb_data = train_df["current_tumor_burden"]
      tox_data = train_df["current_toxicity"]
      unc_data = train_df["current_uncertainty"]
      visit_data = train_df["current_visit_index_normalized"]
    elif "state" in train_df.columns:
      states = train_df["state"]
      if len(states) > 0 and isinstance(states.iloc[0], str):
        states = states.apply(ast.literal_eval)
      state_matrix = np.array(states.tolist())
      tb_data = state_matrix[:, 0]
      tox_data = state_matrix[:, 1]
      unc_data = state_matrix[:, 2]
      visit_data = state_matrix[:, 4]
    else:
      raise KeyError(
          "train_df must contain either explicit state columns or a 'state'"
          " vector column."
      )

    self.bins["tb"] = np.quantile(
        tb_data, np.linspace(0, 1, self.n_bins_tb + 1)
    )
    self.bins["tox"] = np.quantile(
        tox_data, np.linspace(0, 1, self.n_bins_tox + 1)
    )
    self.bins["unc"] = np.quantile(
        unc_data, np.linspace(0, 1, self.n_bins_unc + 1)
    )
    self.bins["visit"] = np.quantile(
        visit_data, np.linspace(0, 1, self.n_bins_visit + 1)
    )

  def transform(self, state_vector):
    """Convert continuous state vector into discrete feature bin indices."""
    if isinstance(state_vector, str):
      state_vector = ast.literal_eval(state_vector)

    tb, tox, unc, prog, visit_norm, prev_act = state_vector

    tb_idx = int(
        min(
            max(np.digitize(tb, self.bins["tb"]) - 1, 0), self.n_bins_tb - 1
        )
    )
    tox_idx = int(
        min(
            max(np.digitize(tox, self.bins["tox"]) - 1, 0), self.n_bins_tox - 1
        )
    )
    unc_idx = int(
        min(
            max(np.digitize(unc, self.bins["unc"]) - 1, 0), self.n_bins_unc - 1
        )
    )
    prog_idx = int(prog > 0.5)
    visit_idx = int(
        min(
            max(np.digitize(visit_norm, self.bins["visit"]) - 1, 0),
            self.n_bins_visit - 1,
        )
    )
    prev_act_idx = int(prev_act) + 1  # -1..3 -> 0..4

    return (tb_idx, tox_idx, unc_idx, prog_idx, visit_idx, prev_act_idx)

  def transform_agent_obs(
      self, full_discrete_state, obs_design="full", agent_id=0
  ):
    if obs_design == "full":
      return full_discrete_state
    elif obs_design == "specialized":
      tb_idx, tox_idx, unc_idx, prog_idx, visit_idx, prev_act_idx = (
          full_discrete_state
      )
      if agent_id == 0:  # Agent 1: Tumor Burden & Progression
        return (tb_idx, prog_idx)
      elif agent_id == 1:  # Agent 2: Toxicity
        return (tox_idx,)
      else:  # Agent 3: Uncertainty, Visit Index & Prev Action
        return (unc_idx, visit_idx, prev_act_idx)
    else:
      raise ValueError(f"Unknown observation design: {obs_design}")