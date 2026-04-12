import json
import pathlib
import numpy as np
from planner_rl.state_encoder import ACTIONS, ACTION_INDEX, N_ACTIONS, STATE_DIM

# Exploration coefficient.
ALPHA: float = 1.0


class LinUCBPolicy:

    def __init__(self, save_path: str | pathlib.Path = "rl/policy_weights.json"):
        self.save_path = pathlib.Path(save_path)
        # Per-arm matrices — shape (STATE_DIM, STATE_DIM) and (STATE_DIM,)
        if self.save_path.exists():
            self._load()
        else:
            self._init_fresh()


    def select_action(
        self,
        state: np.ndarray,
        forbidden: list[str] | None = None,
    ) -> tuple[str, np.ndarray]:
        #Select the action with the highest UCB score.
        forbidden = forbidden or []
        ucb_scores = np.full(N_ACTIONS, -np.inf, dtype=np.float64)

        for i, action in enumerate(ACTIONS):
            if action in forbidden:
                continue
            A_inv         = np.linalg.inv(self.A[i])
            theta         = A_inv @ self.b[i]
            exploit       = float(theta @ state)
            explore       = float(ALPHA * np.sqrt(state @ A_inv @ state))
            ucb_scores[i] = exploit + explore

        chosen_idx = int(np.argmax(ucb_scores))

        # Softmax over valid scores for probability logging
        valid_mask = ucb_scores > -np.inf
        probs = np.zeros(N_ACTIONS, dtype=np.float64)
        if valid_mask.any():
            vals        = ucb_scores[valid_mask]
            exp_vals    = np.exp(vals - vals.max())   
            probs[valid_mask] = exp_vals / exp_vals.sum()

        return ACTIONS[chosen_idx], probs

    def update(
        self,
        state:        np.ndarray,
        action_index: int,
        reward:       float,
    ) -> None:

        x = state.astype(np.float64)
        self.A[action_index] += np.outer(x, x)
        self.b[action_index] += reward * x
        self._save()

    def get_theta(self, action_index: int) -> np.ndarray:
        """Return the learned weight vector θ for a given arm (for diagnostics)."""
        return np.linalg.inv(self.A[action_index]) @ self.b[action_index]

    def _init_fresh(self) -> None:
        self.A = [np.eye(STATE_DIM, dtype=np.float64) for _ in range(N_ACTIONS)]
        self.b = [np.zeros(STATE_DIM, dtype=np.float64) for _ in range(N_ACTIONS)]

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha":   ALPHA,
            "actions": ACTIONS,
            "A":       [m.tolist() for m in self.A],
            "b":       [v.tolist() for v in self.b],
        }
        self.save_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self.save_path.read_text())
        # Validate that the saved action list matches current ACTIONS
        if data.get("actions") != ACTIONS:
            print(
                "[rl] Saved policy actions don't match current ACTIONS. "
                "Resetting weights.",
            )
            self._init_fresh()
            return
        self.A = [np.array(m, dtype=np.float64) for m in data["A"]]
        self.b = [np.array(v, dtype=np.float64) for v in data["b"]]