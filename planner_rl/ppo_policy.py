import pathlib
import numpy as np
from planner_rl.state_encoder import ACTIONS, ACTION_INDEX, N_ACTIONS, STATE_DIM

#Hyperparameters
GAMMA          = 0.99
LAMBDA_GAE     = 0.95
CLIP_EPS       = 0.2
LR_ACTOR       = 3e-4
LR_CRITIC      = 1e-3
ENTROPY_START  = 0.05
ENTROPY_END    = 0.003
ENTROPY_DECAY  = 60_000   
PPO_EPOCHS     = 4
HIDDEN_1       = 64
HIDDEN_2       = 32
LOG_RATIO_CLIP = 3.0



def relu(x):      return np.maximum(0.0, x)
def relu_grad(x): return (x > 0).astype(np.float64)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / fan_in)

def _clip_grads(grads: dict, max_norm: float = 0.5):
    total = float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for k in grads:
            grads[k] = grads[k] * scale



class ActorNetwork:
    def __init__(self):
        self.W1 = he_init(STATE_DIM, HIDDEN_1)
        self.b1 = np.zeros(HIDDEN_1)
        self.W2 = he_init(HIDDEN_1, HIDDEN_2)
        self.b2 = np.zeros(HIDDEN_2)
        self.W3 = he_init(HIDDEN_2, N_ACTIONS)
        self.b3 = np.zeros(N_ACTIONS)
        self._total_steps = 0
        self._init_adam()

    def forward(self, x):
        x  = x.astype(np.float64)
        z1 = x @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = relu(z2)
        logits = a2 @ self.W3 + self.b3
        probs  = softmax(logits)
        return probs, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "logits": logits}

    def get_probs(self, x):
        p, _ = self.forward(x)
        return p

    def get_log_prob(self, x, idx):
        p, _ = self.forward(x)
        return float(np.log(p[idx] + 1e-8))

    def _entropy_coef(self) -> float:
        t = min(self._total_steps / ENTROPY_DECAY, 1.0)
        return ENTROPY_START + t * (ENTROPY_END - ENTROPY_START)

    def backward_and_update(self, cache, action_idx, advantage, old_log_prob):
        self._total_steps += 1
        entropy_coef = self._entropy_coef()

        probs     = softmax(cache["logits"])
        log_prob  = float(np.log(probs[action_idx] + 1e-8))
        log_ratio = float(np.clip(log_prob - old_log_prob, -LOG_RATIO_CLIP, LOG_RATIO_CLIP))
        ratio     = float(np.exp(log_ratio))
        c_ratio   = float(np.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS))

        if advantage >= 0:
            clip_active = ratio > (1.0 + CLIP_EPS)
        else:
            clip_active = ratio < (1.0 - CLIP_EPS)

        surr1     = ratio   * advantage
        surr2     = c_ratio * advantage

        score = -probs.copy()
        score[action_idx] += 1.0
        
        if clip_active:
            d_logits = np.zeros_like(probs)
        else:
            d_logits = -ratio * advantage * score

        ent = np.sum(probs * np.log(probs + 1e-8))
        d_logits += entropy_coef * probs * (np.log(probs + 1e-8) - ent)

        dW3 = np.outer(cache["a2"], d_logits); db3 = d_logits.copy()
        d_a2 = d_logits @ self.W3.T
        d_z2 = d_a2 * relu_grad(cache["z2"])
        dW2  = np.outer(cache["a1"], d_z2); db2 = d_z2.copy()
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * relu_grad(cache["z1"])
        dW1  = np.outer(cache["x"], d_z1); db1 = d_z1.copy()

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        _clip_grads(grads, max_norm=0.5)
        self._adam_update(grads, LR_ACTOR)
        return float(min(surr1, surr2))

    def _init_adam(self):
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def _adam_update(self, grads, lr):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g**2
            mh = self._m[k] / (1 - b1**self._t)
            vh = self._v[k] / (1 - b2**self._t)
            getattr(self, k)[:] -= lr * mh / (np.sqrt(vh) + eps)

    def to_dict(self):
        d = {k: v.tolist() for k, v in self._params().items()}
        d["_total_steps"] = self._total_steps
        return d

    def from_dict(self, d):
        for k, v in d.items():
            if k == "_total_steps":
                self._total_steps = int(v)
            else:
                setattr(self, k, np.array(v, dtype=np.float64))
        self._init_adam()

class CriticNetwork:
    def __init__(self):
        self.W1 = he_init(STATE_DIM, HIDDEN_1)
        self.b1 = np.zeros(HIDDEN_1)
        self.W2 = he_init(HIDDEN_1, HIDDEN_2)
        self.b2 = np.zeros(HIDDEN_2)
        self.W3 = he_init(HIDDEN_2, 1)
        self.b3 = np.zeros(1)
        self._init_adam()

    def forward(self, x):
        x  = x.astype(np.float64)
        z1 = x @ self.W1 + self.b1; a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = relu(z2)
        val = float((a2 @ self.W3 + self.b3)[0])
        return val, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def value(self, x):
        v, _ = self.forward(x)
        return v

    def backward_and_update(self, cache, target):
        a2    = cache["a2"]
        val   = float((a2 @ self.W3 + self.b3)[0])
        d_val = 2.0 * (val - target)
        dW3   = a2.reshape(-1, 1) * d_val; db3 = np.array([d_val])
        d_a2  = (self.W3 * d_val).flatten()
        d_z2  = d_a2 * relu_grad(cache["z2"])
        dW2   = np.outer(cache["a1"], d_z2); db2 = d_z2.copy()
        d_a1  = d_z2 @ self.W2.T
        d_z1  = d_a1 * relu_grad(cache["z1"])
        dW1   = np.outer(cache["x"], d_z1); db1 = d_z1.copy()
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        _clip_grads(grads, max_norm=1.0)
        self._adam_update(grads, LR_CRITIC)
        return float((val - target) ** 2)

    def _init_adam(self):
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def _adam_update(self, grads, lr):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g**2
            mh = self._m[k] / (1 - b1**self._t)
            vh = self._v[k] / (1 - b2**self._t)
            getattr(self, k)[:] -= lr * mh / (np.sqrt(vh) + eps)

    def to_dict(self):  return {k: v.tolist() for k, v in self._params().items()}

    def from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, np.array(v, dtype=np.float64))
        self._init_adam()


def compute_gae(rewards, values, next_value, dones):
    advantages, gae = [], 0.0
    values_ext = values + [next_value]
    for t in reversed(range(len(rewards))):
        mask  = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * values_ext[t + 1] * mask - values_ext[t]
        gae   = delta + GAMMA * LAMBDA_GAE * mask * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class PPOPolicy:

    def __init__(self, save_path: str = "rl/ppo_weights.npz"):
        self.save_path = pathlib.Path(save_path)
        self.actor     = ActorNetwork()
        self.critic    = CriticNetwork()
        self._buffer: list[dict] = []
        if self.save_path.exists():
            self._load()

    def select_action(self, state, forbidden=None):
        probs = self.actor.get_probs(state).copy()
        if forbidden:
            for i, a in enumerate(ACTIONS):
                if a in forbidden:
                    probs[i] = 0.0
            total = probs.sum()
            if total < 1e-8:
                full_probs = np.zeros(N_ACTIONS)
                full_probs[ACTION_INDEX["run_writer"]] = 1.0
                return "run_writer", full_probs
            probs /= total

        chosen_idx = int(np.random.choice(N_ACTIONS, p=probs))
        full_probs = self.actor.get_probs(state).copy()
        return ACTIONS[chosen_idx], full_probs

    def update(self, state, action_index, reward, done: bool = True):
        log_prob = self.actor.get_log_prob(state, action_index)
        value    = self.critic.value(state)
        self._buffer.append({
            "state":        state.copy(),
            "action_index": action_index,
            "reward":       reward,
            "log_prob":     log_prob,
            "value":        value,
            "done":         done,
        })
        if len(self._buffer) >= 10:
            self.flush_update()

    def flush_update(self):
        if not self._buffer:
            return {}
        states        = [t["state"]        for t in self._buffer]
        actions       = [t["action_index"] for t in self._buffer]
        rewards       = [t["reward"]       for t in self._buffer]
        old_log_probs = [t["log_prob"]     for t in self._buffer]
        values        = [t["value"]        for t in self._buffer]
        dones         = [t["done"]         for t in self._buffer]

        last_done  = dones[-1]
        next_val   = 0.0 if last_done else self.critic.value(states[-1])
        advantages, returns = compute_gae(rewards, values, next_val, dones)

        adv_arr = np.array(advantages)
        if adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        actor_losses, critic_losses = [], []
        n = len(states)
        for _ in range(PPO_EPOCHS):
            for i in np.random.permutation(n):
                _, ac = self.actor.forward(states[i])
                actor_losses.append(
                    self.actor.backward_and_update(
                        ac, actions[i], float(adv_arr[i]), old_log_probs[i]))
                _, cc = self.critic.forward(states[i])
                critic_losses.append(
                    self.critic.backward_and_update(cc, returns[i]))

        self._buffer.clear()
        self._save()
        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
        }

    def get_theta(self, action_index):
        W_eff = self.actor.W1 @ self.actor.W2 @ self.actor.W3
        return W_eff[:, action_index]

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        ad, cd = self.actor.to_dict(), self.critic.to_dict()
        np.savez(
            str(self.save_path),
            actions=np.array(ACTIONS),
            **{f"actor_{k}":  np.array(v) for k, v in ad.items()},
            **{f"critic_{k}": np.array(v) for k, v in cd.items()},
        )

    def _load(self):
        data = np.load(str(self.save_path), allow_pickle=True)
        if list(data["actions"]) != ACTIONS:
            print("[ppo] Saved actions mismatch — resetting weights.")
            return
        self.actor.from_dict(
            {k[6:]: data[k] for k in data.files if k.startswith("actor_")})
        self.critic.from_dict(
            {k[7:]: data[k] for k in data.files if k.startswith("critic_")})