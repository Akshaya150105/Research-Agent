from __future__ import annotations

import json
import math
import pathlib
import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


MODEL_PATH = pathlib.Path("shared_memory/gap_rl_model.json")

# LinUCB
ALPHA             = 1.5    
ALPHA_MIN         = 0.3   
FEATURE_DIM       = 18     

# Simulation training
SIM_EPISODES      = 5_000  
SIM_CANDIDATES    = 30     
SIM_SELECT_K      = 8      

# Epsilon-greedy warm-up
EPSILON_START         = 0.30   
EPSILON_END           = 0.02   
EPSILON_WARMUP_FRAC   = 0.60   

# Reward weights  
W_NOVELTY         = 0.20
W_SPECIFICITY     = 0.18
W_COVERAGE        = 0.18
W_CONSISTENCY     = 0.14
W_CONFIDENCE      = 0.18
W_PRIORITY        = 0.12   


# Gap categories and priority levels used by the RL agent
class GapType(str, Enum):
    COMBINATORIAL = "combinatorial"
    LIMITATION    = "limitation"
    CROSS_PAPER   = "cross_paper"


class GapPriority(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"

# Numeric mapping for priority levels (used in reward calculation)
_PRIORITY_SCORE = {GapPriority.HIGH: 1.0, GapPriority.MEDIUM: 0.5, GapPriority.LOW: 0.1}


# Session-level context extracted from input papers
@dataclass
class SessionContext:

    #Summary of the current session used by the RL agent.

    n_papers:             int   = 1
    n_contradictions:     int   = 0
    n_high_critiques:     int   = 0
    used_gap_matrix:      bool  = False      
    domain_diversity:     float = 0.5        
    mean_coverage_gain:   float = 0.5        


def session_context_from_papers(papers: list[dict],
                                comp_ctx: Optional[dict] = None,
                                used_gap_matrix: bool = False) -> SessionContext:
    
    """
    Builds session context from input papers and comparator data
    """
    n_papers = len(papers)

    # contradiction count from comparator context
    n_contradictions = 0
    if comp_ctx and comp_ctx.get("available"):
        n_contradictions = len(comp_ctx.get("contradictions", []))

    # high-severity critiques across papers
    n_high = 0
    for p in papers:
        cs = p.get("critique_summary", {})
        if isinstance(cs, dict):
            n_high += len(cs.get("high_weakness_types", []))

    # domain diversity using entropy over entity types
    type_counts: dict[str, int] = {}
    for p in papers:
        ei = p.get("entity_index", {})
        if isinstance(ei, dict):
            for etype, edict in ei.items():
                if isinstance(edict, dict):
                    type_counts[etype] = type_counts.get(etype, 0) + len(edict)
    total = sum(type_counts.values()) or 1
    entropy = 0.0
    for v in type_counts.values():
        p_i = v / total
        if p_i > 0:
            entropy -= p_i * math.log2(p_i)
    max_entropy = math.log2(max(len(type_counts), 1))
    diversity = (entropy / max_entropy) if max_entropy > 0 else 0.5

    return SessionContext(
        n_papers          = n_papers,
        n_contradictions  = n_contradictions,
        n_high_critiques  = n_high,
        used_gap_matrix   = used_gap_matrix,
        domain_diversity  = round(diversity, 4),
        mean_coverage_gain= 0.5,  
    )


# Feature engineering for RL state representation

def _gap_type_onehot(gap_type: str) -> list[float]:
    """One-hot encoding for gap type"""
    order = [GapType.COMBINATORIAL.value,
             GapType.LIMITATION.value,
             GapType.CROSS_PAPER.value]
    return [1.0 if gap_type == t else 0.0 for t in order]


def _entity_counts(gap: dict) -> tuple[int, int]:
    """Returns number of unique entities and papers involved."""
    entities = gap.get("entities_involved", [])
    papers   = gap.get("papers_involved",   [])
    return len(set(entities)), len(set(papers))


def _redundancy_score(gap: dict, already_selected: list[dict]) -> float:
    """Measures overlap with already selected gaps using Jaccard similarity."""
    if not already_selected:
        return 0.0
    my_ents = set(gap.get("entities_involved", []))
    if not my_ents:
        return 0.0
    sims = []
    for sel in already_selected:
        their_ents = set(sel.get("entities_involved", []))
        if not their_ents:
            sims.append(0.0)
            continue
        inter = len(my_ents & their_ents)
        union = len(my_ents | their_ents)
        sims.append(inter / union if union else 0.0)
    return max(sims)


def build_feature_vector(gap: dict,
                         already_selected: list[dict],
                         ctx: SessionContext,
                         n_candidates: int) -> np.ndarray:
    """
    Builds the feature vector representing a gap candidate

    """
    type_oh = _gap_type_onehot(gap.get("gap_type", "combinatorial"))
    n_ents, n_papers = _entity_counts(gap)
    priority_str = gap.get("priority", GapPriority.MEDIUM.value)
    try:
        priority_score = _PRIORITY_SCORE[GapPriority(priority_str)]
    except (ValueError, KeyError):
        priority_score = 0.5
    redundancy = _redundancy_score(gap, already_selected)
    addressed_flag = 1.0 if gap.get("addressed_status") == "not_addressed" else 0.0

    vec = [
        type_oh[0],
        type_oh[1],
        type_oh[2],
        priority_score,
        float(gap.get("confidence", 0.5)),
        min(n_ents  / 10.0, 1.0),
        min(n_papers / max(n_candidates, 1), 1.0),
        redundancy,
        float(gap.get("llm_validated", False)),
        float(not gap.get("needs_review", False)),
        addressed_flag,
        min(ctx.n_papers          / 20.0, 1.0),
        min(ctx.n_contradictions  / 10.0, 1.0),
        min(ctx.n_high_critiques  / 10.0, 1.0),
        float(ctx.used_gap_matrix),
        ctx.domain_diversity,
        ctx.mean_coverage_gain,
        1.0,   # bias
    ]
    assert len(vec) == FEATURE_DIM, f"Expected {FEATURE_DIM} dims, got {len(vec)}"
    return np.array(vec, dtype=np.float64)


# Reward function used by RL agent

def compute_reward(gap: dict,
                   already_selected: list[dict],
                   ctx: SessionContext,
                   contradiction_entity_set: set[str]) -> float:

    """Computes reward score for a selected gap."""

    redundancy = _redundancy_score(gap, already_selected)
    r_novelty  = 1.0 - redundancy                                  

    n_ents, _ = _entity_counts(gap)
    r_specificity = min(n_ents / 5.0, 1.0)                        

    _, n_papers  = _entity_counts(gap)
    n_total      = max(ctx.n_papers, 1)
    r_coverage   = min(n_papers / n_total, 1.0)                   

    gap_ents     = set(gap.get("entities_involved", []))
    n_conflict   = len(gap_ents & contradiction_entity_set)
    conflict_frac = n_conflict / max(len(gap_ents), 1)
    r_consistency = 1.0 - conflict_frac                           

    r_confidence = float(gap.get("confidence", 0.5))              

    priority_str = gap.get("priority", GapPriority.MEDIUM.value)
    gap_type_str = gap.get("gap_type", GapType.COMBINATORIAL.value)
    priority_bonus = {
        GapPriority.HIGH.value:   1.0,
        GapPriority.MEDIUM.value: 0.5,
        GapPriority.LOW.value:    0.0,
    }.get(priority_str, 0.5)

    cross_paper_bonus = 0.25 if gap_type_str == GapType.CROSS_PAPER.value else 0.0
    r_priority = min(priority_bonus + cross_paper_bonus, 1.0)     

    r_total = (W_NOVELTY      * r_novelty
             + W_SPECIFICITY  * r_specificity
             + W_COVERAGE     * r_coverage
             + W_CONSISTENCY  * r_consistency
             + W_CONFIDENCE   * r_confidence
             + W_PRIORITY     * r_priority)

    return float(np.clip(r_total, 0.0, 1.0))


# Disjoint LinUCB model for gap selection

class LinUCBDisjoint:
    """
    Contextual bandit with separate parameters for each gap type.
    """
    def __init__(self, d: int = FEATURE_DIM, alpha: float = ALPHA):
        self.d     = d
        self.alpha = alpha
        self.arms  = [t.value for t in GapType]

        self.A: dict[str, np.ndarray] = {
            arm: np.eye(d, dtype=np.float64) for arm in self.arms
        }
        self.b: dict[str, np.ndarray] = {
            arm: np.zeros(d, dtype=np.float64) for arm in self.arms
        }
        self._A_inv_cache: dict[str, Optional[np.ndarray]] = {
            arm: np.eye(d, dtype=np.float64) for arm in self.arms
        }

    def _arm(self, gap: dict) -> str:
        gtype = gap.get("gap_type", GapType.COMBINATORIAL.value)
        return gtype if gtype in self.arms else GapType.COMBINATORIAL.value

    def _A_inv(self, arm: str) -> np.ndarray:
        if self._A_inv_cache[arm] is None:
            self._A_inv_cache[arm] = np.linalg.inv(self.A[arm])
        return self._A_inv_cache[arm]

    def score(self, gap: dict, x: np.ndarray) -> float:
        arm   = self._arm(gap)
        A_inv = self._A_inv(arm)
        theta = A_inv @ self.b[arm]
        exploit = float(theta @ x)
        explore = float(self.alpha * math.sqrt(float(x @ A_inv @ x)))
        return exploit + explore

    def update(self, gap: dict, x: np.ndarray, reward: float):
        arm              = self._arm(gap)
        self.A[arm]     += np.outer(x, x)
        self.b[arm]     += reward * x
        self._A_inv_cache[arm] = None  

    def to_dict(self) -> dict:
        return {
            "d":     self.d,
            "alpha": self.alpha,
            "arms":  self.arms,
            "A": {arm: self.A[arm].tolist() for arm in self.arms},
            "b": {arm: self.b[arm].tolist() for arm in self.arms},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCBDisjoint":
        model       = cls(d=data["d"], alpha=data["alpha"])
        model.arms  = data["arms"]
        model.A     = {arm: np.array(v) for arm, v in data["A"].items()}
        model.b     = {arm: np.array(v) for arm, v in data["b"].items()}
        model._A_inv_cache = {arm: None for arm in model.arms}
        return model


# Synthetic environment to generate realistic gap candidates for RL training

class GapEnvironment:

    TYPE_PROBS   = [0.70, 0.20, 0.10]   # combinatorial, limitation, cross_paper
    PRIORITY_PROB = {
        GapType.COMBINATORIAL: [0.25, 0.50, 0.25],  
        GapType.LIMITATION:    [0.20, 0.55, 0.25],
        GapType.CROSS_PAPER:   [0.60, 0.30, 0.10],
    }

    # confidence ranges per (type, priority)
    CONF_RANGE = {
        (GapType.COMBINATORIAL, GapPriority.HIGH):   (0.65, 0.95),
        (GapType.COMBINATORIAL, GapPriority.MEDIUM): (0.45, 0.75),
        (GapType.COMBINATORIAL, GapPriority.LOW):    (0.35, 0.60),
        (GapType.LIMITATION,    GapPriority.HIGH):   (0.70, 0.90),
        (GapType.LIMITATION,    GapPriority.MEDIUM): (0.55, 0.75),
        (GapType.LIMITATION,    GapPriority.LOW):    (0.40, 0.60),
        (GapType.CROSS_PAPER,   GapPriority.HIGH):   (0.80, 0.95),
        (GapType.CROSS_PAPER,   GapPriority.MEDIUM): (0.60, 0.80),
        (GapType.CROSS_PAPER,   GapPriority.LOW):    (0.40, 0.65),
    }

    # entity pools (representative tokens)
    METHODS  = ["transformer", "lstm", "cnn", "bert", "gpt", "t5",
                 "roberta", "xlnet", "gru", "attention", "seq2seq",
                 "dqn", "ppo", "sac", "mamba", "s4", "ssm"]
    DATASETS = ["squad", "glue", "superglue", "wikitext", "imdb",
                 "cnn_dm", "xsum", "multinews", "reddit", "newsqa",
                 "coqa", "quac", "natural_questions", "triviaqa"]
    TASKS    = ["summarisation", "qa", "classification", "ner",
                "translation", "generation", "parsing", "retrieval",
                "dialogue", "reading_comprehension"]
    METRICS  = ["bleu", "rouge", "f1", "accuracy", "meteor",
                "bertscore", "perplexity", "em"]
    ADDRESSED_STATUSES = ["not_addressed", "partially_addressed",
                           "addressed", "unknown"]
    ADDRESSED_PROBS    = [0.50, 0.25, 0.15, 0.10]

    def __init__(self, seed: int = 42):
        self.base_seed = seed
        self.rng = random.Random(seed)

    def _sample_entities(self, gap_type: GapType,
                          priority: GapPriority,
                          n_papers_session: int) -> tuple[list[str], list[str]]:
        """Returns (entities_involved, papers_involved)."""
        if gap_type == GapType.COMBINATORIAL:
            n_ents = self.rng.randint(2, 5)
            ents   = (self.rng.sample(self.METHODS,   min(2, n_ents // 2 + 1))
                    + self.rng.sample(self.DATASETS,   min(2, n_ents // 2 + 1))
                    + self.rng.sample(self.TASKS,      1))
        elif gap_type == GapType.LIMITATION:
            n_ents = self.rng.randint(1, 4)
            ents   = (self.rng.sample(self.TASKS,    min(2, n_ents))
                    + self.rng.sample(self.METRICS,   1))
        else:   
            n_ents = self.rng.randint(3, 6)
            ents   = (self.rng.sample(self.METHODS,   2)
                    + self.rng.sample(self.DATASETS,   2)
                    + self.rng.sample(self.METRICS,    1))

        max_papers = max(1, min(3, n_papers_session))
        if gap_type == GapType.CROSS_PAPER:
            lo = min(2, max_papers)
            n_papers = self.rng.randint(lo, max(lo, max_papers))
        else:
            n_papers = self.rng.randint(1, max_papers)
        paper_ids  = [f"paper_{self.rng.randint(0, n_papers_session - 1)}"
                      for _ in range(n_papers)]
        return list(set(ents)), list(set(paper_ids))

    def sample_gap(self, n_papers_session: int,
                   llm_available: bool = True) -> dict:
        """Generate one synthetic gap dict compatible with Gap dataclass."""
        types     = [t for t in GapType]
        gap_type  = self.rng.choices(types, weights=self.TYPE_PROBS, k=1)[0]
        priorities= [p for p in GapPriority]
        priority  = self.rng.choices(
            priorities,
            weights=self.PRIORITY_PROB[gap_type],
            k=1
        )[0]
        conf_lo, conf_hi = self.CONF_RANGE[(gap_type, priority)]
        confidence = round(self.rng.uniform(conf_lo, conf_hi), 3)

        llm_validated = llm_available and self.rng.random() < 0.70
        needs_review  = confidence < 0.60 and self.rng.random() < 0.40
        addressed     = self.rng.choices(
            self.ADDRESSED_STATUSES,
            weights=self.ADDRESSED_PROBS,
            k=1
        )[0]

        entities, papers = self._sample_entities(gap_type, priority, n_papers_session)

        return {
            "gap_id":            f"sim_{uuid.uuid4().hex[:6]}",
            "gap_type":          gap_type.value,
            "priority":          priority.value,
            "confidence":        confidence,
            "entities_involved": entities,
            "papers_involved":   papers,
            "llm_validated":     llm_validated,
            "needs_review":      needs_review,
            "addressed_status":  addressed,
            "description":       f"Simulated gap [{gap_type.value}]",
            "evidence":          "Simulated",
            "suggestion":        "Simulated",
        }

    def reseed(self, episode: int) -> None:
        """
        Re-seed the RNG per episode so each episode sees a fresh
        random sequence even when the trainer is created with a fixed seed.
        This prevents the model from memorising the same pseudo-random path.
        """
        self.rng = random.Random(self.base_seed + episode * 7919)  # prime stride

    def sample_session(self, n_candidates: int = SIM_CANDIDATES
                       ) -> tuple[list[dict], SessionContext, set[str]]:
        """
        Generate a full simulated session:
          - n_candidates gap candidates
          - A SessionContext drawn from realistic priors
          - A contradiction_entity_set (subset of entity pool)

        Returns (candidates, ctx, contradiction_entity_set).
        """
        n_papers       = self.rng.randint(1, 10)   
        n_contrads     = self.rng.randint(0, min(5, n_papers))
        n_high_crit    = self.rng.randint(0, 6)    
        used_gap_mat   = self.rng.random() < 0.50
        diversity      = round(self.rng.uniform(0.1, 1.0), 3)  
        coverage_gain  = round(self.rng.uniform(0.2, 1.0), 3)

        ctx = SessionContext(
            n_papers          = n_papers,
            n_contradictions  = n_contrads,
            n_high_critiques  = n_high_crit,
            used_gap_matrix   = used_gap_mat,
            domain_diversity  = diversity,
            mean_coverage_gain= coverage_gain,
        )

        llm_avail = self.rng.random() < 0.80
        candidates = [
            self.sample_gap(n_papers, llm_available=llm_avail)
            for _ in range(n_candidates)
        ]

        all_ents   = (self.METHODS + self.DATASETS
                      + self.TASKS  + self.METRICS)
        frac       = self.rng.uniform(0.10, 0.30)   
        k          = max(1, int(frac * len(all_ents)))
        contradiction_set = set(self.rng.sample(all_ents, k))

        return candidates, ctx, contradiction_set


# Trainer for LinUCB using simulated environment

class BanditTrainer:

    """
    Trains the LinUCB model using simulated sessions.
    """

    def __init__(self,
                 n_episodes:   int   = SIM_EPISODES,
                 n_candidates: int   = SIM_CANDIDATES,
                 k_select:     int   = SIM_SELECT_K,
                 seed:         int   = 42,
                 alpha_start:  float = ALPHA,
                 alpha_min:    float = ALPHA_MIN,
                 eps_start:    float = EPSILON_START,
                 eps_end:      float = EPSILON_END,
                 eps_warmup_frac: float = EPSILON_WARMUP_FRAC,
                 verbose:      bool  = True):
        self.n_episodes      = n_episodes
        self.n_candidates    = n_candidates
        self.k_select        = k_select
        self.env             = GapEnvironment(seed=seed)
        self.alpha_start     = alpha_start
        self.alpha_min       = alpha_min
        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.eps_warmup_eps  = max(1, int(eps_warmup_frac * n_episodes))
        self.model           = LinUCBDisjoint(alpha=alpha_start)
        self.verbose         = verbose
        self._rng            = random.Random(seed ^ 0xDEAD)  # separate rng for eps

    def _epsilon(self, ep: int) -> float:
        """Linearly decay epsilon"""
        if ep >= self.eps_warmup_eps:
            return self.eps_end
        frac = ep / self.eps_warmup_eps
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _alpha(self, ep: int) -> float:
        """Cosine annealing for exploration parameter"""
        frac = ep / max(self.n_episodes, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.alpha_min + cosine * (self.alpha_start - self.alpha_min)

    def _run_episode(self, ep: int) -> float:
        self.env.reseed(ep)

        candidates, ctx, contradiction_set = self.env.sample_session(
            self.n_candidates
        )
        remaining      = list(candidates)
        selected       = []
        episode_reward = 0.0

        self.model.alpha = self._alpha(ep)

        epsilon = self._epsilon(ep)

        for step in range(min(self.k_select, len(remaining))):
            if self._rng.random() < epsilon:
                best_idx = self._rng.randrange(len(remaining))
                chosen   = remaining[best_idx]
                x_chosen = build_feature_vector(chosen, selected, ctx,
                                                len(candidates))
            else:
                scores = []
                xs     = []
                for gap in remaining:
                    x = build_feature_vector(gap, selected, ctx, len(candidates))
                    s = self.model.score(gap, x)
                    scores.append(s)
                    xs.append(x)

                best_idx = int(np.argmax(scores))
                chosen   = remaining[best_idx]
                x_chosen = xs[best_idx]

            reward = compute_reward(chosen, selected, ctx, contradiction_set)
            self.model.update(chosen, x_chosen, reward)

            selected.append(chosen)
            remaining.pop(best_idx)
            episode_reward += reward

        return episode_reward / max(len(selected), 1)

    def train(self) -> LinUCBDisjoint:
        t0 = time.time()
        reward_history: list[float] = []
        log_every = max(1, self.n_episodes // 10)

        for ep in range(1, self.n_episodes + 1):
            avg_r = self._run_episode(ep)
            reward_history.append(avg_r)

            if self.verbose and ep % log_every == 0:
                window = reward_history[-log_every:]
                eps_now = self._epsilon(ep)
                alpha_now = self._alpha(ep)
                print(f"  [train] Episode {ep:>5}/{self.n_episodes}  "
                      f"avg_reward (last {log_every}): {sum(window)/len(window):.4f}  "
                      f"| α={alpha_now:.3f}  ε={eps_now:.3f}")

        elapsed = time.time() - t0
        if self.verbose:
            overall = sum(reward_history) / len(reward_history)
            late     = sum(reward_history[-500:]) / min(500, len(reward_history))
            print(f"\n  [train] Done in {elapsed:.1f}s  |  "
                  f"overall avg: {overall:.4f}  |  "
                  f"late-phase avg (last 500): {late:.4f}")
        self.model.alpha = self.alpha_min
        return self.model


# Policy Evaluation  

def evaluate_policy(model: LinUCBDisjoint,
                    n_eval_episodes: int = 200,
                    seed: int = 999) -> dict:
    """
    Evaluate trained RL policy against baselines using simulated sessions

    """
    env = GapEnvironment(seed=seed)

    rl_rewards,        random_rewards,    heuristic_rewards = [], [], []

    for _ in range(n_eval_episodes):
        candidates, ctx, contradiction_set = env.sample_session(SIM_CANDIDATES)
        k = SIM_SELECT_K

        # RL policy
        remaining  = list(candidates)
        selected   = []
        rl_r       = 0.0
        for _ in range(min(k, len(remaining))):
            scores = [
                model.score(g, build_feature_vector(g, selected, ctx,
                                                    len(candidates)))
                for g in remaining
            ]
            bi     = int(np.argmax(scores))
            chosen = remaining[bi]
            rl_r  += compute_reward(chosen, selected, ctx, contradiction_set)
            selected.append(chosen)
            remaining.pop(bi)
        rl_rewards.append(rl_r / max(len(selected), 1))

        # Random baseline
        shuffled   = list(candidates)
        random.shuffle(shuffled)
        picked_rnd = shuffled[:k]
        rnd_r      = 0.0
        for i, g in enumerate(picked_rnd):
            rnd_r += compute_reward(g, picked_rnd[:i], ctx, contradiction_set)
        random_rewards.append(rnd_r / max(len(picked_rnd), 1))

        # Heuristic baseline (current gap_detector sort)
        priority_order = {GapPriority.HIGH.value: 0,
                          GapPriority.MEDIUM.value: 1,
                          GapPriority.LOW.value: 2}
        sorted_h   = sorted(candidates,
                            key=lambda g: (priority_order.get(g["priority"], 1),
                                           -g.get("confidence", 0.5)))
        picked_h   = sorted_h[:k]
        h_r        = 0.0
        for i, g in enumerate(picked_h):
            h_r += compute_reward(g, picked_h[:i], ctx, contradiction_set)
        heuristic_rewards.append(h_r / max(len(picked_h), 1))

    results = {
        "rl_mean":        round(sum(rl_rewards)        / n_eval_episodes, 4),
        "random_mean":    round(sum(random_rewards)    / n_eval_episodes, 4),
        "heuristic_mean": round(sum(heuristic_rewards) / n_eval_episodes, 4),
        "rl_vs_random_lift":    round(
            (sum(rl_rewards) - sum(random_rewards)) / max(sum(random_rewards), 1e-9), 4),
        "rl_vs_heuristic_lift": round(
            (sum(rl_rewards) - sum(heuristic_rewards)) / max(sum(heuristic_rewards), 1e-9), 4),
        "n_eval_episodes": n_eval_episodes,
    }
    return results


#  GAP SELECTOR RL  

class GapSelectorRL:
    """
    Wraps a trained LinUCB model and exposes a select() method.
    """

    VERSION = "1.0.0"

    def __init__(self, model: LinUCBDisjoint):
        self.model = model

    # save / load model 
    
    def save(self, path: pathlib.Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":    self.VERSION,
            "saved_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model":      self.model.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  [GapSelectorRL] Model saved → {path}")

    @classmethod
    def load(cls, path: pathlib.Path = MODEL_PATH) -> "GapSelectorRL":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        model = LinUCBDisjoint.from_dict(payload["model"])
        print(f"  [GapSelectorRL] Loaded model from {path} "
              f"(saved {payload.get('saved_at', '?')})")
        return cls(model)

    @classmethod
    def load_or_train(cls,
                      path:        pathlib.Path = MODEL_PATH,
                      n_episodes:  int          = SIM_EPISODES,
                      verbose:     bool         = True) -> "GapSelectorRL":
        """
        Load existing model or train a new one
        """
        if path.exists():
            try:
                return cls.load(path)
            except Exception as e:
                print(f"  [GapSelectorRL] Load failed ({e}), retraining...")

        print(f"  [GapSelectorRL] No saved model at {path}. "
              f"Running simulation training ({n_episodes} episodes)...")
        trainer  = BanditTrainer(n_episodes=n_episodes, verbose=verbose)
        model    = trainer.train()
        selector = cls(model)

        if verbose:
            print("  [GapSelectorRL] Evaluating trained policy...")
            results = evaluate_policy(model)
            print(f"  [GapSelectorRL] Evaluation results:")
            print(f"    RL mean reward      : {results['rl_mean']}")
            print(f"    Heuristic baseline  : {results['heuristic_mean']}")
            print(f"    Random baseline     : {results['random_mean']}")
            print(f"    Lift vs heuristic   : {results['rl_vs_heuristic_lift']:+.2%}")
            print(f"    Lift vs random      : {results['rl_vs_random_lift']:+.2%}")

        selector.save(path)
        return selector


    def select(self,
               candidates: list,
               ctx: SessionContext,
               top_k: int = 15,
               contradiction_entity_set: Optional[set] = None) -> list:
        
        """Select top-k gaps using RL policy"""

        if not candidates:
            return candidates

        contradiction_set = contradiction_entity_set or set()

        def _to_dict(g) -> dict:
            if hasattr(g, "__dataclass_fields__"):
                d = asdict(g)
                for key in ("gap_type", "priority", "addressed_status"):
                    v = d.get(key)
                    if hasattr(v, "value"):
                        d[key] = v.value
                return d
            return g   

        gap_dicts   = [_to_dict(g) for g in candidates]
        originals   = list(candidates)

        remaining_idx = list(range(len(gap_dicts)))
        selected_idx  = []
        selected_dcts = []

        n_total = len(gap_dicts)

        while len(selected_idx) < min(top_k, n_total) and remaining_idx:
            scores = []
            for ri in remaining_idx:
                g = gap_dicts[ri]
                x = build_feature_vector(g, selected_dcts, ctx, n_total)
                s = self.model.score(g, x)
                scores.append((s, ri))

            _, best_ri  = max(scores, key=lambda t: t[0])
            selected_idx.append(best_ri)
            selected_dcts.append(gap_dicts[best_ri])
            remaining_idx.remove(best_ri)

        return [originals[i] for i in selected_idx]


def train_and_evaluate(n_episodes: int = SIM_EPISODES,
                       save_path:  pathlib.Path = MODEL_PATH,
                       seed:       int = 42,
                       verbose:    bool = True) -> GapSelectorRL:

    """Run simulation training and evaluation."""

    print("=" * 60)
    print(f"  Gap Selector RL  —  Simulation Training")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Candidates: {SIM_CANDIDATES} per episode")
    print(f"  Select K  : {SIM_SELECT_K}")
    print(f"  Features  : {FEATURE_DIM}")
    print(f"  Alpha (UCB): {ALPHA}")
    print("=" * 60)

    trainer  = BanditTrainer(n_episodes=n_episodes, seed=seed, verbose=verbose)
    model    = trainer.train()
    selector = GapSelectorRL(model)

    print("\n" + "-" * 60)
    print("  Evaluating on held-out simulated sessions...")
    results  = evaluate_policy(model, n_eval_episodes=500, seed=seed + 1)
    print(f"  RL mean reward      : {results['rl_mean']}")
    print(f"  Heuristic baseline  : {results['heuristic_mean']}")
    print(f"  Random baseline     : {results['random_mean']}")
    print(f"  Lift vs heuristic   : {results['rl_vs_heuristic_lift']:+.2%}")
    print(f"  Lift vs random      : {results['rl_vs_random_lift']:+.2%}")
    print("-" * 60)

    selector.save(save_path)
    return selector

 
INTEGRATION_INSTRUCTIONS = """
Integration Steps for gap_selector_rl

1. Add imports:
   from gap_selector_rl import GapSelectorRL, session_context_from_papers

2. In __init__:
   self.use_rl = True
   self._rl_selector = GapSelectorRL.load_or_train()

3. Replace ranking step:
   ctx = session_context_from_papers(papers, comp_ctx, used_gap_matrix)
   result.gaps = self._rl_selector.select(result.gaps, ctx, top_k=15)

4. Run once for training:
   python gap_selector_rl.py
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gap Selector RL — simulation training"
    )
    parser.add_argument("--episodes", type=int, default=SIM_EPISODES,
                        help=f"Training episodes (default {SIM_EPISODES})")
    parser.add_argument("--save",     type=str, default=str(MODEL_PATH),
                        help="Model save path")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--quiet",    action="store_true")
    parser.add_argument("--info",     action="store_true",
                        help="Print integration instructions and exit")
    args = parser.parse_args()

    if args.info:
        print(INTEGRATION_INSTRUCTIONS)
    else:
        train_and_evaluate(
            n_episodes = args.episodes,
            save_path  = pathlib.Path(args.save),
            seed       = args.seed,
            verbose    = not args.quiet,
        )