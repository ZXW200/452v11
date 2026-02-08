"""
Classical game theory strategies.
LLMStrategy is in llm_strategy.py.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import random

from .games import Action


class Strategy(ABC):
    """Base class. Subclasses must implement choose_action."""
    name: str = "Base Strategy"
    description: str = "Base strategy class"
    description_cn: str = "策略基类"

    @abstractmethod
    def choose_action(self,
                      history: List[Tuple[Action, Action]],
                      opponent_name: str = None) -> Action:
        """Choose action based on history [(my_action, opp_action), ...]."""
        pass

    def reset(self):
        """Reset strategy state."""
        pass


# --- Fixed Strategies ---

class AlwaysCooperate(Strategy):
    """Always cooperate."""
    name = "Always Cooperate"
    description = "Always choose to cooperate regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择合作"

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.COOPERATE


class AlwaysDefect(Strategy):
    """Always defect."""
    name = "Always Defect"
    description = "Always choose to defect regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择背叛"

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.DEFECT


class RandomStrategy(Strategy):
    """Random cooperate or defect."""
    name = "Random"
    description = "Randomly choose cooperate or defect with equal probability"
    description_cn = "以相等概率随机选择合作或背叛"

    def __init__(self, coop_prob: float = 0.5):
        self.coop_prob = coop_prob

    def choose_action(self, history, opponent_name=None) -> Action:
        return Action.COOPERATE if random.random() < self.coop_prob else Action.DEFECT


# --- Conditional Strategies ---

class TitForTat(Strategy):
    """Copy opponent's last action. Start with cooperate."""
    name = "Tit for Tat"
    description = "Start with cooperation, then copy opponent's last move"
    description_cn = "第一轮合作，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE
        # Mirror opponent's last action
        return history[-1][1]


class TitForTwoTats(Strategy):
    """Defect only after opponent defects twice in a row."""
    name = "Tit for Two Tats"
    description = "Defect only if opponent defected in last TWO rounds"
    description_cn = "只有对手连续两轮背叛才报复"

    def choose_action(self, history, opponent_name=None) -> Action:
        if len(history) < 2:
            return Action.COOPERATE
        # Check if opponent defected in last 2 rounds
        if history[-1][1] == Action.DEFECT and history[-2][1] == Action.DEFECT:
            return Action.DEFECT
        return Action.COOPERATE


class GrimTrigger(Strategy):
    """Cooperate until opponent defects once, then always defect."""
    name = "Grim Trigger"
    description = "Cooperate until opponent defects once, then always defect"
    description_cn = "开始合作，一旦对手背叛就永远背叛"

    def __init__(self):
        self.triggered = False

    def choose_action(self, history, opponent_name=None) -> Action:
        # Auto-reset for new game
        if not history and self.triggered:
            self.reset()

        if self.triggered:
            return Action.DEFECT

        # Check if opponent ever defected
        for my_action, opp_action in history:
            if opp_action == Action.DEFECT:
                self.triggered = True
                return Action.DEFECT

        return Action.COOPERATE

    def reset(self):
        self.triggered = False


class Pavlov(Strategy):
    """Win-Stay, Lose-Shift. Repeat if same actions, switch otherwise."""
    name = "Pavlov"
    description = "Repeat last action if it gave good payoff, otherwise switch"
    description_cn = "如果上一轮结果好就重复，否则改变选择"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE

        my_last, opp_last = history[-1]

        # Same actions -> keep; different -> switch
        if my_last == opp_last:
            return my_last
        else:
            return Action.COOPERATE if my_last == Action.DEFECT else Action.DEFECT


class SuspiciousTitForTat(Strategy):
    """Like TFT but starts with defect."""
    name = "Suspicious Tit for Tat"
    description = "Start with defection, then copy opponent's last move"
    description_cn = "第一轮背叛，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.DEFECT
        return history[-1][1]


class GenerousTitForTat(Strategy):
    """Like TFT but sometimes forgives defection."""
    name = "Generous Tit for Tat"
    description = "Like TFT, but forgive defection with some probability"
    description_cn = "类似以牙还牙，但有一定概率原谅背叛"

    def __init__(self, forgiveness: float = 0.1):
        self.forgiveness = forgiveness

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE
        # Forgive defection with some probability
        if history[-1][1] == Action.DEFECT:
            if random.random() < self.forgiveness:
                return Action.COOPERATE
            return Action.DEFECT
        return Action.COOPERATE


class Extort2(Strategy):
    """Zero-determinant Extort-2 strategy (Press & Dyson 2012)."""
    name = "Extort-2"
    description = "Zero-determinant strategy that extorts opponent"
    description_cn = "零行列式勒索策略，确保收益优势"

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE

        my_last, opp_last = history[-1]

        # Cooperation probability based on last round (standard PD payoffs)
        if my_last == Action.COOPERATE and opp_last == Action.COOPERATE:
            p = 8/9   # CC
        elif my_last == Action.COOPERATE and opp_last == Action.DEFECT:
            p = 1/2   # CD
        elif my_last == Action.DEFECT and opp_last == Action.COOPERATE:
            p = 1/3   # DC
        else:
            p = 0     # DD

        return Action.COOPERATE if random.random() < p else Action.DEFECT


class GradualStrategy(Strategy):
    """Punish N times for Nth defection, then cooperate twice to reconcile."""
    name = "Gradual"
    description = "Punish defection with increasing retaliation, then reconcile"
    description_cn = "对背叛进行递增报复，然后和解"

    def __init__(self):
        self.defect_count = 0
        self.punish_remaining = 0
        self.calm_remaining = 0

    def choose_action(self, history, opponent_name=None) -> Action:
        # Auto-reset for new game
        if not history and (self.defect_count > 0 or self.punish_remaining > 0 or self.calm_remaining > 0):
            self.reset()

        if self.punish_remaining > 0:
            self.punish_remaining -= 1
            return Action.DEFECT

        if self.calm_remaining > 0:
            self.calm_remaining -= 1
            return Action.COOPERATE

        if history and history[-1][1] == Action.DEFECT:
            self.defect_count += 1
            self.punish_remaining = self.defect_count - 1  # 本轮也算一次
            self.calm_remaining = 2
            return Action.DEFECT

        return Action.COOPERATE

    def reset(self):
        self.defect_count = 0
        self.punish_remaining = 0
        self.calm_remaining = 0


# --- Probability-based Strategies ---

class ProbabilisticCooperator(Strategy):
    """Match opponent's cooperation rate probabilistically."""
    name = "Probabilistic Cooperator"
    description = "Match opponent's cooperation rate probabilistically"
    description_cn = "概率性地匹配对手的合作率"

    def __init__(self, base_coop: float = 0.5):
        self.base_coop = base_coop

    def choose_action(self, history, opponent_name=None) -> Action:
        if not history:
            return Action.COOPERATE if random.random() < self.base_coop else Action.DEFECT

        # Match opponent's cooperation rate
        opp_coop_count = sum(1 for _, opp in history if opp == Action.COOPERATE)
        opp_coop_rate = opp_coop_count / len(history)

        return Action.COOPERATE if random.random() < opp_coop_rate else Action.DEFECT


# --- Strategy Registry ---

STRATEGY_REGISTRY = {
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "random": RandomStrategy,
    "tit_for_tat": TitForTat,
    "tit_for_two_tats": TitForTwoTats,
    "grim_trigger": GrimTrigger,
    "pavlov": Pavlov,
    "suspicious_tit_for_tat": SuspiciousTitForTat,
    "generous_tit_for_tat": GenerousTitForTat,
    "gradual": GradualStrategy,
    "extort2": Extort2,
    "probabilistic": ProbabilisticCooperator,
}


def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """Create a strategy instance by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[strategy_name](**kwargs)
