# Classical game theory strategies / 经典博弈论策略
# LLMStrategy is in llm_strategy.py / LLMStrategy 在 llm_strategy.py 中
from abc import ABC, abstractmethod
import random

from .games import Action


# Base class. Subclasses must implement choose_action / 基类，子类必须实现 choose_action
class Strategy(ABC):
    name: str = "Base Strategy"
    description: str = "Base strategy class"
    description_cn: str = "策略基类"

    # Choose action based on history [(my_action, opp_action), ...] / 基于历史记录选择动作
    @abstractmethod
    def choose_action(self, history, opponent_name=None):
        pass

    # Reset strategy state / 重置策略状态
    def reset(self):
        pass


# --- Fixed Strategies ---

# Always cooperate / 始终合作
class AlwaysCooperate(Strategy):
    name = "Always Cooperate"
    description = "Always choose to cooperate regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择合作"

    def choose_action(self, history, opponent_name=None):
        return Action.COOPERATE


# Always defect / 始终背叛
class AlwaysDefect(Strategy):
    name = "Always Defect"
    description = "Always choose to defect regardless of opponent's actions"
    description_cn = "无论对手如何，始终选择背叛"

    def choose_action(self, history, opponent_name=None):
        return Action.DEFECT


# Random cooperate or defect / 随机合作或背叛
class RandomStrategy(Strategy):
    name = "Random"
    description = "Randomly choose cooperate or defect with equal probability"
    description_cn = "以相等概率随机选择合作或背叛"

    def __init__(self, coop_prob=0.5):
        self.coop_prob = coop_prob

    def choose_action(self, history, opponent_name=None):
        return Action.COOPERATE if random.random() < self.coop_prob else Action.DEFECT


# --- Conditional Strategies ---

# Copy opponent's last action. Start with cooperate / 模仿对手上一轮动作，第一轮合作
class TitForTat(Strategy):
    name = "Tit for Tat"
    description = "Start with cooperation, then copy opponent's last move"
    description_cn = "第一轮合作，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None):
        if not history:
            return Action.COOPERATE
        # Mirror opponent's last action
        return history[-1][1]


# Defect only after opponent defects twice in a row / 对手连续两轮背叛才报复
class TitForTwoTats(Strategy):
    name = "Tit for Two Tats"
    description = "Defect only if opponent defected in last TWO rounds"
    description_cn = "只有对手连续两轮背叛才报复"

    def choose_action(self, history, opponent_name=None):
        if len(history) < 2:
            return Action.COOPERATE
        # Check if opponent defected in last 2 rounds
        if history[-1][1] == Action.DEFECT and history[-2][1] == Action.DEFECT:
            return Action.DEFECT
        return Action.COOPERATE


# Cooperate until opponent defects once, then always defect / 开始合作，一旦对手背叛就永远背叛
class GrimTrigger(Strategy):
    name = "Grim Trigger"
    description = "Cooperate until opponent defects once, then always defect"
    description_cn = "开始合作，一旦对手背叛就永远背叛"

    # Initialize grim trigger state / 初始化冷酷触发状态
    def __init__(self):
        self.triggered = False

    def choose_action(self, history, opponent_name=None):
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

    # Reset triggered state / 重置触发状态
    def reset(self):
        self.triggered = False


# Win-Stay, Lose-Shift. Repeat if same actions, switch otherwise / 赢则保持，输则改变
class Pavlov(Strategy):
    name = "Pavlov"
    description = "Repeat last action if it gave good payoff, otherwise switch"
    description_cn = "如果上一轮结果好就重复，否则改变选择"

    def choose_action(self, history, opponent_name=None):
        if not history:
            return Action.COOPERATE

        my_last, opp_last = history[-1]

        # Same actions -> keep; different -> switch
        if my_last == opp_last:
            return my_last
        else:
            return Action.COOPERATE if my_last == Action.DEFECT else Action.DEFECT


# Like TFT but starts with defect / 类似以牙还牙但第一轮背叛
class SuspiciousTitForTat(Strategy):
    name = "Suspicious Tit for Tat"
    description = "Start with defection, then copy opponent's last move"
    description_cn = "第一轮背叛，之后模仿对手上一轮的动作"

    def choose_action(self, history, opponent_name=None):
        if not history:
            return Action.DEFECT
        return history[-1][1]


# Like TFT but sometimes forgives defection / 类似以牙还牙，但有概率原谅背叛
class GenerousTitForTat(Strategy):
    name = "Generous Tit for Tat"
    description = "Like TFT, but forgive defection with some probability"
    description_cn = "类似以牙还牙，但有一定概率原谅背叛"

    def __init__(self, forgiveness=0.1):
        self.forgiveness = forgiveness

    def choose_action(self, history, opponent_name=None):
        if not history:
            return Action.COOPERATE
        # Forgive defection with some probability
        if history[-1][1] == Action.DEFECT:
            if random.random() < self.forgiveness:
                return Action.COOPERATE
            return Action.DEFECT
        return Action.COOPERATE


# Zero-determinant Extort-2 strategy (Press & Dyson 2012) / 零行列式勒索策略
class Extort2(Strategy):
    name = "Extort-2"
    description = "Zero-determinant strategy that extorts opponent"
    description_cn = "零行列式勒索策略，确保收益优势"

    def choose_action(self, history, opponent_name=None):
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


# Punish N times for Nth defection, then cooperate twice to reconcile / 对第N次背叛惩罚N次，然后合作两次和解
class GradualStrategy(Strategy):
    name = "Gradual"
    description = "Punish defection with increasing retaliation, then reconcile"
    description_cn = "对背叛进行递增报复，然后和解"

    # Initialize gradual strategy counters / 初始化渐进策略计数器
    def __init__(self):
        self.defect_count = 0
        self.punish_remaining = 0
        self.calm_remaining = 0

    def choose_action(self, history, opponent_name=None):
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
            self.punish_remaining = self.defect_count - 1  # This round counts as one / 本轮也算一次
            self.calm_remaining = 2
            return Action.DEFECT

        return Action.COOPERATE

    # Reset all counters / 重置所有计数器
    def reset(self):
        self.defect_count = 0
        self.punish_remaining = 0
        self.calm_remaining = 0


# --- Probability-based Strategies ---

# Match opponent's cooperation rate probabilistically / 概率性地匹配对手的合作率
class ProbabilisticCooperator(Strategy):
    name = "Probabilistic Cooperator"
    description = "Match opponent's cooperation rate probabilistically"
    description_cn = "概率性地匹配对手的合作率"

    def __init__(self, base_coop=0.5):
        self.base_coop = base_coop

    def choose_action(self, history, opponent_name=None):
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


# Create a strategy instance by name / 根据名称创建策略实例
def create_strategy(strategy_name, **kwargs):
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(STRATEGY_REGISTRY.keys())}")

    return STRATEGY_REGISTRY[strategy_name](**kwargs)
