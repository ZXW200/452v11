# Game definitions and payoff matrices / 游戏定义和收益矩阵
# Supports: Prisoner's Dilemma, Snowdrift, Stag Hunt / 支持：囚徒困境、雪堆博弈、猎鹿博弈
from enum import Enum
from dataclasses import dataclass


# Game actions: cooperate or defect / 游戏动作：合作或背叛
class Action(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"


# Game type config: name, payoff matrix, and description / 游戏类型配置：名称、收益矩阵和描述
@dataclass
class GameConfig:
    name: str
    payoff_matrix: dict
    description: str
    description_cn: str


# --- Game Definitions ---

# Prisoner's Dilemma: T=5, R=3, P=1, S=0
PRISONERS_DILEMMA = GameConfig(
    name="Prisoner's Dilemma",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (3, 3),   # R, R
        (Action.COOPERATE, Action.DEFECT): (0, 5),      # S, T
        (Action.DEFECT, Action.COOPERATE): (5, 0),      # T, S
        (Action.DEFECT, Action.DEFECT): (1, 1),         # P, P
    },
    description="Classic Prisoner's Dilemma: Mutual cooperation yields moderate reward, "
                "but defection tempts with higher individual payoff.",
    description_cn="经典囚徒困境：双方合作获得中等收益，但背叛诱惑着更高的个人收益。"
)

# Snowdrift / Hawk-Dove: T > R > S > P (mutual defection is worst)
SNOWDRIFT = GameConfig(
    name="Snowdrift Game",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (3, 3),
        (Action.COOPERATE, Action.DEFECT): (1, 5),
        (Action.DEFECT, Action.COOPERATE): (5, 1),
        (Action.DEFECT, Action.DEFECT): (0, 0),
    },
    description="Snowdrift/Hawk-Dove: Unlike PD, mutual defection is worst outcome. "
                "Better to cooperate if opponent defects.",
    description_cn="雪堆博弈：与囚徒困境不同，双方背叛是最差结果。如果对手背叛，合作反而更好。"
)

# Stag Hunt: R > T > P > S (cooperation needs mutual trust)
STAG_HUNT = GameConfig(
    name="Stag Hunt",
    payoff_matrix={
        (Action.COOPERATE, Action.COOPERATE): (5, 5),
        (Action.COOPERATE, Action.DEFECT): (0, 3),
        (Action.DEFECT, Action.COOPERATE): (3, 0),
        (Action.DEFECT, Action.DEFECT): (2, 2),
    },
    description="Stag Hunt: Cooperation yields highest reward but requires trust. "
                "Safe defection gives guaranteed but lower payoff.",
    description_cn="猎鹿博弈：合作产生最高收益但需要信任。安全的背叛给出有保障但较低的收益。"
)


# --- Game Registry ---
GAME_REGISTRY = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "snowdrift": SNOWDRIFT,
    "stag_hunt": STAG_HUNT,
}


# --- Utility Functions ---

# Return (payoff1, payoff2) for given actions / 返回给定动作的(收益1, 收益2)
def get_payoff(game, action1, action2):
    return game.payoff_matrix[(action1, action2)]


# Generate text description of payoff matrix for LLM prompts / 生成收益矩阵的文本描述用于LLM提示
def get_payoff_description(game):
    cc = game.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
    cd = game.payoff_matrix[(Action.COOPERATE, Action.DEFECT)]
    dc = game.payoff_matrix[(Action.DEFECT, Action.COOPERATE)]
    dd = game.payoff_matrix[(Action.DEFECT, Action.DEFECT)]

    return f"""Payoff Matrix for {game.name}:
- Both Cooperate: You get {cc[0]}, Opponent gets {cc[1]}
- You Cooperate, Opponent Defects: You get {cd[0]}, Opponent gets {cd[1]}
- You Defect, Opponent Cooperates: You get {dc[0]}, Opponent gets {dc[1]}
- Both Defect: You get {dd[0]}, Opponent gets {dd[1]}"""


# Parse action from string (cooperate/defect) / 从字符串解析动作（合作/背叛）
def action_from_string(s):
    s = s.lower().strip()
    if s in ["cooperate", "c", "合作"]:
        return Action.COOPERATE
    elif s in ["defect", "d", "背叛", "叛变"]:
        return Action.DEFECT
    else:
        raise ValueError(f"Unknown action: {s}")
