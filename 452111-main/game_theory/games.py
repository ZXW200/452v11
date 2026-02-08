"""
Game definitions and payoff matrices.
Supports: Prisoner's Dilemma, Snowdrift, Stag Hunt.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple


class Action(Enum):
    """Game actions: cooperate or defect."""
    COOPERATE = "cooperate"
    DEFECT = "defect"


@dataclass
class GameConfig:
    """Game type config: name, payoff matrix, and description."""
    name: str
    payoff_matrix: Dict[Tuple[Action, Action], Tuple[float, float]]
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

def get_payoff(game: GameConfig, action1: Action, action2: Action) -> Tuple[float, float]:
    """Return (payoff1, payoff2) for given actions."""
    return game.payoff_matrix[(action1, action2)]


def get_payoff_description(game: GameConfig) -> str:
    """Generate text description of payoff matrix for LLM prompts."""
    cc = game.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
    cd = game.payoff_matrix[(Action.COOPERATE, Action.DEFECT)]
    dc = game.payoff_matrix[(Action.DEFECT, Action.COOPERATE)]
    dd = game.payoff_matrix[(Action.DEFECT, Action.DEFECT)]
    
    return f"""Payoff Matrix for {game.name}:
- Both Cooperate: You get {cc[0]}, Opponent gets {cc[1]}
- You Cooperate, Opponent Defects: You get {cd[0]}, Opponent gets {cd[1]}
- You Defect, Opponent Cooperates: You get {dc[0]}, Opponent gets {dc[1]}
- Both Defect: You get {dd[0]}, Opponent gets {dd[1]}"""


def action_from_string(s: str) -> Action:
    """Parse action from string (cooperate/defect)."""
    s = s.lower().strip()
    if s in ["cooperate", "c", "合作"]:
        return Action.COOPERATE
    elif s in ["defect", "d", "背叛", "叛变"]:
        return Action.DEFECT
    else:
        raise ValueError(f"Unknown action: {s}")
