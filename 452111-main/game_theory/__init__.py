"""
Game theory multi-agent simulation package.
LLMStrategy: from game_theory.llm_strategy import LLMStrategy
"""

from .games import (
    Action,
    GameConfig,
    PRISONERS_DILEMMA,
    SNOWDRIFT,
    STAG_HUNT,
    GAME_REGISTRY,
    get_payoff,
    get_payoff_description,
    action_from_string,
)

from .strategies import (
    Strategy,
    AlwaysCooperate,
    AlwaysDefect,
    RandomStrategy,
    TitForTat,
    TitForTwoTats,
    GrimTrigger,
    Pavlov,
    SuspiciousTitForTat,
    GradualStrategy,
    ProbabilisticCooperator,
    STRATEGY_REGISTRY,
    create_strategy,
)

from .network import (
    InteractionNetwork,
    FullyConnectedNetwork,
    RingNetwork,
    GridNetwork,
    StarNetwork,
    SmallWorldNetwork,
    ScaleFreeNetwork,
    RandomNetwork,
    NETWORK_REGISTRY,
    create_network,
    visualize_network,
)

from .simulation import (
    AgentState,
    GameSimulation,
    create_simulation,
    run_quick_experiment,
)

__version__ = "0.6.0"