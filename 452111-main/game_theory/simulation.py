"""
Game simulation engine. Runs multi-agent games with parallel LLM API calls.
"""
import json
import os
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field

# Thread pool size for parallel API calls
MAX_API_WORKERS = 100

from .games import GameConfig, Action, get_payoff, PRISONERS_DILEMMA
from .network import InteractionNetwork, FullyConnectedNetwork
from .strategies import Strategy, create_strategy


@dataclass
class AgentState:
    """Agent state: tracks strategy, history, and payoff."""
    name: str
    strategy: Strategy
    description: str = ""
    personality: str = ""
    
    game_history: List[Dict] = field(default_factory=list)
    total_payoff: float = 0.0
    opponent_models: Dict[str, str] = field(default_factory=dict)
    
    def record_game(self, opponent: str, my_action: Action, opp_action: Action, payoff: float):
        """Record one game round."""
        self.game_history.append({
            "round": len(self.game_history) + 1,
            "opponent": opponent,
            "my_action": my_action.value,
            "opp_action": opp_action.value,
            "payoff": payoff,
        })
        self.total_payoff += payoff
    
    def get_history_with(self, opponent: str) -> List[Tuple[Action, Action]]:
        """Get history of games with a specific opponent."""
        history = []
        for g in self.game_history:
            if g["opponent"] == opponent:
                my_act = Action(g["my_action"])
                opp_act = Action(g["opp_action"])
                history.append((my_act, opp_act))
        return history
    
    def get_cooperation_rate(self) -> float:
        """Calculate cooperation rate across all games."""
        if not self.game_history:
            return 0.0
        coop_count = sum(1 for g in self.game_history if g["my_action"] == "cooperate")
        return coop_count / len(self.game_history)
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "strategy": self.strategy.name,
            "description": self.description,
            "total_payoff": self.total_payoff,
            "cooperation_rate": self.get_cooperation_rate(),
            "game_history": self.game_history,
        }


class GameSimulation:
    """Main game simulation. Runs agents through rounds of a game."""

    def __init__(self,
                 agents: Dict[str, AgentState],
                 game_config: GameConfig,
                 network: InteractionNetwork,
                 rounds: int = 100,
                 verbose: bool = True):
        self.agents = agents
        self.game_config = game_config
        self.network = network
        self.total_rounds = rounds
        self.verbose = verbose
        
        self.current_round = 0
        self.round_results: List[Dict] = []
        
    def run(self,
            round_callback: Callable = None) -> Dict:
        """Run the full simulation and return results."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Game Theory Simulation")
            print(f"{'='*60}")
            print(f"Game: {self.game_config.name}")
            print(f"Agents: {list(self.agents.keys())}")
            print(f"Network: {self.network.__class__.__name__}")
            print(f"Total Rounds: {self.total_rounds}")
            print(f"{'='*60}\n")
        
        for round_num in range(1, self.total_rounds + 1):
            self.current_round = round_num
            round_data = self._run_single_round()
            self.round_results.append(round_data)
            
            if round_callback:
                round_callback(round_num, round_data)

            if self.verbose and round_num % 10 == 0:
                self._print_progress(round_num)
        
        results = self._compile_results()
        
        if self.verbose:
            self._print_final_results(results)
        
        return results
    
    def _run_single_round(self) -> Dict:
        """Run one round with parallel action decisions."""
        round_data = {
            "round": self.current_round,
            "interactions": [],
            "round_payoffs": {name: 0.0 for name in self.agents}
        }

        pairs = self.network.get_interaction_pairs()

        decision_tasks = []
        for agent1_name, agent2_name in pairs:
            agent1 = self.agents[agent1_name]
            agent2 = self.agents[agent2_name]

            history1 = agent1.get_history_with(agent2_name)
            history2 = agent2.get_history_with(agent1_name)

            decision_tasks.append((agent1.strategy, history1, agent2_name))
            decision_tasks.append((agent2.strategy, history2, agent1_name))

        def execute_decision(task):
            strategy, history, opponent_name = task
            return strategy.choose_action(history, opponent_name)

        # Parallel execution of all decisions
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
            actions = list(executor.map(execute_decision, decision_tasks))

        # Match actions back to pairs
        for i, (agent1_name, agent2_name) in enumerate(pairs):
            agent1 = self.agents[agent1_name]
            agent2 = self.agents[agent2_name]

            action1 = actions[i * 2]
            action2 = actions[i * 2 + 1]

            payoff1, payoff2 = get_payoff(self.game_config, action1, action2)

            agent1.record_game(agent2_name, action1, action2, payoff1)
            agent2.record_game(agent1_name, action2, action1, payoff2)

            if hasattr(agent1.strategy, 'update_payoff'):
                agent1.strategy.update_payoff(payoff1)
            if hasattr(agent2.strategy, 'update_payoff'):
                agent2.strategy.update_payoff(payoff2)

            round_data["interactions"].append({
                "agent1": agent1_name,
                "agent2": agent2_name,
                "action1": action1.value,
                "action2": action2.value,
                "payoff1": payoff1,
                "payoff2": payoff2,
            })

            round_data["round_payoffs"][agent1_name] += payoff1
            round_data["round_payoffs"][agent2_name] += payoff2

        return round_data
    
    
    def _compile_results(self) -> Dict:
        """Compile final simulation results."""
        final_payoffs = {
            name: agent.total_payoff 
            for name, agent in self.agents.items()
        }
        
        cooperation_rates = {
            name: agent.get_cooperation_rate()
            for name, agent in self.agents.items()
        }
        
        # Cooperation rate over time
        cooperation_evolution = []
        for round_data in self.round_results:
            total_coop = 0
            total_actions = 0
            for interaction in round_data["interactions"]:
                if interaction["action1"] == "cooperate":
                    total_coop += 1
                if interaction["action2"] == "cooperate":
                    total_coop += 1
                total_actions += 2
            
            rate = total_coop / total_actions if total_actions > 0 else 0
            cooperation_evolution.append(rate)
        
        return {
            "config": {
                "game": self.game_config.name,
                "network": self.network.__class__.__name__,
                "total_rounds": self.total_rounds,
                "num_agents": len(self.agents),
            },
            "final_payoffs": final_payoffs,
            "cooperation_rates": cooperation_rates,
            "cooperation_evolution": cooperation_evolution,
            "agent_details": {
                name: agent.to_dict() 
                for name, agent in self.agents.items()
            },
            "round_history": self.round_results,
        }
    
    def _print_progress(self, round_num: int):
        """Print round progress."""
        payoffs = [(name, agent.total_payoff) for name, agent in self.agents.items()]
        payoffs.sort(key=lambda x: x[1], reverse=True)
        
        # Current cooperation rate
        last_round = self.round_results[-1]
        coop_count = sum(
            (1 if i["action1"] == "cooperate" else 0) + 
            (1 if i["action2"] == "cooperate" else 0)
            for i in last_round["interactions"]
        )
        total_actions = len(last_round["interactions"]) * 2
        coop_rate = coop_count / total_actions if total_actions > 0 else 0
        
        print(f"Round {round_num:4d} | Coop Rate: {coop_rate:.1%} | "
              f"Top: {payoffs[0][0]}({payoffs[0][1]:.1f})")
    
    def _print_final_results(self, results: Dict):
        """Print final rankings and stats."""
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        
        print("\nFinal Rankings:")
        payoffs = list(results["final_payoffs"].items())
        payoffs.sort(key=lambda x: x[1], reverse=True)
        for i, (name, payoff) in enumerate(payoffs, 1):
            coop_rate = results["cooperation_rates"][name]
            strategy = self.agents[name].strategy.name
            print(f"  {i}. {name:15s} | Payoff: {payoff:7.1f} | "
                  f"Coop: {coop_rate:.1%} | Strategy: {strategy}")
        
        print(f"\nOverall Cooperation Rate: "
              f"{sum(results['cooperation_rates'].values())/len(results['cooperation_rates']):.1%}")
    
    def save_results(self, output_dir: str = "experiments/results") -> str:
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/sim_{self.game_config.name.replace(' ', '_')}_{timestamp}.json"
        
        results = self._compile_results()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filename}")
        return filename


# --- Helper Functions ---

def create_simulation(
    num_agents: int = 5,
    strategy_config: Dict[str, str] = None,
    game_type: str = "prisoners_dilemma",
    network_type: str = "fully_connected",
    rounds: int = 100,
    **kwargs
) -> GameSimulation:
    """Create a simulation with given agents and game type."""
    from .games import GAME_REGISTRY
    from .network import create_network
    
    agent_names = [f"Agent_{i}" for i in range(num_agents)]

    if strategy_config is None:
        strategy_config = {name: "tit_for_tat" for name in agent_names}
    
    agents = {}
    for name in agent_names:
        strategy_name = strategy_config.get(name, "tit_for_tat")
        strategy = create_strategy(strategy_name)
        agents[name] = AgentState(
            name=name,
            strategy=strategy,
            description=f"Agent using {strategy_name} strategy",
        )
    
    game_config = GAME_REGISTRY.get(game_type, PRISONERS_DILEMMA)
    network = create_network(network_type, agent_names, **kwargs)
    
    return GameSimulation(
        agents=agents,
        game_config=game_config,
        network=network,
        rounds=rounds,
    )


def run_quick_experiment(
    strategies: List[str] = None,
    game_type: str = "prisoners_dilemma",
    network_type: str = "fully_connected",
    rounds: int = 50,
    verbose: bool = True,
) -> Dict:
    """Run a quick experiment with given strategies."""
    if strategies is None:
        strategies = ["tit_for_tat", "always_cooperate", "always_defect", "random", "pavlov"]
    
    num_agents = len(strategies)
    agent_names = [f"Agent_{i}" for i in range(num_agents)]
    strategy_config = dict(zip(agent_names, strategies))
    
    sim = create_simulation(
        num_agents=num_agents,
        strategy_config=strategy_config,
        game_type=game_type,
        network_type=network_type,
        rounds=rounds,
    )
    sim.verbose = verbose
    
    return sim.run()
