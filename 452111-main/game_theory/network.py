"""
Network topologies for agent interactions.
Optional dependency: networkx (for SmallWorld and ScaleFree).
"""
import random
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

# Optional networkx import
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using simple network implementation.")


class InteractionNetwork(ABC):
    """Base class for interaction networks."""

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.n = len(agents)
        self._edges: List[Tuple[str, str]] = []
        self._adjacency: Dict[str, List[str]] = {a: [] for a in agents}
    
    @abstractmethod
    def _build_network(self):
        """Build the network edges."""
        pass
    
    def _add_edge(self, a1: str, a2: str):
        """Add undirected edge."""
        if a1 != a2 and a2 not in self._adjacency[a1]:
            self._edges.append((a1, a2))
            self._adjacency[a1].append(a2)
            self._adjacency[a2].append(a1)
    
    def get_neighbors(self, agent: str) -> List[str]:
        """Get agent's neighbors."""
        return self._adjacency.get(agent, [])
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges."""
        return self._edges.copy()
    
    def get_interaction_pairs(self) -> List[Tuple[str, str]]:
        """Get interaction pairs for this round. Subclasses can override."""
        return self.get_all_edges()
    
    def get_degree(self, agent: str) -> int:
        """Get number of neighbors."""
        return len(self._adjacency.get(agent, []))
    
    def get_network_stats(self) -> Dict:
        """Get network statistics."""
        degrees = [self.get_degree(a) for a in self.agents]
        return {
            "num_agents": self.n,
            "num_edges": len(self._edges),
            "avg_degree": sum(degrees) / self.n if self.n > 0 else 0,
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "type": self.__class__.__name__,
            "agents": self.agents,
            "edges": self._edges,
            "adjacency": self._adjacency,
        }


# --- Network Implementations ---

class FullyConnectedNetwork(InteractionNetwork):
    """All agents interact with each other."""
    name = "Fully Connected"
    
    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i+1:]:
                self._add_edge(a1, a2)


class RingNetwork(InteractionNetwork):
    """Each agent interacts with two neighbors in a ring."""
    name = "Ring"
    
    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i in range(self.n):
            self._add_edge(self.agents[i], self.agents[(i+1) % self.n])


class GridNetwork(InteractionNetwork):
    """2D grid structure."""
    name = "Grid"
    
    def __init__(self, agents: List[str], cols: int = None):
        self.cols = cols or int(len(agents) ** 0.5)
        self.rows = (len(agents) + self.cols - 1) // self.cols
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, agent in enumerate(self.agents):
            row, col = i // self.cols, i % self.cols
            
            if col < self.cols - 1 and i + 1 < self.n:
                self._add_edge(agent, self.agents[i + 1])

            if row < self.rows - 1 and i + self.cols < self.n:
                self._add_edge(agent, self.agents[i + self.cols])


class StarNetwork(InteractionNetwork):
    """One center node connected to all others."""
    name = "Star"
    
    def __init__(self, agents: List[str], center: str = None):
        self.center = center or agents[0]
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for agent in self.agents:
            if agent != self.center:
                self._add_edge(self.center, agent)


class SmallWorldNetwork(InteractionNetwork):
    """Watts-Strogatz small world network."""
    name = "Small World"
    
    def __init__(self, agents: List[str], k: int = 4, p: float = 0.3):
        self.k = min(k, len(agents) - 1)
        if self.k % 2 == 1:
            self.k -= 1
        self.p = p
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        if HAS_NETWORKX:
            G = nx.watts_strogatz_graph(self.n, self.k, self.p)
            for i, j in G.edges():
                self._add_edge(self.agents[i], self.agents[j])
        else:
            # Fallback: ring + random rewiring
            for i in range(self.n):
                for j in range(1, self.k // 2 + 1):
                    self._add_edge(self.agents[i], self.agents[(i + j) % self.n])
            
            new_edges = []
            for a1, a2 in self._edges:
                if random.random() < self.p:
                    candidates = [a for a in self.agents if a != a1 and a not in self._adjacency[a1]]
                    if candidates:
                        new_a2 = random.choice(candidates)
                        new_edges.append((a1, new_a2))
                    else:
                        new_edges.append((a1, a2))
                else:
                    new_edges.append((a1, a2))
            
            self._edges = []
            self._adjacency = {a: [] for a in self.agents}
            for a1, a2 in new_edges:
                self._add_edge(a1, a2)


class ScaleFreeNetwork(InteractionNetwork):
    """Barabasi-Albert preferential attachment network."""
    name = "Scale Free"
    
    def __init__(self, agents: List[str], m: int = 2):
        self.m = min(m, len(agents) - 1)
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        if HAS_NETWORKX:
            G = nx.barabasi_albert_graph(self.n, self.m)
            for i, j in G.edges():
                self._add_edge(self.agents[i], self.agents[j])
        else:
            # Fallback: preferential attachment
            for i in range(self.m + 1):
                for j in range(i + 1, self.m + 1):
                    self._add_edge(self.agents[i], self.agents[j])
            
            # Add nodes one by one
            for i in range(self.m + 1, self.n):
                new_agent = self.agents[i]
                # Connect based on degree
                total_degree = sum(self.get_degree(a) for a in self.agents[:i])
                if total_degree == 0:
                    targets = random.sample(self.agents[:i], min(self.m, i))
                else:
                    probs = [self.get_degree(a) / total_degree for a in self.agents[:i]]
                    targets = []
                    candidates = list(self.agents[:i])
                    for _ in range(min(self.m, i)):
                        if not candidates:
                            break
                        # Weighted random selection
                        r = random.random()
                        cumsum = 0
                        for idx, (cand, prob) in enumerate(zip(candidates, probs)):
                            cumsum += prob
                            if r <= cumsum:
                                targets.append(cand)
                                candidates.pop(idx)
                                probs.pop(idx)
                                # Renormalize
                                if probs:
                                    s = sum(probs)
                                    probs = [p/s for p in probs]
                                break
                
                for target in targets:
                    self._add_edge(new_agent, target)


class RandomNetwork(InteractionNetwork):
    """Erdos-Renyi random network."""
    name = "Random (ER)"
    
    def __init__(self, agents: List[str], p: float = 0.3):
        self.p = p
        super().__init__(agents)
        self._build_network()
    
    def _build_network(self):
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i+1:]:
                if random.random() < self.p:
                    self._add_edge(a1, a2)


# --- Network Registry ---

NETWORK_REGISTRY = {
    "fully_connected": FullyConnectedNetwork,
    "ring": RingNetwork,
    "grid": GridNetwork,
    "star": StarNetwork,
    "small_world": SmallWorldNetwork,
    "scale_free": ScaleFreeNetwork,
    "random": RandomNetwork,
}


def create_network(network_type: str, agents: List[str], **kwargs) -> InteractionNetwork:
    """Create a network by type name."""
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. "
                        f"Available: {list(NETWORK_REGISTRY.keys())}")
    
    return NETWORK_REGISTRY[network_type](agents, **kwargs)


# --- Visualization ---

def visualize_network(network: InteractionNetwork, output_path: str = None):
    """Visualize network structure using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, cannot visualize")
        return
    
    if HAS_NETWORKX:
        G = nx.Graph()
        G.add_nodes_from(network.agents)
        G.add_edges_from(network.get_all_edges())
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=500,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)
        
        plt.title(f"{network.__class__.__name__}\n"
                 f"Nodes: {network.n}, Edges: {len(network._edges)}")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Network visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    else:
        print("networkx not installed, cannot visualize")
