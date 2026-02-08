# Network topologies for agent interactions / 智能体交互的网络拓扑
# Optional dependency: networkx (for SmallWorld and ScaleFree) / 可选依赖：networkx（用于小世界和无标度网络）
import random
from abc import ABC, abstractmethod

# Optional networkx import
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using simple network implementation.")


# Base class for interaction networks / 交互网络基类
class InteractionNetwork(ABC):

    # Initialize with agent list / 使用智能体列表初始化
    def __init__(self, agents):
        self.agents = agents
        self.n = len(agents)
        self._edges = []
        self._adjacency = {a: [] for a in agents}

    # Build the network edges / 构建网络边
    @abstractmethod
    def _build_network(self):
        pass

    # Add undirected edge / 添加无向边
    def _add_edge(self, a1, a2):
        if a1 != a2 and a2 not in self._adjacency[a1]:
            self._edges.append((a1, a2))
            self._adjacency[a1].append(a2)
            self._adjacency[a2].append(a1)

    # Get agent's neighbors / 获取智能体的邻居
    def get_neighbors(self, agent):
        return self._adjacency.get(agent, [])

    # Get all edges / 获取所有边
    def get_all_edges(self):
        return self._edges.copy()

    # Get interaction pairs for this round. Subclasses can override / 获取本轮交互对，子类可重写
    def get_interaction_pairs(self):
        return self.get_all_edges()

    # Get number of neighbors / 获取邻居数量
    def get_degree(self, agent):
        return len(self._adjacency.get(agent, []))

    # Get network statistics / 获取网络统计数据
    def get_network_stats(self):
        degrees = [self.get_degree(a) for a in self.agents]
        return {
            "num_agents": self.n,
            "num_edges": len(self._edges),
            "avg_degree": sum(degrees) / self.n if self.n > 0 else 0,
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }

    # Export as dictionary / 导出为字典
    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "agents": self.agents,
            "edges": self._edges,
            "adjacency": self._adjacency,
        }


# --- Network Implementations ---

# All agents interact with each other / 所有智能体互相交互
class FullyConnectedNetwork(InteractionNetwork):
    name = "Fully Connected"

    def __init__(self, agents):
        super().__init__(agents)
        self._build_network()

    # Build fully connected edges / 构建全连接边
    def _build_network(self):
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i+1:]:
                self._add_edge(a1, a2)


# Each agent interacts with two neighbors in a ring / 每个智能体与环上的两个邻居交互
class RingNetwork(InteractionNetwork):
    name = "Ring"

    def __init__(self, agents):
        super().__init__(agents)
        self._build_network()

    # Build ring edges / 构建环形边
    def _build_network(self):
        for i in range(self.n):
            self._add_edge(self.agents[i], self.agents[(i+1) % self.n])


# 2D grid structure / 二维网格结构
class GridNetwork(InteractionNetwork):
    name = "Grid"

    def __init__(self, agents, cols=None):
        self.cols = cols or int(len(agents) ** 0.5)
        self.rows = (len(agents) + self.cols - 1) // self.cols
        super().__init__(agents)
        self._build_network()

    # Build grid edges / 构建网格边
    def _build_network(self):
        for i, agent in enumerate(self.agents):
            row, col = i // self.cols, i % self.cols

            if col < self.cols - 1 and i + 1 < self.n:
                self._add_edge(agent, self.agents[i + 1])

            if row < self.rows - 1 and i + self.cols < self.n:
                self._add_edge(agent, self.agents[i + self.cols])


# One center node connected to all others / 一个中心节点连接所有其他节点
class StarNetwork(InteractionNetwork):
    name = "Star"

    def __init__(self, agents, center=None):
        self.center = center or agents[0]
        super().__init__(agents)
        self._build_network()

    # Build star edges from center / 从中心构建星形边
    def _build_network(self):
        for agent in self.agents:
            if agent != self.center:
                self._add_edge(self.center, agent)


# Watts-Strogatz small world network / Watts-Strogatz小世界网络
class SmallWorldNetwork(InteractionNetwork):
    name = "Small World"

    def __init__(self, agents, k=4, p=0.3):
        self.k = min(k, len(agents) - 1)
        if self.k % 2 == 1:
            self.k -= 1
        self.p = p
        super().__init__(agents)
        self._build_network()

    # Build small world network with rewiring / 构建带重连的小世界网络
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


# Barabasi-Albert preferential attachment network / Barabasi-Albert优先连接网络
class ScaleFreeNetwork(InteractionNetwork):
    name = "Scale Free"

    def __init__(self, agents, m=2):
        self.m = min(m, len(agents) - 1)
        super().__init__(agents)
        self._build_network()

    # Build scale-free network with preferential attachment / 构建带优先连接的无标度网络
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


# Erdos-Renyi random network / Erdos-Renyi随机网络
class RandomNetwork(InteractionNetwork):
    name = "Random (ER)"

    def __init__(self, agents, p=0.3):
        self.p = p
        super().__init__(agents)
        self._build_network()

    # Build random edges with probability p / 以概率p构建随机边
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


# Create a network by type name / 根据类型名称创建网络
def create_network(network_type, agents, **kwargs):
    if network_type not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network type: {network_type}. "
                        f"Available: {list(NETWORK_REGISTRY.keys())}")

    return NETWORK_REGISTRY[network_type](agents, **kwargs)


# --- Visualization ---

# Visualize network structure using matplotlib / 使用matplotlib可视化网络结构
def visualize_network(network, output_path=None):
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
