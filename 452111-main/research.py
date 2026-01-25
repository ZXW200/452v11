"""
博弈论 LLM 多智能体研究实验脚本 v10
Game Theory LLM Multi-Agent Research Experiments

实验列表 (使用 python research.py <exp_name> 运行):
  exp1  - Pure vs Hybrid - LLM自己分析 vs 代码辅助
  exp2  - 记忆视窗对比 - 5/10/20/全部历史
  exp3  - 多LLM对比 - DeepSeek vs GPT vs Gemini
  exp4  - Cheap Talk 三方对战 - 3个LLM Round-Robin 语言交流博弈
  exp4b - Cheap Talk 一对一 - 指定双方LLM的语言交流博弈
  exp5  - 群体动力学 (单Provider)
  exp5b - 群体动力学 (多Provider)
  exp6  - Baseline 对比 - LLM vs 经典策略

所有实验默认遍历三种博弈: 囚徒困境 / 雪堆博弈 / 猎鹿博弈
结果按 results/{时间戳}/{博弈类型}/ 分目录保存
"""

import json
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 博弈论模块导入
from game_theory.games import (
    PRISONERS_DILEMMA, SNOWDRIFT, STAG_HUNT,
    Action, GameConfig, get_payoff, get_payoff_description, GAME_REGISTRY
)
from game_theory.llm_strategy import LLMStrategy
from game_theory.strategies import (
    TitForTat, AlwaysCooperate, AlwaysDefect,
    GrimTrigger, Pavlov, RandomStrategy
)
from game_theory.network import (
    FullyConnectedNetwork, SmallWorldNetwork, ScaleFreeNetwork, NETWORK_REGISTRY
)
from game_theory.simulation import AgentState, GameSimulation


# ============================================================
# 全局配置
# ============================================================

GAME_NAMES_CN = {
    "prisoners_dilemma": "囚徒困境",
    "snowdrift": "雪堆博弈",
    "stag_hunt": "猎鹿博弈",
}

NETWORK_NAMES_CN = {
    "fully_connected": "完全连接",
    "small_world": "小世界",
    "scale_free": "无标度",
}

# 默认实验参数
DEFAULT_CONFIG = {
    "n_repeats": 3,      # 重复次数（论文建议30次）
    "rounds": 20,        # 每次对局轮数
    "provider": "deepseek",  # 默认LLM
    "verbose": True,
}


# ============================================================
# 结果保存管理
# ============================================================

class ResultManager:
    """
    实验结果管理器

    目录结构:
    results/
    └── 20250121_143052/           # 时间戳
        ├── experiment_config.json  # 实验配置
        ├── details/                # 每次实验详细数据
        │   └── {实验名}_{模型名}_{次数}_{轮数}.json
        ├── summary/                # 汇总报告 (CSV 格式)
        │   └── {实验名}.csv
        ├── prisoners_dilemma/      # 博弈类型
        │   ├── pure_vs_hybrid.json
        │   └── pure_vs_hybrid.png
        ├── snowdrift/
        └── stag_hunt/
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.root_dir, exist_ok=True)

        # 创建 details 和 summary 目录
        self.details_dir = os.path.join(self.root_dir, "details")
        self.summary_dir = os.path.join(self.root_dir, "summary")
        os.makedirs(self.details_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        # 为每个博弈创建子目录
        for game_name in GAME_REGISTRY.keys():
            game_dir = os.path.join(self.root_dir, game_name)
            os.makedirs(game_dir, exist_ok=True)

        print(f"Results dir: {self.root_dir}")

    def get_game_dir(self, game_name: str) -> str:
        """获取博弈类型目录"""
        return os.path.join(self.root_dir, game_name)

    def save_json(self, game_name: str, experiment_name: str, data: Dict) -> str:
        """保存 JSON 数据"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Saved: {filepath}")
        return filepath

    def save_figure(self, game_name: str, experiment_name: str, fig: plt.Figure) -> str:
        """保存图表"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}.png")

        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {filepath}")
        return filepath

    def save_config(self, config: Dict):
        """保存实验配置"""
        filepath = os.path.join(self.root_dir, "experiment_config.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Config saved: {filepath}")

    def save_summary(self, all_results: Dict):
        """保存汇总报告到根目录"""
        filepath = os.path.join(self.root_dir, "summary.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Summary saved: {filepath}")

    def save_transcript(self, game_name: str, experiment_name: str, content: str) -> str:
        """保存易读的 transcript 文本文件"""
        game_dir = self.get_game_dir(game_name)
        filepath = os.path.join(game_dir, f"{experiment_name}_transcript.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  Saved: {filepath}")
        return filepath

    def save_detail(self, experiment_name: str, provider: str, trial: int, rounds: int, data: Dict) -> str:
        """保存单次实验详细数据到 details 目录"""
        filename = f"{experiment_name}_{provider}_{trial}_{rounds}.json"
        filepath = os.path.join(self.details_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def save_round_records(self, experiment_name: str, game_name: str, provider: str, records: List[Dict]) -> str:
        """
        保存每轮记录到单个文件

        Args:
            experiment_name: 实验名称
            game_name: 博弈类型
            provider: LLM提供商
            records: 每轮记录列表
        """
        filename = f"{experiment_name}_{game_name}_{provider}_rounds.json"
        filepath = os.path.join(self.details_dir, filename)

        data = {
            "experiment": experiment_name,
            "game": game_name,
            "provider": provider,
            "total_records": len(records),
            "records": records,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Rounds: {filepath}")
        return filepath

    def save_experiment_summary(self, experiment_name: str, data: Dict) -> str:
        """保存实验汇总到 summary 目录 (CSV 格式)"""
        filepath = os.path.join(self.summary_dir, f"{experiment_name}.csv")

        rows = self._flatten_summary_to_rows(experiment_name, data)
        if rows:
            fieldnames = ["experiment", "game", "condition", "payoff", "coop_rate"]
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"  Summary: {filepath}")
        return filepath

    def _flatten_summary_to_rows(self, experiment_name: str, data: Dict) -> List[Dict]:
        """将嵌套的实验数据展平为 CSV 行"""
        rows = []

        for game_name, game_data in data.items():
            if isinstance(game_data, dict):
                for key, stats in game_data.items():
                    if isinstance(stats, dict) and "payoff" in stats:
                        row = self._make_summary_row(experiment_name, game_name, key, stats)
                        rows.append(row)
                    elif isinstance(stats, dict) and "payoffs" in stats:
                        # group_dynamics 结构: payoffs/coop_rates/rankings
                        rows.extend(self._make_group_summary_rows(experiment_name, game_name, key, stats))
                    elif isinstance(stats, dict):
                        for sub_key, sub_stats in stats.items():
                            if isinstance(sub_stats, dict) and "payoff" in sub_stats:
                                row = self._make_summary_row(experiment_name, game_name, f"{key}_{sub_key}", sub_stats)
                                rows.append(row)

        return rows

    def _make_group_summary_rows(self, experiment: str, game: str, network: str, stats: Dict) -> List[Dict]:
        """生成 group_dynamics 的 summary 行"""
        rows = []
        payoffs = stats.get("payoffs", {})
        coop_rates = stats.get("coop_rates", {})

        for agent_id, payoff in payoffs.items():
            coop = coop_rates.get(agent_id, 0)
            rows.append({
                "experiment": experiment,
                "game": game,
                "condition": f"{network}_{agent_id}",
                "payoff": f"{payoff:.1f}",
                "coop_rate": f"{coop * 100:.1f}%",
            })

        return rows

    def _make_summary_row(self, experiment: str, game: str, condition: str, stats: Dict) -> Dict:
        """生成单行 summary 数据"""
        payoff = stats.get("payoff", {})
        coop = stats.get("coop_rate", {})

        payoff_str = f"{payoff.get('mean', 0):.1f} ± {payoff.get('std', 0):.1f}"
        coop_str = f"{coop.get('mean', 0) * 100:.1f}%"

        return {
            "experiment": experiment,
            "game": game,
            "condition": condition,
            "payoff": payoff_str,
            "coop_rate": coop_str,
        }


# ============================================================
# 统计工具
# ============================================================

def compute_statistics(values: List[float]) -> Dict:
    """计算统计量 + 95% 置信区间"""
    if not values:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0, "n": 0}

    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0
    n = len(arr)

    if n > 1:
        se = std / np.sqrt(n)
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
    else:
        ci_low = ci_high = mean

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "n": n
    }


def compute_cooperation_rate(history: List[Action]) -> float:
    """计算合作率"""
    if not history:
        return 0.0
    cooperations = sum(1 for a in history if a == Action.COOPERATE)
    return cooperations / len(history)


def make_history_tuples(my_history: List[Action], opp_history: List[Action]) -> List[Tuple[Action, Action]]:
    """
    将两个独立的历史列表转换为元组列表
    用于兼容传统策略的接口

    Args:
        my_history: 我的动作历史
        opp_history: 对手动作历史

    Returns:
        [(我的动作, 对手动作), ...]
    """
    return list(zip(my_history, opp_history))


def print_separator(title: str = "", char: str = "=", width: int = 60):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_game_header(game_name: str):
    """打印博弈类型标题"""
    cn_name = GAME_NAMES_CN.get(game_name, game_name)
    print(f"\n{'─' * 50}")
    print(f"  Game: {cn_name}")
    print(f"{'─' * 50}")


# ============================================================
# 可视化工具
# ============================================================

def plot_comparison_bar(
    data: Dict[str, Dict],
    title: str,
    ylabel: str = "得分",
    game_name: str = "",
) -> plt.Figure:
    """绘制对比柱状图"""

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(data.keys())
    means = [d["payoff"]["mean"] for d in data.values()]
    stds = [d["payoff"]["std"] for d in data.values()]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {GAME_NAMES_CN.get(game_name, game_name)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_cooperation_comparison(
    data: Dict[str, Dict],
    title: str,
    game_name: str = "",
) -> plt.Figure:
    """绘制得分和合作率对比图"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(data.keys())

    # 得分图
    means = [d["payoff"]["mean"] for d in data.values()]
    stds = [d["payoff"]["std"] for d in data.values()]
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax1.set_ylabel("得分")
    ax1.set_title("得分对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    for bar, mean in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # 合作率图
    coop_means = [d["coop_rate"]["mean"] * 100 for d in data.values()]
    coop_stds = [d["coop_rate"]["std"] * 100 for d in data.values()]
    bars2 = ax2.bar(x, coop_means, yerr=coop_stds, capsize=5, color='forestgreen', alpha=0.8)
    ax2.set_ylabel("合作率 (%)")
    ax2.set_title("合作率对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim(0, 105)

    for bar, mean in zip(bars2, coop_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f"{title} - {GAME_NAMES_CN.get(game_name, game_name)}", fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================
# 实验基类
# ============================================================

class BaseExperiment:
    """实验基类"""
    name = "base"
    description = "Base Experiment"

    def __init__(self, result_manager: ResultManager, **kwargs):
        self.result_manager = result_manager
        self.provider = kwargs.get("provider", DEFAULT_CONFIG["provider"])
        self.n_repeats = kwargs.get("n_repeats", DEFAULT_CONFIG["n_repeats"])
        self.rounds = kwargs.get("rounds", DEFAULT_CONFIG["rounds"])
        self.games = kwargs.get("games") or list(GAME_REGISTRY.keys())

    def run(self) -> Dict:
        raise NotImplementedError

    def _print_summary(self, results: Dict):
        raise NotImplementedError


# ============================================================
# 实验1: Pure vs Hybrid
# ============================================================

class Exp1_PureVsHybrid(BaseExperiment):
    """实验1: Pure vs Hybrid LLM"""
    name = "exp1"
    description = "Pure vs Hybrid LLM"

    def run(self) -> Dict:
        print_separator(f"实验1: {self.description}")
        print("Pure:   LLM 自己从历史分析对手")
        print("Hybrid: 代码分析好告诉 LLM")
        print(f"Provider: {self.provider} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            results = {"pure": [], "hybrid": []}
            coop_rates = {"pure": [], "hybrid": []}
            all_round_records = []

            for mode in ["pure", "hybrid"]:
                print(f"\n  Mode: {mode.upper()}")

                for trial in range(self.n_repeats):
                    print(f"    Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                    try:
                        llm_strategy = LLMStrategy(
                            provider=self.provider,
                            mode=mode,
                            game_config=game_config,
                        )

                        opponent = TitForTat()

                        llm_payoff = 0
                        llm_history = []
                        opp_history = []

                        for r in range(self.rounds):
                            llm_action = llm_strategy.choose_action(llm_history, opp_history)
                            opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                            payoff, _ = get_payoff(game_config, llm_action, opp_action)
                            llm_payoff += payoff

                            llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                            all_round_records.append({
                                "mode": mode,
                                "trial": trial + 1,
                                "round": r + 1,
                                "llm_response": llm_response,
                                "llm_action": llm_action.name,
                                "opp_action": opp_action.name,
                                "payoff": payoff,
                                "cumulative_payoff": llm_payoff,
                            })

                            llm_history.append(llm_action)
                            opp_history.append(opp_action)

                        coop_rate = compute_cooperation_rate(llm_history)
                        results[mode].append(llm_payoff)
                        coop_rates[mode].append(coop_rate)

                        if hasattr(llm_strategy, 'get_parse_quality'):
                            parse_quality = llm_strategy.get_parse_quality()
                            success_rate = parse_quality.get('success_rate', 0)
                        elif hasattr(llm_strategy, 'parser'):
                            parse_quality = llm_strategy.parser.get_stats()
                            success_rate = parse_quality.get('success_rate', 0)
                        else:
                            success_rate = 0

                        detail_data = {
                            "experiment": "exp1_pure_vs_hybrid",
                            "game": game_name,
                            "mode": mode,
                            "trial": trial + 1,
                            "rounds": self.rounds,
                            "payoff": llm_payoff,
                            "coop_rate": coop_rate,
                            "parse_success_rate": success_rate,
                            "llm_history": [a.name for a in llm_history],
                            "opp_history": [a.name for a in opp_history],
                            "llm_responses": llm_strategy.raw_responses.copy(),
                        }
                        self.result_manager.save_detail(f"exp1_{game_name}_{mode}", self.provider, trial + 1, self.rounds, detail_data)

                        print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}, Parse: {success_rate:.0%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            game_results = {
                "pure": {
                    "payoff": compute_statistics(results["pure"]),
                    "coop_rate": compute_statistics(coop_rates["pure"]),
                },
                "hybrid": {
                    "payoff": compute_statistics(results["hybrid"]),
                    "coop_rate": compute_statistics(coop_rates["hybrid"]),
                },
            }

            all_results[game_name] = game_results

            self.result_manager.save_json(game_name, "exp1_pure_vs_hybrid", game_results)
            self.result_manager.save_round_records("exp1", game_name, self.provider, all_round_records)

            fig = plot_cooperation_comparison(game_results, "Pure vs Hybrid", game_name)
            self.result_manager.save_figure(game_name, "exp1_pure_vs_hybrid", fig)

        self._print_summary(all_results)
        self.result_manager.save_experiment_summary("exp1_pure_vs_hybrid", all_results)

        return all_results

    def _print_summary(self, results: Dict):
        print_separator("汇总: Pure vs Hybrid")
        print(f"{'Game':<12} {'Pure Payoff':<18} {'Hybrid Payoff':<18} {'Pure Coop':<14} {'Hybrid Coop':<14}")
        print("-" * 76)

        for game_name, stats in results.items():
            cn_name = GAME_NAMES_CN.get(game_name, game_name)

            pure_pay = stats["pure"]["payoff"]
            hybrid_pay = stats["hybrid"]["payoff"]
            pure_coop = stats["pure"]["coop_rate"]
            hybrid_coop = stats["hybrid"]["coop_rate"]

            pure_str = f"{pure_pay['mean']:.1f} ± {pure_pay['std']:.1f}"
            hybrid_str = f"{hybrid_pay['mean']:.1f} ± {hybrid_pay['std']:.1f}"
            pure_coop_str = f"{pure_coop['mean']:.1%}"
            hybrid_coop_str = f"{hybrid_coop['mean']:.1%}"

            print(f"{cn_name:<12} {pure_str:<18} {hybrid_str:<18} {pure_coop_str:<14} {hybrid_coop_str:<14}")


# ============================================================
# 实验2: 记忆视窗对比
# ============================================================

class Exp2_MemoryWindow(BaseExperiment):
    """实验2: 记忆视窗对比"""
    name = "exp2"
    description = "记忆视窗对比"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.windows = kwargs.get("windows", [5, 10, 20, None])
        if self.rounds < 30:
            self.rounds = 30  # 记忆视窗实验至少30轮

    def run(self) -> Dict:
        print_separator(f"实验2: {self.description}")
        print(f"测试不同历史记忆长度: {self.windows}")
        print(f"Provider: {self.provider} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            window_results = {}
            all_round_records = []

            for window in self.windows:
                window_label = str(window) if window else "全部"
                print(f"\n  Window: {window_label}")

                payoffs = []
                coop_rates = []

                for trial in range(self.n_repeats):
                    print(f"    Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                    try:
                        llm_strategy = LLMStrategy(
                            provider=self.provider,
                            mode="pure",
                            game_config=game_config,
                            history_window=window,
                        )

                        opponent = GrimTrigger()

                        llm_payoff = 0
                        llm_history = []
                        opp_history = []

                        for r in range(self.rounds):
                            llm_action = llm_strategy.choose_action(llm_history, opp_history)
                            opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                            payoff, _ = get_payoff(game_config, llm_action, opp_action)
                            llm_payoff += payoff

                            llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                            all_round_records.append({
                                "window": window_label,
                                "trial": trial + 1,
                                "round": r + 1,
                                "llm_response": llm_response,
                                "llm_action": llm_action.name,
                                "opp_action": opp_action.name,
                                "payoff": payoff,
                                "cumulative_payoff": llm_payoff,
                            })

                            llm_history.append(llm_action)
                            opp_history.append(opp_action)

                        coop_rate = compute_cooperation_rate(llm_history)
                        payoffs.append(llm_payoff)
                        coop_rates.append(coop_rate)

                        print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                window_results[window_label] = {
                    "payoff": compute_statistics(payoffs),
                    "coop_rate": compute_statistics(coop_rates),
                }

            all_results[game_name] = window_results

            self.result_manager.save_json(game_name, "exp2_memory_window", window_results)
            self.result_manager.save_round_records("exp2", game_name, self.provider, all_round_records)

            fig = plot_cooperation_comparison(window_results, "记忆视窗对比", game_name)
            self.result_manager.save_figure(game_name, "exp2_memory_window", fig)

        self._print_summary(all_results)
        self.result_manager.save_experiment_summary("exp2_memory_window", all_results)

        return all_results

    def _print_summary(self, results: Dict):
        print_separator("汇总: 记忆视窗对比")

        for game_name, window_stats in results.items():
            cn_name = GAME_NAMES_CN.get(game_name, game_name)
            print(f"\n{cn_name}:")
            print(f"  {'Window':<8} {'Payoff':<18} {'Coop Rate':<12}")
            print(f"  {'-' * 38}")

            for window, stats in window_stats.items():
                pay = stats["payoff"]
                coop = stats["coop_rate"]
                pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
                coop_str = f"{coop['mean']:.1%}"
                print(f"  {window:<8} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 实验3: 多 LLM 对比
# ============================================================

class Exp3_MultiLLM(BaseExperiment):
    """实验3: 多 LLM 对比"""
    name = "exp3"
    description = "多 LLM 对比"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.providers = kwargs.get("providers", ["deepseek", "openai", "gemini"])

    def run(self) -> Dict:
        print_separator(f"实验3: {self.description}")
        print(f"对比 LLM: {self.providers}")
        print(f"Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            provider_results = {}
            all_round_records = []

            for provider in self.providers:
                print(f"\n  Provider: {provider.upper()}")

                payoffs = []
                coop_rates = []

                for trial in range(self.n_repeats):
                    print(f"    Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                    try:
                        llm_strategy = LLMStrategy(
                            provider=provider,
                            mode="hybrid",
                            game_config=game_config,
                        )

                        opponent = TitForTat()

                        llm_payoff = 0
                        llm_history = []
                        opp_history = []

                        for r in range(self.rounds):
                            llm_action = llm_strategy.choose_action(llm_history, opp_history)
                            opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                            payoff, _ = get_payoff(game_config, llm_action, opp_action)
                            llm_payoff += payoff

                            llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                            all_round_records.append({
                                "provider": provider,
                                "trial": trial + 1,
                                "round": r + 1,
                                "llm_response": llm_response,
                                "llm_action": llm_action.name,
                                "opp_action": opp_action.name,
                                "payoff": payoff,
                                "cumulative_payoff": llm_payoff,
                            })

                            llm_history.append(llm_action)
                            opp_history.append(opp_action)

                        coop_rate = compute_cooperation_rate(llm_history)
                        payoffs.append(llm_payoff)
                        coop_rates.append(coop_rate)

                        print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                provider_results[provider] = {
                    "payoff": compute_statistics(payoffs),
                    "coop_rate": compute_statistics(coop_rates),
                }

            all_results[game_name] = provider_results

            self.result_manager.save_json(game_name, "exp3_multi_llm", provider_results)
            self.result_manager.save_round_records("exp3", game_name, "all", all_round_records)

            fig = plot_cooperation_comparison(provider_results, "多 LLM 对比", game_name)
            self.result_manager.save_figure(game_name, "exp3_multi_llm", fig)

        self._print_summary(all_results)
        self.result_manager.save_experiment_summary("exp3_multi_llm", all_results)

        return all_results

    def _print_summary(self, results: Dict):
        print_separator("汇总: 多 LLM 对比")

        for game_name, provider_stats in results.items():
            cn_name = GAME_NAMES_CN.get(game_name, game_name)
            print(f"\n{cn_name}:")
            print(f"  {'LLM':<12} {'Payoff':<18} {'Coop Rate':<12}")
            print(f"  {'-' * 42}")

            sorted_providers = sorted(
                provider_stats.items(),
                key=lambda x: x[1]["payoff"]["mean"],
                reverse=True
            )

            for provider, stats in sorted_providers:
                pay = stats["payoff"]
                coop = stats["coop_rate"]
                pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
                coop_str = f"{coop['mean']:.1%}"
                print(f"  {provider:<12} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 实验4: Cheap Talk 三方对战 (3 LLM Round-Robin)
# ============================================================

class Exp4_CheapTalk3LLM(BaseExperiment):
    """实验4: Cheap Talk 三方对战"""
    name = "exp4"
    description = "Cheap Talk 三方对战 (3 LLM)"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.providers = kwargs.get("providers", ["deepseek", "openai", "gemini"])

    def run(self) -> Dict:
        print_separator(f"实验4: {self.description}")
        print("对比: 无交流 vs 有语言交流 (Round-Robin)")
        print(f"Providers: {self.providers} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            pairs = []
            for i in range(len(self.providers)):
                for j in range(i + 1, len(self.providers)):
                    pairs.append((self.providers[i], self.providers[j]))

            results = {"no_talk": {}, "cheap_talk": {}}
            coop_rates = {"no_talk": {}, "cheap_talk": {}}
            promise_kept = {}
            all_round_records = []

            for provider in self.providers:
                results["no_talk"][provider] = []
                results["cheap_talk"][provider] = []
                coop_rates["no_talk"][provider] = []
                coop_rates["cheap_talk"][provider] = []
                promise_kept[provider] = []

            detailed_trials = {"no_talk": [], "cheap_talk": []}

            for mode in ["no_talk", "cheap_talk"]:
                print(f"\n  Mode: {mode}")

                for trial in range(self.n_repeats):
                    print(f"    Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                    try:
                        use_cheap_talk = (mode == "cheap_talk")

                        llms = {}
                        for provider in self.providers:
                            llms[provider] = LLMStrategy(
                                provider=provider,
                                mode="hybrid",
                                game_config=game_config,
                                enable_cheap_talk=use_cheap_talk,
                                agent_name=f"Player({provider})",
                            )

                        total_payoffs = {p: 0 for p in self.providers}
                        histories = {p: [] for p in self.providers}
                        messages = {p: [] for p in self.providers}

                        round_details = []

                        for r in range(self.rounds):
                            round_data = {"round": r + 1, "matches": []}

                            for p1, p2 in pairs:
                                llm1 = llms[p1]
                                llm2 = llms[p2]

                                msg1 = ""
                                msg2 = ""
                                if use_cheap_talk:
                                    if hasattr(llm1, 'generate_message'):
                                        msg1 = llm1.generate_message(histories[p1], histories[p2], p2)
                                    if hasattr(llm2, 'generate_message'):
                                        msg2 = llm2.generate_message(histories[p2], histories[p1], p1)

                                action1 = llm1.choose_action(histories[p1], histories[p2], p2, opponent_message=msg2)
                                action2 = llm2.choose_action(histories[p2], histories[p1], p1, opponent_message=msg1)

                                payoff1, payoff2 = get_payoff(game_config, action1, action2)
                                total_payoffs[p1] += payoff1
                                total_payoffs[p2] += payoff2

                                match_data = {
                                    "pair": f"{p1}_vs_{p2}",
                                    "p1": p1,
                                    "p2": p2,
                                    "p1_message": msg1,
                                    "p2_message": msg2,
                                    "p1_action": action1.name,
                                    "p2_action": action2.name,
                                    "p1_payoff": payoff1,
                                    "p2_payoff": payoff2,
                                }
                                round_data["matches"].append(match_data)

                                all_round_records.append({
                                    "mode": mode,
                                    "trial": trial + 1,
                                    "round": r + 1,
                                    "pair": f"{p1}_vs_{p2}",
                                    "p1": p1,
                                    "p2": p2,
                                    "p1_message": msg1,
                                    "p2_message": msg2,
                                    "p1_action": action1.name,
                                    "p2_action": action2.name,
                                    "p1_payoff": payoff1,
                                    "p2_payoff": payoff2,
                                })

                            for p1, p2 in pairs:
                                for match in round_data["matches"]:
                                    if match["p1"] == p1 and match["p2"] == p2:
                                        histories[p1].append(Action[match["p1_action"]])
                                        histories[p2].append(Action[match["p2_action"]])
                                        if use_cheap_talk:
                                            if match["p1_message"]:
                                                messages[p1].append(match["p1_message"])
                                            if match["p2_message"]:
                                                messages[p2].append(match["p2_message"])
                                        break

                            round_details.append(round_data)

                        coop_rate_dict = {}
                        for provider in self.providers:
                            coop_rate = compute_cooperation_rate(histories[provider])
                            results[mode][provider].append(total_payoffs[provider])
                            coop_rates[mode][provider].append(coop_rate)
                            coop_rate_dict[provider] = coop_rate

                            if use_cheap_talk and messages[provider]:
                                kept = _analyze_promise_keeping(messages[provider], histories[provider])
                                promise_kept[provider].append(kept)

                        trial_record = {
                            "trial": trial + 1,
                            "payoffs": total_payoffs.copy(),
                            "coop_rates": coop_rate_dict,
                            "total_social_payoff": sum(total_payoffs.values()),
                            "rounds": round_details,
                        }
                        detailed_trials[mode].append(trial_record)

                        total_social = sum(total_payoffs.values())
                        avg_coop = sum(coop_rate_dict.values()) / len(coop_rate_dict)
                        print(f"Social: {total_social:.1f}, Avg coop: {avg_coop:.1%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            game_results = {}
            for mode in ["no_talk", "cheap_talk"]:
                mode_results = {"providers": {}}
                for provider in self.providers:
                    mode_results["providers"][provider] = {
                        "payoff": compute_statistics(results[mode][provider]),
                        "coop_rate": compute_statistics(coop_rates[mode][provider]),
                    }

                social_payoffs = [sum(t["payoffs"].values()) for t in detailed_trials[mode]]
                mode_results["social_payoff"] = compute_statistics(social_payoffs)

                if mode == "cheap_talk":
                    for provider in self.providers:
                        if promise_kept[provider]:
                            mode_results["providers"][provider]["promise_kept"] = compute_statistics(promise_kept[provider])

                game_results[mode] = mode_results

            all_results[game_name] = game_results

            self.result_manager.save_json(game_name, "exp4_cheap_talk_3llm", game_results)
            self.result_manager.save_round_records("exp4", game_name, "3llm", all_round_records)

            transcript = self._generate_transcript(game_name, detailed_trials)
            self.result_manager.save_transcript(game_name, "exp4_cheap_talk_3llm", transcript)

            fig = self._plot_results(game_results, game_name)
            if fig:
                self.result_manager.save_figure(game_name, "exp4_cheap_talk_3llm", fig)

        self._print_summary(all_results)
        self.result_manager.save_experiment_summary("exp4_cheap_talk_3llm", all_results)

        return all_results

    def _generate_transcript(self, game_name: str, detailed_trials: Dict) -> str:
        """生成3 LLM Cheap Talk 交互记录"""
        cn_name = GAME_NAMES_CN.get(game_name, game_name)

        lines = []
        lines.append("=" * 70)
        lines.append(f"CHEAP TALK 三方对战实验记录 - {cn_name}")
        lines.append(f"Providers: {', '.join(self.providers)}")
        lines.append(f"对战模式: 3 LLM Round-Robin")
        lines.append("=" * 70)
        lines.append("")

        for mode in ["no_talk", "cheap_talk"]:
            mode_name = "无交流模式 (No Talk)" if mode == "no_talk" else "有交流模式 (Cheap Talk)"
            lines.append("-" * 70)
            lines.append(f"【{mode_name}】")
            lines.append("-" * 70)

            for trial_data in detailed_trials[mode]:
                trial_num = trial_data["trial"]
                payoffs = trial_data.get("payoffs", {})
                coop_rates = trial_data.get("coop_rates", {})
                total_social = trial_data.get("total_social_payoff", 0)

                lines.append("")
                lines.append(f">>> Trial {trial_num}")
                lines.append(f"    社会总收益: {total_social:.1f}")
                for p in self.providers:
                    lines.append(f"    {p}: 得分={payoffs.get(p, 0):.1f}, 合作率={coop_rates.get(p, 0):.1%}")
                lines.append("")

                for rd in trial_data.get("rounds", []):
                    round_num = rd["round"]
                    lines.append(f"  Round {round_num:2d}:")
                    for match in rd.get("matches", []):
                        p1, p2 = match["p1"], match["p2"]
                        p1_msg = match.get("p1_message", "")
                        p2_msg = match.get("p2_message", "")
                        p1_action = match.get("p1_action", "")
                        p2_action = match.get("p2_action", "")
                        p1_payoff = match.get("p1_payoff", 0)
                        p2_payoff = match.get("p2_payoff", 0)

                        lines.append(f"    [{p1} vs {p2}]")
                        if p1_msg:
                            lines.append(f"      {p1} says: \"{p1_msg}\"")
                        if p2_msg:
                            lines.append(f"      {p2} says: \"{p2_msg}\"")
                        p1_symbol = "合作" if p1_action == "COOPERATE" else "背叛"
                        p2_symbol = "合作" if p2_action == "COOPERATE" else "背叛"
                        lines.append(f"      {p1}: {p1_symbol} | {p2}: {p2_symbol} | 得分: {p1_payoff}/{p2_payoff}")
                    lines.append("")

            lines.append("")

        lines.append("=" * 70)
        lines.append("记录结束")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _plot_results(self, game_results: Dict, game_name: str) -> Optional[plt.Figure]:
        """绘制3 LLM Cheap Talk 对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        provider_colors = {
            "deepseek": "#4CAF50",
            "openai": "#2196F3",
            "gemini": "#FF9800",
        }

        ax1 = axes[0]
        x = np.arange(len(self.providers))
        width = 0.35

        no_talk_means = [game_results["no_talk"]["providers"][p]["payoff"]["mean"] for p in self.providers]
        no_talk_stds = [game_results["no_talk"]["providers"][p]["payoff"]["std"] for p in self.providers]
        cheap_talk_means = [game_results["cheap_talk"]["providers"][p]["payoff"]["mean"] for p in self.providers]
        cheap_talk_stds = [game_results["cheap_talk"]["providers"][p]["payoff"]["std"] for p in self.providers]

        ax1.bar(x - width/2, no_talk_means, width, yerr=no_talk_stds, label='No Talk', color='gray', alpha=0.7, capsize=5)
        ax1.bar(x + width/2, cheap_talk_means, width, yerr=cheap_talk_stds, label='Cheap Talk',
                color=[provider_colors.get(p, '#9C27B0') for p in self.providers], alpha=0.8, capsize=5)

        ax1.set_ylabel('得分')
        ax1.set_title('各 LLM 得分对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.providers)
        ax1.legend()

        ax2 = axes[1]
        no_talk_coop = [game_results["no_talk"]["providers"][p]["coop_rate"]["mean"] * 100 for p in self.providers]
        cheap_talk_coop = [game_results["cheap_talk"]["providers"][p]["coop_rate"]["mean"] * 100 for p in self.providers]

        ax2.bar(x - width/2, no_talk_coop, width, label='No Talk', color='gray', alpha=0.7)
        ax2.bar(x + width/2, cheap_talk_coop, width, label='Cheap Talk',
                color=[provider_colors.get(p, '#9C27B0') for p in self.providers], alpha=0.8)

        ax2.set_ylabel('合作率 (%)')
        ax2.set_title('各 LLM 合作率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.providers)
        ax2.set_ylim(0, 105)
        ax2.legend()

        game_cn = GAME_NAMES_CN.get(game_name, game_name)
        fig.suptitle(f"Cheap Talk 三方对战 - {game_cn}", fontsize=14)
        plt.tight_layout()
        return fig

    def _print_summary(self, results: Dict):
        """打印3 LLM Cheap Talk 汇总"""
        print_separator("汇总: Cheap Talk 三方对战 (3 LLM)")

        for game_name, stats in results.items():
            cn_name = GAME_NAMES_CN.get(game_name, game_name)
            print(f"\n{cn_name}:")

            for mode in ["no_talk", "cheap_talk"]:
                mode_name = "No Talk" if mode == "no_talk" else "Cheap Talk"
                social = stats[mode]["social_payoff"]
                print(f"\n  [{mode_name}] 社会总收益: {social['mean']:.1f} ± {social['std']:.1f}")

                for provider in self.providers:
                    p_stats = stats[mode]["providers"][provider]
                    payoff = p_stats["payoff"]
                    coop = p_stats["coop_rate"]
                    print(f"    {provider}: {payoff['mean']:.1f} ± {payoff['std']:.1f}, Coop: {coop['mean']:.1%}")

                    if mode == "cheap_talk" and "promise_kept" in p_stats:
                        pk = p_stats["promise_kept"]
                        print(f"      承诺遵守: {pk['mean']:.1%}")


# ============================================================
# 实验4b: Cheap Talk 一对一 (1v1)
# ============================================================

class Exp4b_CheapTalk1v1(BaseExperiment):
    """实验4b: Cheap Talk 一对一"""
    name = "exp4b"
    description = "Cheap Talk 一对一 (1v1)"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.provider1 = kwargs.get("provider1", self.provider)
        self.provider2 = kwargs.get("provider2", self.provider)

    def run(self) -> Dict:
        print_separator(f"实验4b: {self.description}")
        print("对比: 无交流 vs 有语言交流 (LLM vs LLM)")
        if self.provider1 == self.provider2:
            print(f"Provider: {self.provider1} vs {self.provider2} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")
        else:
            print(f"Provider: {self.provider1} vs {self.provider2} (跨模型对战) | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            results = {"no_talk": [], "cheap_talk": []}
            coop_rates = {"no_talk": {"player1": [], "player2": []}, "cheap_talk": {"player1": [], "player2": []}}
            promise_kept = {"player1": [], "player2": []}
            all_round_records = []

            detailed_trials = {"no_talk": [], "cheap_talk": []}

            for mode in ["no_talk", "cheap_talk"]:
                print(f"\n  Mode: {mode}")

                for trial in range(self.n_repeats):
                    print(f"    Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                    try:
                        use_cheap_talk = (mode == "cheap_talk")

                        llm1 = LLMStrategy(
                            provider=self.provider1,
                            mode="hybrid",
                            game_config=game_config,
                            enable_cheap_talk=use_cheap_talk,
                            agent_name=f"Player1({self.provider1})",
                        )

                        llm2 = LLMStrategy(
                            provider=self.provider2,
                            mode="hybrid",
                            game_config=game_config,
                            enable_cheap_talk=use_cheap_talk,
                            agent_name=f"Player2({self.provider2})",
                        )

                        total_payoff_1 = 0
                        total_payoff_2 = 0
                        history_1 = []
                        history_2 = []
                        messages_1 = []
                        messages_2 = []

                        round_details = []

                        for r in range(self.rounds):
                            msg1 = ""
                            msg2 = ""
                            if use_cheap_talk:
                                if hasattr(llm1, 'generate_message'):
                                    msg1 = llm1.generate_message(history_1, history_2, "Player2")
                                    messages_1.append(msg1)
                                if hasattr(llm2, 'generate_message'):
                                    msg2 = llm2.generate_message(history_2, history_1, "Player1")
                                    messages_2.append(msg2)

                            action1 = llm1.choose_action(history_1, history_2, "Player2", opponent_message=msg2)
                            action2 = llm2.choose_action(history_2, history_1, "Player1", opponent_message=msg1)

                            payoff1, payoff2 = get_payoff(game_config, action1, action2)
                            total_payoff_1 += payoff1
                            total_payoff_2 += payoff2

                            history_1.append(action1)
                            history_2.append(action2)

                            llm1_response = llm1.raw_responses[-1] if llm1.raw_responses else ""
                            llm2_response = llm2.raw_responses[-1] if llm2.raw_responses else ""
                            round_details.append({
                                "round": r + 1,
                                "player1_message": msg1,
                                "player2_message": msg2,
                                "player1_action": action1.name,
                                "player2_action": action2.name,
                                "player1_payoff": payoff1,
                                "player2_payoff": payoff2,
                                "player1_cumulative": total_payoff_1,
                                "player2_cumulative": total_payoff_2,
                            })

                            all_round_records.append({
                                "mode": mode,
                                "trial": trial + 1,
                                "round": r + 1,
                                "player1_message": msg1,
                                "player2_message": msg2,
                                "player1_response": llm1_response,
                                "player2_response": llm2_response,
                                "player1_action": action1.name,
                                "player2_action": action2.name,
                                "player1_payoff": payoff1,
                                "player2_payoff": payoff2,
                            })

                        coop_rate_1 = compute_cooperation_rate(history_1)
                        coop_rate_2 = compute_cooperation_rate(history_2)
                        total_payoff = total_payoff_1 + total_payoff_2

                        results[mode].append(total_payoff)
                        coop_rates[mode]["player1"].append(coop_rate_1)
                        coop_rates[mode]["player2"].append(coop_rate_2)

                        trial_record = {
                            "trial": trial + 1,
                            "player1_payoff": total_payoff_1,
                            "player2_payoff": total_payoff_2,
                            "total_payoff": total_payoff,
                            "player1_coop_rate": coop_rate_1,
                            "player2_coop_rate": coop_rate_2,
                            "rounds": round_details,
                        }

                        if use_cheap_talk:
                            if messages_1:
                                kept1 = _analyze_promise_keeping(messages_1, history_1)
                                promise_kept["player1"].append(kept1)
                                trial_record["player1_promise_keeping"] = kept1
                            if messages_2:
                                kept2 = _analyze_promise_keeping(messages_2, history_2)
                                promise_kept["player2"].append(kept2)
                                trial_record["player2_promise_keeping"] = kept2

                        detailed_trials[mode].append(trial_record)

                        provider_label = f"{self.provider1}_vs_{self.provider2}" if self.provider1 != self.provider2 else self.provider1
                        detail_data = {
                            "experiment": "exp4b_cheap_talk_1v1",
                            "game": game_name,
                            "mode": mode,
                            "trial": trial + 1,
                            "rounds": self.rounds,
                            "player1_payoff": total_payoff_1,
                            "player2_payoff": total_payoff_2,
                            "total_payoff": total_payoff,
                            "player1_coop_rate": coop_rate_1,
                            "player2_coop_rate": coop_rate_2,
                            "player1_messages": messages_1 if use_cheap_talk else [],
                            "player2_messages": messages_2 if use_cheap_talk else [],
                            "player1_history": [a.name for a in history_1],
                            "player2_history": [a.name for a in history_2],
                            "player1_responses": llm1.raw_responses.copy(),
                            "player2_responses": llm2.raw_responses.copy(),
                        }
                        self.result_manager.save_detail(f"exp4b_{game_name}_{mode}", provider_label, trial + 1, self.rounds, detail_data)

                        avg_coop = (coop_rate_1 + coop_rate_2) / 2
                        print(f"Total: {total_payoff:.1f}, Avg coop: {avg_coop:.1%}")

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            game_results = {
                "no_talk": {
                    "total_payoff": compute_statistics(results["no_talk"]),
                    "player1_coop_rate": compute_statistics(coop_rates["no_talk"]["player1"]),
                    "player2_coop_rate": compute_statistics(coop_rates["no_talk"]["player2"]),
                },
                "cheap_talk": {
                    "total_payoff": compute_statistics(results["cheap_talk"]),
                    "player1_coop_rate": compute_statistics(coop_rates["cheap_talk"]["player1"]),
                    "player2_coop_rate": compute_statistics(coop_rates["cheap_talk"]["player2"]),
                    "player1_promise_kept": compute_statistics(promise_kept["player1"]) if promise_kept["player1"] else None,
                    "player2_promise_kept": compute_statistics(promise_kept["player2"]) if promise_kept["player2"] else None,
                },
            }

            all_results[game_name] = game_results

            self.result_manager.save_json(game_name, "exp4b_cheap_talk_1v1", game_results)

            provider_label = f"{self.provider1}_vs_{self.provider2}" if self.provider1 != self.provider2 else self.provider1
            self.result_manager.save_round_records("exp4b", game_name, provider_label, all_round_records)

            transcript = self._generate_transcript(game_name, detailed_trials)
            self.result_manager.save_transcript(game_name, "exp4b_cheap_talk_1v1", transcript)

            plot_data = {}
            for mode_key in ["no_talk", "cheap_talk"]:
                mode_stats = game_results[mode_key]
                avg_coop_mean = (mode_stats["player1_coop_rate"]["mean"] + mode_stats["player2_coop_rate"]["mean"]) / 2
                avg_coop_std = (mode_stats["player1_coop_rate"]["std"] + mode_stats["player2_coop_rate"]["std"]) / 2
                plot_data[mode_key] = {
                    "payoff": mode_stats["total_payoff"],
                    "coop_rate": {"mean": avg_coop_mean, "std": avg_coop_std},
                }
            fig = plot_cooperation_comparison(plot_data, "Cheap Talk 一对一 (1v1)", game_name)
            self.result_manager.save_figure(game_name, "exp4b_cheap_talk_1v1", fig)

        self._print_summary(all_results)
        self.result_manager.save_experiment_summary("exp4b_cheap_talk_1v1", all_results)

        return all_results

    def _generate_transcript(self, game_name: str, detailed_trials: Dict) -> str:
        """生成一对一 Cheap Talk 交互记录"""
        cn_name = GAME_NAMES_CN.get(game_name, game_name)

        lines = []
        lines.append("=" * 70)
        lines.append(f"CHEAP TALK 一对一实验记录 - {cn_name}")
        lines.append(f"Player1: {self.provider1} | Player2: {self.provider2}")
        lines.append(f"对战模式: LLM vs LLM (双向交流)")
        lines.append("=" * 70)
        lines.append("")

        for mode in ["no_talk", "cheap_talk"]:
            mode_name = "无交流模式 (No Talk)" if mode == "no_talk" else "有交流模式 (Cheap Talk)"
            lines.append("-" * 70)
            lines.append(f"【{mode_name}】")
            lines.append("-" * 70)

            for trial_data in detailed_trials[mode]:
                trial_num = trial_data["trial"]
                p1_payoff = trial_data.get("player1_payoff", 0)
                p2_payoff = trial_data.get("player2_payoff", 0)
                total_payoff = trial_data.get("total_payoff", p1_payoff + p2_payoff)
                p1_coop = trial_data.get("player1_coop_rate", 0)
                p2_coop = trial_data.get("player2_coop_rate", 0)

                lines.append("")
                lines.append(f">>> Trial {trial_num}")
                lines.append(f"    社会总收益: {total_payoff:.1f} | P1得分: {p1_payoff:.1f} | P2得分: {p2_payoff:.1f}")
                lines.append(f"    P1合作率: {p1_coop:.1%} | P2合作率: {p2_coop:.1%}")

                if "player1_promise_keeping" in trial_data:
                    lines.append(f"    P1承诺遵守率: {trial_data['player1_promise_keeping']:.1%}")
                if "player2_promise_keeping" in trial_data:
                    lines.append(f"    P2承诺遵守率: {trial_data['player2_promise_keeping']:.1%}")

                lines.append("")

                for rd in trial_data["rounds"]:
                    round_num = rd["round"]
                    p1_msg = rd.get("player1_message", "")
                    p2_msg = rd.get("player2_message", "")
                    p1_action = rd.get("player1_action", "")
                    p2_action = rd.get("player2_action", "")
                    p1_payoff_rd = rd.get("player1_payoff", 0)
                    p2_payoff_rd = rd.get("player2_payoff", 0)

                    lines.append(f"  Round {round_num:2d}:")

                    if p1_msg:
                        lines.append(f"    P1 says: \"{p1_msg}\"")
                    if p2_msg:
                        lines.append(f"    P2 says: \"{p2_msg}\"")

                    p1_symbol = "合作" if p1_action == "COOPERATE" else "背叛"
                    p2_symbol = "合作" if p2_action == "COOPERATE" else "背叛"

                    lines.append(f"    P1: {p1_symbol} | P2: {p2_symbol}")
                    lines.append(f"    得分: P1={p1_payoff_rd}, P2={p2_payoff_rd}")
                    lines.append("")

            lines.append("")

        lines.append("=" * 70)
        lines.append("记录结束")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _print_summary(self, results: Dict):
        """打印一对一 Cheap Talk 汇总"""
        print_separator("汇总: Cheap Talk 一对一 (1v1)")

        for game_name, stats in results.items():
            cn_name = GAME_NAMES_CN.get(game_name, game_name)
            print(f"\n{cn_name}:")

            no_talk = stats["no_talk"]
            cheap_talk = stats["cheap_talk"]

            no_talk_avg_coop = (no_talk['player1_coop_rate']['mean'] + no_talk['player2_coop_rate']['mean']) / 2
            cheap_talk_avg_coop = (cheap_talk['player1_coop_rate']['mean'] + cheap_talk['player2_coop_rate']['mean']) / 2

            print(f"  No talk:    Total Payoff {no_talk['total_payoff']['mean']:.1f} ± {no_talk['total_payoff']['std']:.1f}, "
                  f"Avg Coop {no_talk_avg_coop:.1%}")
            print(f"  Cheap talk: Total Payoff {cheap_talk['total_payoff']['mean']:.1f} ± {cheap_talk['total_payoff']['std']:.1f}, "
                  f"Avg Coop {cheap_talk_avg_coop:.1%}")

            if cheap_talk.get("player1_promise_kept"):
                print(f"  P1 Promise kept: {cheap_talk['player1_promise_kept']['mean']:.1%}")
            if cheap_talk.get("player2_promise_kept"):
                print(f"  P2 Promise kept: {cheap_talk['player2_promise_kept']['mean']:.1%}")


# 辅助函数（保留在class外部供其他地方调用）
def _analyze_promise_keeping(messages: List[str], actions: List[Action]) -> float:
    """分析承诺遵守率"""
    if not messages or not actions:
        return 0.0

    kept_count = 0
    promise_count = 0

    cooperation_keywords = ["合作", "cooperate", "trust", "信任", "一起"]

    for msg, action in zip(messages, actions):
        if msg and any(kw in msg.lower() for kw in cooperation_keywords):
            promise_count += 1
            if action == Action.COOPERATE:
                kept_count += 1

    return kept_count / promise_count if promise_count > 0 else 1.0


# ============================================================
# 实验5: 群体动力学
# ============================================================

class Exp5_GroupDynamics(BaseExperiment):
    """实验5: 群体动力学 (单 Provider)"""
    name = "exp5"
    description = "群体动力学 (单 Provider)"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.n_agents = kwargs.get("n_agents", 10)
        self.networks = kwargs.get("networks", ["fully_connected", "small_world"])

    def run(self) -> Dict:
        print_separator(f"实验5: {self.description}")
        print(f"Agent数量: {self.n_agents} | Provider: {self.provider}")
        print(f"网络: {self.networks} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            network_results = {}
            all_round_records = []

            for network_name in self.networks:
                network_cn = NETWORK_NAMES_CN.get(network_name, network_name)
                print(f"\n  网络: {network_cn}")

                try:
                    all_trials_payoffs = defaultdict(list)
                    all_trials_coop_rates = defaultdict(list)

                    for i in range(self.n_repeats):
                        print(f"    Repeat {i + 1}/{self.n_repeats}...", end=" ", flush=True)

                        strategies = []

                        n_llm = max(2, int(self.n_agents * 0.2))
                        n_classic = self.n_agents - n_llm

                        for k in range(n_llm):
                            strategies.append((
                                f"LLM_{k + 1}",
                                LLMStrategy(provider=self.provider, mode="hybrid", game_config=game_config)
                            ))

                        classic_classes = [
                            TitForTat, AlwaysCooperate, AlwaysDefect,
                            Pavlov, GrimTrigger, RandomStrategy
                        ]
                        for k in range(n_classic):
                            StrategyClass = classic_classes[k % len(classic_classes)]
                            strategies.append((
                                f"{StrategyClass.__name__}_{k + 1}",
                                StrategyClass()
                            ))

                        agent_names = [name for name, _ in strategies]
                        NetworkClass = NETWORK_REGISTRY[network_name]
                        network = NetworkClass(agent_names)

                        agents = {}
                        for name, strategy in strategies:
                            agents[name] = AgentState(name=name, strategy=strategy)

                        sim = GameSimulation(
                            agents=agents,
                            network=network,
                            game_config=game_config,
                            rounds=self.rounds,
                            verbose=False
                        )

                        sim.run()

                        trial_payoffs = {}
                        trial_coop_rates = {}
                        for aid, agent in agents.items():
                            all_trials_payoffs[aid].append(agent.total_payoff)
                            trial_payoffs[aid] = agent.total_payoff

                            history = agent.game_history
                            if history:
                                actions = [Action(h["my_action"]) for h in history]
                                rate = compute_cooperation_rate(actions)
                            else:
                                rate = 0.0
                            all_trials_coop_rates[aid].append(rate)
                            trial_coop_rates[aid] = rate

                        for aid, agent in agents.items():
                            if hasattr(agent.strategy, 'raw_responses') and agent.game_history:
                                responses = agent.strategy.raw_responses
                                for r_idx, hist in enumerate(agent.game_history):
                                    llm_response = responses[r_idx] if r_idx < len(responses) else ""
                                    all_round_records.append({
                                        "network": network_name,
                                        "trial": i + 1,
                                        "round": r_idx + 1,
                                        "agent": aid,
                                        "llm_response": llm_response,
                                        "my_action": hist.get("my_action", ""),
                                        "opponent": hist.get("opponent", ""),
                                        "opp_action": hist.get("opp_action", ""),
                                        "payoff": hist.get("payoff", 0),
                                    })

                        print("Done")

                    final_payoffs = {k: np.mean(v) for k, v in all_trials_payoffs.items()}
                    coop_rates = {k: np.mean(v) for k, v in all_trials_coop_rates.items()}

                    network_results[network_name] = {
                        "payoffs": final_payoffs,
                        "coop_rates": coop_rates,
                        "rankings": sorted(final_payoffs.items(), key=lambda x: x[1], reverse=True),
                    }

                    print(f"    Avg ranking (Top 5):")
                    for rank, (aid, payoff) in enumerate(network_results[network_name]["rankings"][:5], 1):
                        coop = coop_rates.get(aid, 0)
                        marker = "[LLM]" if aid.startswith("LLM") else "[Classic]"
                        print(f"      {marker} {rank}. {aid}: {payoff:.1f} (Coop: {coop:.1%})")

                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()
                    network_results[network_name] = {"error": str(e)}

            all_results[game_name] = network_results
            self.result_manager.save_json(game_name, "exp5_group_dynamics", network_results)
            self.result_manager.save_round_records("exp5", game_name, self.provider, all_round_records)

            fig = _plot_group_rankings(network_results, game_name)
            if fig:
                self.result_manager.save_figure(game_name, "exp5_group_dynamics", fig)

        self.result_manager.save_experiment_summary("exp5_group_dynamics", all_results)

        return all_results


class Exp5b_GroupDynamicsMulti(BaseExperiment):
    """实验5b: 多 Provider 群体动力学"""
    name = "exp5b"
    description = "多 Provider 群体动力学"

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.n_agents = kwargs.get("n_agents", 10)
        self.providers = kwargs.get("providers", ["deepseek", "openai", "gemini"])
        self.networks = kwargs.get("networks", ["fully_connected", "small_world"])

    def run(self) -> Dict:
        print_separator(f"实验5b: {self.description}")
        print(f"Agent数量: {self.n_agents} | Providers: {self.providers}")
        print(f"网络: {self.networks} | Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            network_results = {}
            all_round_records = []

            for network_name in self.networks:
                network_cn = NETWORK_NAMES_CN.get(network_name, network_name)
                print(f"\n  网络: {network_cn}")

                try:
                    all_trials_payoffs = defaultdict(list)
                    all_trials_coop_rates = defaultdict(list)

                    for i in range(self.n_repeats):
                        print(f"    Repeat {i + 1}/{self.n_repeats}...", end=" ", flush=True)

                        strategies = []

                        min_llms = len(self.providers)
                        n_llm_total = max(min_llms, int(self.n_agents * 0.2))
                        n_classic = self.n_agents - n_llm_total

                        base_count = n_llm_total // len(self.providers)
                        remainder = n_llm_total % len(self.providers)
                        llm_counts = [base_count + 1 if k < remainder else base_count for k in range(len(self.providers))]

                        current_llm_idx = 1
                        for provider, count in zip(self.providers, llm_counts):
                            for _ in range(count):
                                strategies.append((
                                    f"LLM_{provider}_{current_llm_idx}",
                                    LLMStrategy(provider=provider, mode="hybrid", game_config=game_config)
                                ))
                                current_llm_idx += 1

                        classic_classes = [
                            TitForTat, AlwaysCooperate, AlwaysDefect,
                            Pavlov, GrimTrigger, RandomStrategy
                        ]
                        for k in range(n_classic):
                            StrategyClass = classic_classes[k % len(classic_classes)]
                            strategies.append((
                                f"{StrategyClass.__name__}_{k + 1}",
                                StrategyClass()
                            ))

                        agent_names = [name for name, _ in strategies]
                        NetworkClass = NETWORK_REGISTRY[network_name]
                        network = NetworkClass(agent_names)

                        agents = {}
                        for name, strategy in strategies:
                            agents[name] = AgentState(name=name, strategy=strategy)

                        sim = GameSimulation(
                            agents=agents,
                            network=network,
                            game_config=game_config,
                            rounds=self.rounds,
                            verbose=False
                        )

                        sim.run()

                        trial_payoffs = {}
                        trial_coop_rates = {}
                        llm_responses = {}
                        for aid, agent in agents.items():
                            all_trials_payoffs[aid].append(agent.total_payoff)
                            trial_payoffs[aid] = agent.total_payoff

                            history = agent.game_history
                            if history:
                                actions = [Action(h["my_action"]) for h in history]
                                rate = compute_cooperation_rate(actions)
                            else:
                                rate = 0.0
                            all_trials_coop_rates[aid].append(rate)
                            trial_coop_rates[aid] = rate

                            if hasattr(agent.strategy, 'raw_responses'):
                                llm_responses[aid] = agent.strategy.raw_responses.copy()

                        for aid, agent in agents.items():
                            if hasattr(agent.strategy, 'raw_responses') and agent.game_history:
                                responses = agent.strategy.raw_responses
                                for r_idx, hist in enumerate(agent.game_history):
                                    llm_response = responses[r_idx] if r_idx < len(responses) else ""
                                    all_round_records.append({
                                        "network": network_name,
                                        "trial": i + 1,
                                        "round": r_idx + 1,
                                        "agent": aid,
                                        "llm_response": llm_response,
                                        "my_action": hist.get("my_action", ""),
                                        "opponent": hist.get("opponent", ""),
                                        "opp_action": hist.get("opp_action", ""),
                                        "payoff": hist.get("payoff", 0),
                                    })

                        detail_data = {
                            "experiment": "exp5b_group_dynamics_multi",
                            "game": game_name,
                            "network": network_name,
                            "providers": self.providers,
                            "trial": i + 1,
                            "rounds": self.rounds,
                            "n_agents": self.n_agents,
                            "payoffs": trial_payoffs,
                            "coop_rates": trial_coop_rates,
                            "llm_responses": llm_responses,
                        }
                        self.result_manager.save_detail(f"exp5b_{game_name}_{network_name}", "multi", i + 1, self.rounds, detail_data)

                        print("Done")

                    final_payoffs = {k: np.mean(v) for k, v in all_trials_payoffs.items()}
                    coop_rates = {k: np.mean(v) for k, v in all_trials_coop_rates.items()}

                    llm_results = {k: v for k, v in final_payoffs.items() if k.startswith("LLM_")}
                    traditional_results = {k: v for k, v in final_payoffs.items() if not k.startswith("LLM_")}

                    network_results[network_name] = {
                        "payoffs": final_payoffs,
                        "coop_rates": coop_rates,
                        "rankings": sorted(final_payoffs.items(), key=lambda x: x[1], reverse=True),
                        "llm_comparison": llm_results,
                        "traditional_comparison": traditional_results,
                    }

                    print(f"    LLM Avg ranking (Top 5):")
                    llm_ranked = sorted(llm_results.items(), key=lambda x: x[1], reverse=True)
                    for rank, (aid, payoff) in enumerate(llm_ranked[:5], 1):
                        coop = coop_rates.get(aid, 0)
                        print(f"      {rank}. {aid}: {payoff:.1f} (Coop: {coop:.1%})")

                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()
                    network_results[network_name] = {"error": str(e)}

            all_results[game_name] = network_results
            self.result_manager.save_json(game_name, "exp5b_group_dynamics_multi", network_results)
            self.result_manager.save_round_records("exp5b", game_name, "multi", all_round_records)

            fig = _plot_multi_provider_comparison(network_results, game_name, self.providers)
            if fig:
                self.result_manager.save_figure(game_name, "exp5b_group_dynamics_multi", fig)

        self.result_manager.save_experiment_summary("exp5b_group_dynamics_multi", all_results)

        return all_results


def _plot_multi_provider_comparison(network_results: Dict, game_name: str, providers: List[str]) -> Optional[plt.Figure]:
    """绘制多 Provider 对比图"""

    valid_networks = [n for n in network_results if "error" not in network_results[n]]
    if not valid_networks:
        return None

    n_networks = len(valid_networks)
    fig, axes = plt.subplots(1, n_networks, figsize=(7 * n_networks, 6))
    if n_networks == 1:
        axes = [axes]

    # 为不同 provider 设置颜色
    provider_colors = {
        "deepseek": "#4CAF50",  # 绿色
        "openai": "#2196F3",    # 蓝色
        "gemini": "#FF9800",    # 橙色
    }

    for ax, network_name in zip(axes, valid_networks):
        data = network_results[network_name]
        rankings = data["rankings"]
        coop_rates = data["coop_rates"]

        names = [r[0] for r in rankings]
        payoffs = [r[1] for r in rankings]

        # 设置颜色
        colors = []
        for name in names:
            if name.startswith("LLM_"):
                provider = name.replace("LLM_", "")
                colors.append(provider_colors.get(provider, "#9C27B0"))
            else:
                colors.append("#757575")  # 灰色表示传统策略

        bars = ax.barh(range(len(names)), payoffs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("总得分")
        ax.set_title(f"{NETWORK_NAMES_CN.get(network_name, network_name)}")
        ax.invert_yaxis()

        # 在柱子上显示合作率
        for i, (name, payoff) in enumerate(zip(names, payoffs)):
            coop = coop_rates.get(name, 0)
            ax.text(payoff + 0.5, i, f"{coop:.0%}", va='center', fontsize=8)

    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=provider_colors.get(p, "#9C27B0"), label=f"LLM_{p}")
        for p in providers
    ]
    legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor="#757575", label="传统策略"))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(providers)+1, bbox_to_anchor=(0.5, 1.02))

    game_cn = GAME_NAMES_CN.get(game_name, game_name)
    fig.suptitle(f"多 Provider 群体动力学 - {game_cn}", fontsize=14, y=1.08)

    plt.tight_layout()
    return fig


def _plot_group_rankings(network_results: Dict, game_name: str) -> Optional[plt.Figure]:
    """绘制群体动力学排名图"""

    valid_networks = [n for n in network_results if "error" not in network_results[n]]
    if not valid_networks:
        return None

    n_networks = len(valid_networks)
    fig, axes = plt.subplots(1, n_networks, figsize=(6 * n_networks, 5))
    if n_networks == 1:
        axes = [axes]

    for ax, network_name in zip(axes, valid_networks):
        data = network_results[network_name]
        rankings = data["rankings"]

        names = [r[0] for r in rankings]
        payoffs = [r[1] for r in rankings]
        colors = ['steelblue' if 'LLM' in n else 'gray' for n in names]

        ax.barh(range(len(names)), payoffs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("总得分")
        ax.set_title(f"{NETWORK_NAMES_CN.get(network_name, network_name)}")
        ax.invert_yaxis()

    fig.suptitle(f"群体动力学 - {GAME_NAMES_CN.get(game_name, game_name)}", fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================
# 实验6: Baseline 对比
# ============================================================

class Exp6_Baseline(BaseExperiment):
    """实验6: Baseline 对比"""
    name = "exp6"
    description = "Baseline 对比"

    BASELINES = {
        "TitForTat": TitForTat,
        "AlwaysCooperate": AlwaysCooperate,
        "AlwaysDefect": AlwaysDefect,
        "GrimTrigger": GrimTrigger,
        "Pavlov": Pavlov,
        "Random": RandomStrategy,
    }

    def __init__(self, result_manager: ResultManager, **kwargs):
        super().__init__(result_manager, **kwargs)
        self.providers = kwargs.get("providers", ["deepseek", "openai", "gemini"])

    def run(self) -> Dict:
        print_separator(f"实验6: {self.description}")
        print(f"LLM Providers: {self.providers}")
        print(f"LLM vs 经典策略: {list(self.BASELINES.keys())}")
        print(f"Repeats: {self.n_repeats} | Rounds: {self.rounds}")

        all_results = {}

        for game_name in self.games:
            game_config = GAME_REGISTRY[game_name]
            print_game_header(game_name)

            game_results = {}
            all_round_records = []

            for provider in self.providers:
                print(f"\n  Provider: {provider.upper()}")

                baseline_results = {}

                for baseline_name, BaselineClass in self.BASELINES.items():
                    print(f"\n    vs {baseline_name}")

                    payoffs = []
                    coop_rates = []

                    for trial in range(self.n_repeats):
                        print(f"      Trial {trial + 1}/{self.n_repeats}...", end=" ", flush=True)

                        try:
                            llm_strategy = LLMStrategy(
                                provider=provider,
                                mode="hybrid",
                                game_config=game_config,
                            )

                            opponent = BaselineClass()

                            llm_payoff = 0
                            llm_history = []
                            opp_history = []

                            for r in range(self.rounds):
                                llm_action = llm_strategy.choose_action(llm_history, opp_history)
                                opp_action = opponent.choose_action(make_history_tuples(opp_history, llm_history))

                                payoff, _ = get_payoff(game_config, llm_action, opp_action)
                                llm_payoff += payoff

                                llm_response = llm_strategy.raw_responses[-1] if llm_strategy.raw_responses else ""
                                all_round_records.append({
                                    "provider": provider,
                                    "baseline": baseline_name,
                                    "trial": trial + 1,
                                    "round": r + 1,
                                    "llm_response": llm_response,
                                    "llm_action": llm_action.name,
                                    "opp_action": opp_action.name,
                                    "payoff": payoff,
                                    "cumulative_payoff": llm_payoff,
                                })

                                llm_history.append(llm_action)
                                opp_history.append(opp_action)

                            coop_rate = compute_cooperation_rate(llm_history)
                            payoffs.append(llm_payoff)
                            coop_rates.append(coop_rate)

                            detail_data = {
                                "experiment": "exp6_baseline",
                                "game": game_name,
                                "provider": provider,
                                "baseline": baseline_name,
                                "trial": trial + 1,
                                "rounds": self.rounds,
                                "payoff": llm_payoff,
                                "coop_rate": coop_rate,
                                "llm_history": [a.name for a in llm_history],
                                "opp_history": [a.name for a in opp_history],
                                "llm_responses": llm_strategy.raw_responses.copy(),
                            }
                            self.result_manager.save_detail(f"exp6_{game_name}_{baseline_name}", provider, trial + 1, self.rounds, detail_data)

                            print(f"Payoff: {llm_payoff:.1f}, Coop rate: {coop_rate:.1%}")

                        except Exception as e:
                            print(f"Error: {e}")
                            continue

                    baseline_results[baseline_name] = {
                        "payoff": compute_statistics(payoffs),
                        "coop_rate": compute_statistics(coop_rates),
                    }

                game_results[provider] = baseline_results

            all_results[game_name] = game_results

            self.result_manager.save_json(game_name, "exp6_baseline", game_results)
            self.result_manager.save_round_records("exp6", game_name, "all", all_round_records)

            fig = _plot_baseline_multi_provider(game_results, game_name, self.providers, self.BASELINES)
            if fig:
                self.result_manager.save_figure(game_name, "exp6_baseline", fig)

        _print_baseline_summary_multi_provider(all_results, self.providers)
        self.result_manager.save_experiment_summary("exp6_baseline", all_results)

        return all_results


def _plot_baseline_multi_provider(
    game_results: Dict,
    game_name: str,
    providers: List[str],
    baselines: Dict
) -> Optional[plt.Figure]:
    """绘制多 Provider Baseline 对比图"""

    n_providers = len(providers)
    n_baselines = len(baselines)

    fig, axes = plt.subplots(1, n_providers, figsize=(6 * n_providers, 6))
    if n_providers == 1:
        axes = [axes]

    # 为不同 provider 设置颜色
    provider_colors = {
        "deepseek": "#4CAF50",
        "openai": "#2196F3",
        "gemini": "#FF9800",
    }

    baseline_names = list(baselines.keys())

    for ax, provider in zip(axes, providers):
        if provider not in game_results:
            continue

        baseline_data = game_results[provider]
        means = [baseline_data[b]["payoff"]["mean"] for b in baseline_names]
        stds = [baseline_data[b]["payoff"]["std"] for b in baseline_names]

        x = np.arange(len(baseline_names))
        color = provider_colors.get(provider, "#9C27B0")
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.8)

        ax.set_ylabel("LLM 得分")
        ax.set_title(f"{provider.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names, rotation=45, ha='right')

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    game_cn = GAME_NAMES_CN.get(game_name, game_name)
    fig.suptitle(f"LLM vs Baselines (多模型对比) - {game_cn}", fontsize=14)
    plt.tight_layout()
    return fig


def _print_baseline_summary_multi_provider(results: Dict, providers: List[str]):
    """打印多 Provider Baseline 对比汇总"""
    print_separator("汇总: LLM vs Baselines (多模型)")

    for game_name, provider_stats in results.items():
        cn_name = GAME_NAMES_CN.get(game_name, game_name)
        print(f"\n{cn_name}:")

        for provider in providers:
            if provider not in provider_stats:
                continue

            print(f"\n  {provider.upper()}:")
            print(f"    {'Opponent':<16} {'Payoff':<18} {'Coop Rate':<12}")
            print(f"    {'-' * 46}")

            baseline_data = provider_stats[provider]
            for baseline, stats in baseline_data.items():
                pay = stats["payoff"]
                coop = stats["coop_rate"]
                pay_str = f"{pay['mean']:.1f} ± {pay['std']:.1f}"
                coop_str = f"{coop['mean']:.1%}"
                print(f"    {baseline:<16} {pay_str:<18} {coop_str:<12}")


# ============================================================
# 主函数
# ============================================================

# ============================================================
# 实验注册表
# ============================================================

EXPERIMENTS = {
    "exp1": Exp1_PureVsHybrid,
    "exp2": Exp2_MemoryWindow,
    "exp3": Exp3_MultiLLM,
    "exp4": Exp4_CheapTalk3LLM,
    "exp4b": Exp4b_CheapTalk1v1,
    "exp5": Exp5_GroupDynamics,
    "exp5b": Exp5b_GroupDynamicsMulti,
    "exp6": Exp6_Baseline,
}

# 别名映射（兼容旧命令）
EXPERIMENT_ALIASES = {
    "pure_hybrid": "exp1",
    "window": "exp2",
    "multi_llm": "exp3",
    "cheap_talk": "exp4",
    "cheap_talk_1v1": "exp4b",
    "group_single": "exp5",
    "group": "exp5b",
    "group_multi": "exp5b",
    "baseline": "exp6",
}


def print_usage():
    """打印使用说明"""
    print("""
博弈论 LLM 研究实验脚本 v10
==========================

用法:
  python research.py <experiment> [options]

实验列表:
  exp1          - 实验1: Pure vs Hybrid LLM
  exp2          - 实验2: 记忆视窗对比
  exp3          - 实验3: 多 LLM 对比
  exp4          - 实验4: Cheap Talk 三方对战 (3 LLM Round-Robin)
  exp4b         - 实验4b: Cheap Talk 一对一 (支持指定双方 provider)
  exp5          - 实验5: 群体动力学（单 Provider）
  exp5b         - 实验5b: 群体动力学（DeepSeek/OpenAI/Gemini 三模型）
  exp6          - 实验6: Baseline 对比（DeepSeek/OpenAI/Gemini 三模型）
  all           - 运行全部实验

旧命令兼容:
  pure_hybrid -> exp1,  window -> exp2,  multi_llm -> exp3
  cheap_talk -> exp4,   cheap_talk_1v1 -> exp4b
  group_single -> exp5, group/group_multi -> exp5b, baseline -> exp6

选项:
  --provider    LLM 提供商 (deepseek/openai/gemini)    [默认: deepseek]
  --provider1   exp4b 实验的 Player1 模型              [默认: 同 --provider]
  --provider2   exp4b 实验的 Player2 模型              [默认: 同 --provider]
  --repeats     重复次数                               [默认: 3]
  --rounds      每次轮数                               [默认: 20]
  --games       指定博弈 (pd/snowdrift/stag_hunt/harmony/all) [默认: all]
  --n_agents    群体动力学实验的智能体数量             [默认: 10]

帮助:
  -h, --help    显示此帮助信息

示例:
  python research.py exp1
  python research.py exp4                          # 3 LLM 三方对战
  python research.py exp4b --provider1 openai --provider2 gemini
  python research.py exp5b --rounds 30 --n_agents 15
  python research.py all --provider openai --repeats 5
""")


def main():
    if len(sys.argv) < 2:
        experiment = "all"
        print("未指定实验，默认运行全部实验...")
    else:
        experiment = sys.argv[1].lower()

        if experiment in ["-h", "--help", "help"]:
            print_usage()
            return

    # 转换别名
    experiment = EXPERIMENT_ALIASES.get(experiment, experiment)

    # 解析参数
    provider = DEFAULT_CONFIG["provider"]
    n_repeats = DEFAULT_CONFIG["n_repeats"]
    rounds = DEFAULT_CONFIG["rounds"]
    games = None
    n_agents = 10
    provider1 = None
    provider2 = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider1" and i + 1 < len(sys.argv):
            provider1 = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider2" and i + 1 < len(sys.argv):
            provider2 = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--repeats" and i + 1 < len(sys.argv):
            n_repeats = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--rounds" and i + 1 < len(sys.argv):
            rounds = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--n_agents" and i + 1 < len(sys.argv):
            n_agents = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--games" and i + 1 < len(sys.argv):
            game_arg = sys.argv[i + 1].lower()
            if game_arg == "all":
                games = None
            elif game_arg == "pd":
                games = ["prisoners_dilemma"]
            elif game_arg == "snowdrift":
                games = ["snowdrift"]
            elif game_arg == "stag_hunt":
                games = ["stag_hunt"]
            else:
                games = [game_arg]
            i += 2
        else:
            i += 1

    # 创建结果管理器
    result_manager = ResultManager()

    # 公共参数
    common_kwargs = {
        "provider": provider,
        "n_repeats": n_repeats,
        "rounds": rounds,
        "games": games,
        "n_agents": n_agents,
        "provider1": provider1 if provider1 else provider,
        "provider2": provider2 if provider2 else provider,
    }

    # 保存实验配置
    config = {
        "experiment": experiment,
        **common_kwargs,
        "games": games or list(GAME_REGISTRY.keys()),
        "timestamp": result_manager.timestamp,
    }
    result_manager.save_config(config)

    # 运行实验
    all_results = {}

    # 确定要运行的实验列表
    if experiment == "all":
        exp_to_run = list(EXPERIMENTS.keys())
    elif experiment in EXPERIMENTS:
        exp_to_run = [experiment]
    else:
        print(f"未知实验: {experiment}")
        print_usage()
        return

    # 按顺序运行实验
    for exp_name in exp_to_run:
        ExpClass = EXPERIMENTS[exp_name]
        exp = ExpClass(result_manager, **common_kwargs)
        results = exp.run()
        all_results[exp_name] = results

    # 保存汇总
    result_manager.save_summary(all_results)

    print_separator("实验完成")
    print(f"Results dir: {result_manager.root_dir}")
    print(f"Total experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
