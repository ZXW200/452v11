# CLAUDE.md - 博弈论 LLM 多智能体研究项目

## 项目概述

本项目研究 LLM（大语言模型）在博弈论场景下的决策行为，包括囚徒困境、雪堆博弈、猎鹿博弈等经典博弈。通过多种实验设计，探索 LLM 的合作/背叛策略、记忆能力、语言交流（Cheap Talk）对决策的影响等。

## 目录结构

```
452111-main/
├── research.py          # 主实验脚本 (v10)，包含所有实验类
├── game_theory/         # 博弈论核心模块
│   ├── games.py         # 博弈定义（囚徒困境、雪堆、猎鹿）
│   ├── llm_api.py       # LLM API 封装（DeepSeek/OpenAI/Gemini）
│   ├── llm_strategy.py  # LLM 策略实现
│   ├── strategies.py    # 经典策略（TitForTat、Pavlov 等）
│   ├── network.py       # 网络拓扑（完全连接、小世界、无标度）
│   ├── simulation.py    # 博弈仿真引擎
│   └── prompts/         # LLM 提示词模板
├── README.md
└── results/             # 实验结果输出目录（自动生成）
```

## 运行实验

```bash
# 单个实验
python research.py exp1                    # Pure vs Hybrid
python research.py exp2                    # 记忆视窗
python research.py exp3                    # 多 LLM 对比
python research.py exp4                    # Cheap Talk 三方对战
python research.py exp4b                   # Cheap Talk 一对一
python research.py exp5                    # 群体动力学（单 Provider）
python research.py exp5b                   # 群体动力学（多 Provider）
python research.py exp6                    # Baseline 对比

# 全部实验
python research.py all

# 常用参数
--provider deepseek/openai/gemini          # LLM 提供商
--provider1 / --provider2                  # exp4b 双方模型
--repeats 3                                # 重复次数
--rounds 20                                # 每次轮数
--games pd/snowdrift/stag_hunt/all         # 博弈类型
--n_agents 10                              # 群体实验智能体数量
```

## 实验列表

| 实验 | 类名 | 说明 |
|------|------|------|
| exp1 | Exp1_PureVsHybrid | Pure（LLM 自己分析历史）vs Hybrid（代码辅助分析） |
| exp2 | Exp2_MemoryWindow | 不同记忆窗口长度（5/10/20/全部）对决策的影响 |
| exp3 | Exp3_MultiLLM | DeepSeek vs OpenAI vs Gemini 对比 |
| exp4 | Exp4_CheapTalk3LLM | 3 个 LLM Round-Robin 语言交流博弈 |
| exp4b | Exp4b_CheapTalk1v1 | 指定双方 LLM 的一对一语言交流博弈 |
| exp5 | Exp5_GroupDynamics | 单 Provider 群体动力学 |
| exp5b | Exp5b_GroupDynamicsMulti | 多 Provider 群体动力学 |
| exp6 | Exp6_Baseline | LLM vs 经典策略（TitForTat、AlwaysCooperate 等） |

## 代码规范

- 实验类继承 `BaseExperiment`，实现 `run()` 和 `_print_summary()` 方法
- 实验注册在 `EXPERIMENTS` 字典，别名在 `EXPERIMENT_ALIASES`
- 结果通过 `ResultManager` 保存，支持 JSON、CSV、PNG、transcript
- 博弈类型使用 `GAME_REGISTRY` 获取配置
- 统计量使用 `compute_statistics()` 计算（含 95% 置信区间）

## 关键类和函数

```python
# 实验基类
class BaseExperiment:
    def __init__(self, result_manager, **kwargs)
    def run(self) -> Dict
    def _print_summary(self, results)

# LLM 策略
class LLMStrategy:
    def choose_action(self, my_history, opp_history, ...)
    def generate_message(...)  # Cheap Talk 消息生成

# 结果管理
class ResultManager:
    def save_json(game_name, exp_name, data)
    def save_figure(game_name, exp_name, fig)
    def save_round_records(...)
```

## 常见修改

### 添加新实验
1. 创建继承 `BaseExperiment` 的新类
2. 实现 `run()` 和 `_print_summary()` 方法
3. 在 `EXPERIMENTS` 字典中注册
4. 可选：在 `EXPERIMENT_ALIASES` 添加别名
5. 更新 `print_usage()` 说明

### 添加新博弈类型
在 `game_theory/games.py` 中添加 `GameConfig` 并注册到 `GAME_REGISTRY`

### 添加新 LLM Provider
在 `game_theory/llm_api.py` 中添加新的 API 调用函数

## 注意事项

- 默认重复次数为 3 次，论文建议 30 次
- exp2 记忆视窗实验会自动将 rounds 设为至少 30 轮
- 结果保存在 `results/{timestamp}/` 目录下
- API Key 需要在环境变量或配置文件中设置
