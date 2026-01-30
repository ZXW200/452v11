# CLAUDE.md - 博弈论 LLM 多智能体研究项目
每次回复都要加主人
## 项目概述

本项目研究 LLM（大语言模型）在博弈论场景下的决策行为，包括囚徒困境、雪堆博弈、猎鹿博弈等经典博弈。通过多种实验设计，探索 LLM 的合作/背叛策略、记忆能力、语言交流（Cheap Talk）对决策的影响等。

## 目录结构

```
452111-main/
├── research.py          # 主实验脚本 (v16)，包含所有实验类
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
    def choose_action(self, history: List[Tuple], opponent_name, opponent_message)
    def generate_message(self, history: List[Tuple], opponent_name)  # Cheap Talk 消息生成
    def reset()  # 重置状态（total_payoff, raw_responses, parser stats）

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

## 版本更新规则

**每次创建 Pull Request 时，必须完成以下检查和更新：**

### 必须更新的版本号

1. **版本号** (`game_theory/__init__.py`)
   ```python
   __version__ = "x.y.z"
   ```

2. **Usage 版本号** (`research.py` 的 `print_usage()` 函数)
   ```
   博弈论 LLM 研究实验脚本 vN
   ```

3. **README 版本号** (`README.md` 标题)
   ```
   博弈论 LLM 多智能体研究框架 vX.Y.Z (vN)
   ```

### 必须检查并同步的内容

4. **README.md** - 检查以下部分是否需要更新：
   - 输出结构（如果修改了 ResultManager）
   - 命令行选项（如果添加/修改了参数）
   - 实验列表（如果添加/修改了实验）
   - 版本历史（添加本次改动）

5. **print_usage()** - 检查 `research.py` 的 `print_usage()` 函数：
   - 实验列表是否完整
   - 命令行参数说明是否准确

6. **更新日志**（在下方记录，包含本次修改内容）

## 更新日志

### v0.5.2 (v16)
- **重构 exp5b 智能体分配**
  - 固定 3 LLM（每个 provider 1 个）+ 8 经典策略 = 11 agents
  - 移除动态 n_agents 参数，改为固定分配
  - 策略列表：TitForTat, TitForTwoTats, GenerousTitForTat, Extort2, Pavlov, GrimTrigger, AlwaysDefect, RandomStrategy
- **新增经典策略** (`game_theory/strategies.py`)
  - GenerousTitForTat: 宽容的以牙还牙，有概率原谅背叛
  - Extort2: Press & Dyson (2012) 零行列式勒索策略
- **修复 exp5b 崩溃**
  - `self.n_agents` 在移除 `n_agents` 参数后未定义，导致运行时崩溃
- **更换 OpenAI base URL**
  - 从 `api.openai.com` 改为 `hiapi.online` 代理

### v0.5.1 (v15)
- **修复 Exp4 除零错误**
  - `research.py:1249` 当 `coop_rate_dict` 为空时添加防护
- **添加 providers 参数验证**
  - Exp3: 至少需要 1 个 provider
  - Exp4: 至少需要 2 个 provider（用于配对对战）
  - Exp5b: 至少需要 1 个 provider
- **增强策略健壮性**
  - GrimTrigger: 添加自动状态重置（history 为空但 triggered 为 true 时）
  - GradualStrategy: 添加自动状态重置（history 为空但有未完成状态时）
  - 防止策略实例复用时的状态泄漏

### v0.5.0 (v14)
- **修复 LLMStrategy 与 GameSimulation 参数格式不匹配**
  - `choose_action(history: List[Tuple], ...)` 统一为元组列表格式
  - `generate_message(history: List[Tuple], ...)` 同步更新
  - research.py 调用点使用 `make_history_tuples()` 转换格式
- **修复 AnomalyRecorder 输出目录错误**
  - 从 `stats/` 改为 `anomalies/`，与目录结构设计一致
- **修复 pure/hybrid 模式历史窗口不一致**
  - pure 模式合作率计算现在也应用 `history_window` 限制
- **添加除零保护**
  - `_build_hybrid_prompt` 中 `opp_coop_rate` 计算添加边界检查

### v0.4.0 (v13)
- 重构 ResultManager 输出目录结构
  - 新目录: raw/, rounds/, stats/, figures/, anomalies/
  - raw/: 原始试验数据 (JSON)
  - rounds/: 轮次数据 (CSV)，便于 pandas 分析
  - stats/: 统计汇总
  - figures/: 所有图表
  - anomalies/: 异常记录
- 新增统一保存接口: save_trial(), save_rounds(), save_stats(), save_fig(), save_anomaly()
- 保留旧方法兼容性，自动输出到新目录

### v0.3.0 (v12)
- 统一版本号管理（research.py、__init__.py、CLAUDE.md 版本号一致）
- 修复裸异常处理问题（llm_api.py 中 `except:` 改为 `except Exception:`）
- 重写 README.md 文档

### v0.2.0 (v11)
- 修复 exp5/exp5b 策略名泄露问题：经典策略命名从 `TitForTat_1` 改为 `Agent_N`
- 添加 `strategy_map` 到结果中，用于分析时映射 agent 到真实策略类型
- 更新 exp5/exp5b 经典策略列表：
  - 新：RandomStrategy, TitForTat, Pavlov, GradualStrategy, ProbabilisticCooperator, SuspiciousTitForTat
  - 移除：AlwaysCooperate, AlwaysDefect, GrimTrigger

### v0.1.0 (v10)
- 初始版本
