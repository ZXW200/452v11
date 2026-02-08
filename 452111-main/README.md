# Game Theory LLM Multi-Agent Research

博弈论 LLM 多智能体研究框架 v0.6.0 (v17)

研究大语言模型在经典博弈场景（囚徒困境、雪堆博弈、猎鹿博弈）中的决策行为。

## 安装依赖

```bash
pip install numpy matplotlib requests
# 可选：高级网络拓扑功能
pip install networkx
```

## 配置 API

通过环境变量设置 API Key：

```bash
export DEEPSEEK_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

或使用配置向导：

```bash
python game_theory/llm_api.py setup
```

## 实验列表

| 实验 | 命令 | 说明 |
|------|------|------|
| exp1 | `python research.py exp1` | Pure vs Hybrid - LLM自己分析 vs 代码辅助 |
| exp2 | `python research.py exp2` | 记忆视窗对比 - 5/10/20/全部历史 |
| exp3 | `python research.py exp3` | 多LLM对比 - DeepSeek vs GPT vs Gemini |
| exp4 | `python research.py exp4` | Cheap Talk 三方对战 - 3个LLM Round-Robin 语言交流 |
| exp4b | `python research.py exp4b` | Cheap Talk 一对一 - 指定双方LLM的语言交流博弈 |
| exp5 | `python research.py exp5` | 群体动力学（3 LLM + 8 经典策略 = 11 agents） |
| exp5b | `python research.py exp5b` | 群体动力学（3 LLM + 8 经典策略 = 11 agents） |
| exp6 | `python research.py exp6` | Baseline 对比 - LLM vs 经典策略 |
| all | `python research.py all` | 运行全部实验 |

### 旧命令兼容

| 旧命令 | 对应实验 |
|--------|----------|
| pure_hybrid | exp1 |
| window | exp2 |
| multi_llm | exp3 |
| cheap_talk | exp4 |
| cheap_talk_1v1 | exp4b |
| group_single | exp5 |
| group / group_multi | exp5b |
| baseline | exp6 |

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--provider` | LLM 提供商 (deepseek/openai/gemini) | deepseek |
| `--provider1` | exp4b 的 Player1 模型 | 同 --provider |
| `--provider2` | exp4b 的 Player2 模型 | 同 --provider |
| `--repeats` | 重复次数 | 3 |
| `--rounds` | 每次轮数 | 20 |
| `--games` | 博弈类型 (pd/snowdrift/stag_hunt/all) | all |
| `--n_agents` | 群体动力学智能体数量（exp5/exp5b 固定为 11） | 11 |
| `-h, --help` | 显示帮助信息 | - |

## 使用示例

```bash
# 显示帮助
python research.py --help

# 运行单个实验
python research.py exp1
python research.py exp6

# 指定 LLM 提供商
python research.py exp1 --provider openai

# Cheap Talk 跨模型对战
python research.py exp4b --provider1 openai --provider2 gemini

# 群体动力学实验（15个智能体，30轮）
python research.py exp5b --n_agents 15 --rounds 30

# 只跑囚徒困境
python research.py exp1 --games pd

# 运行全部实验，重复5次
python research.py all --repeats 5
```

## 输出结构

```
results/{timestamp}/
├── config.json               # 实验配置
├── summary.json              # 总结
├── raw/                      # 原始试验数据 (JSON)
│   └── {exp}_{game}_{condition}_trial{N}.json
├── rounds/                   # 轮次数据 (CSV)
│   └── {exp}_{game}_{condition}_rounds.csv
├── stats/                    # 统计汇总
│   └── {exp}_summary.csv
├── figures/                  # 图表
│   └── {exp}_{game}_{condition}.png
└── anomalies/                # 异常记录
    └── {exp}_anomalies.csv
```

## 项目结构

```
452111-main/
├── research.py              # 主实验脚本
├── game_theory/
│   ├── games.py             # 博弈定义和收益矩阵
│   ├── strategies.py        # 经典博弈策略 (TitForTat, Pavlov 等)
│   ├── llm_strategy.py      # LLM 决策策略
│   ├── llm_api.py           # 统一 LLM API 接口
│   ├── simulation.py        # 群体动力学仿真引擎
│   ├── network.py           # 网络拓扑 (完全连接、小世界等)
│   └── prompts/             # Prompt 模板
└── README.md
```

## 支持的博弈类型

| 博弈 | 命令参数 | 说明 |
|------|----------|------|
| 囚徒困境 | pd | Prisoner's Dilemma |
| 雪堆博弈 | snowdrift | Snowdrift Game |
| 猎鹿博弈 | stag_hunt | Stag Hunt |

## 支持的 LLM

- **DeepSeek** - 默认，性价比最高
- **OpenAI** - GPT-4o
- **Gemini** - Gemini 2.0 Flash
- **Ollama** - 本地模型

## 版本历史

### v0.6.0 (v17)
- 移除和谐博弈 (Harmony Game)：合作占优策略无分析价值，聚焦三种博弈
- 移除 moonshot 代理层：Gemini 直连原生 API (gemini-2.0-flash)
- 统一 exp5 智能体配置：与 exp5b 一致，3 LLM + 8 经典策略 = 11 agents
- 清理未使用代码：移除 plot_comparison_bar()、_trigger_reflection() 空桩、未用 imports
- 补充中英文双语注释：所有模块添加英文注释

### v0.5.2 (v16)
- 重构 exp5b 智能体分配：固定 3 LLM（每 provider 1个）+ 8 经典策略 = 11 agents
- 新增经典策略：GenerousTitForTat（宽容以牙还牙）、Extort2（零行列式勒索策略）
- 更新 exp5b 经典策略列表：TitForTat, TitForTwoTats, GenerousTitForTat, Extort2, Pavlov, GrimTrigger, AlwaysDefect, RandomStrategy
- 修复 exp5b 崩溃：移除未使用的 n_agents 参数后 self.n_agents 未定义
- 更换 OpenAI base URL 为 hiapi.online 代理

### v0.5.1 (v15)
- 修复 Exp4 除零错误：`coop_rate_dict` 为空时的防护
- 添加 providers 参数验证：Exp3/Exp4/Exp5b 防止空列表输入
- 增强策略健壮性：GrimTrigger/GradualStrategy 添加自动状态重置

### v0.5.0 (v14)
- 修复 LLMStrategy 与 GameSimulation 参数格式不匹配
- 修复 AnomalyRecorder 输出目录错误
- 修复 pure/hybrid 模式历史窗口不一致
- 添加除零保护

### v0.4.0 (v13)
- 重构输出目录结构：raw/, rounds/, stats/, figures/, anomalies/
- 新增统一保存接口

### v0.3.0 (v12)
- 统一版本号管理
- 修复异常处理问题
- 更新文档

### v0.2.0 (v11)
- 修复 exp5/exp5b 策略名泄露问题：改为 Agent_N 命名
- 添加 strategy_map 用于分析时映射
- 更新策略列表
