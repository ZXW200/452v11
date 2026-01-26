# 实验3: 多 LLM 对比 (Multi-LLM)

## 实验配置

| 参数 | 值 |
|------|-----|
| LLM Providers | DeepSeek / OpenAI / Gemini |
| 重复次数 | 30 |
| 每次轮数 | 20 |
| 对手策略 | TitForTat |
| 模式 | Hybrid |

## 实验结果汇总

### 囚徒困境 (Prisoner's Dilemma)

| LLM | Payoff | Coop Rate |
|-----|--------|-----------|
| gemini | **60.4 ± 0.8** | 98.5% |
| openai | **60.0 ± 0.0** | **100.0%** |
| deepseek | 57.2 ± 2.7 | 86.5% |

**发现**: OpenAI 表现最稳定（方差=0，100%合作），Gemini 次之，DeepSeek 合作率最低(86.5%)且方差较大。

### 雪堆博弈 (Snowdrift)

| LLM | Payoff | Coop Rate |
|-----|--------|-----------|
| deepseek | **60.0 ± 0.0** | 99.7% |
| openai | **60.0 ± 0.0** | 99.8% |
| gemini | 57.5 ± 3.7 | 85.5% |

**发现**: DeepSeek 和 OpenAI 表现完美，Gemini 在雪堆博弈中表现下降(85.5%合作率)。

### 猎鹿博弈 (Stag Hunt)

| LLM | Payoff | Coop Rate |
|-----|--------|-----------|
| deepseek | **100.0 ± 0.0** | **100.0%** |
| openai | **100.0 ± 0.0** | **100.0%** |
| gemini | 99.8 ± 1.3 | 99.8% |

**发现**: 三者表现接近完美，猎鹿博弈结构促进合作。

## 模型对比总结

| 博弈类型 | 最佳模型 | 最差模型 |
|----------|----------|----------|
| 囚徒困境 | OpenAI (100%) | DeepSeek (86.5%) |
| 雪堆博弈 | DeepSeek/OpenAI (≈100%) | Gemini (85.5%) |
| 猎鹿博弈 | DeepSeek/OpenAI (100%) | Gemini (99.8%) |

## 关键结论

1. **OpenAI 最稳定**: 在所有博弈中方差最小，合作率最高
2. **DeepSeek 囚徒困境弱**: 在囚徒困境中合作率仅 86.5%，但雪堆和猎鹿表现优秀
3. **Gemini 雪堆博弈弱**: 在雪堆博弈中合作率仅 85.5%，其他博弈表现良好
4. **博弈结构影响**: 猎鹿博弈中三者差异最小，囚徒困境差异最大

## 数据文件

- JSON: `results/{timestamp}/{game}/exp3_multi_llm.json`
- 图表: `results/{timestamp}/{game}/exp3_multi_llm.png`
- CSV汇总: `results/{timestamp}/summary/multi_llm.csv`
