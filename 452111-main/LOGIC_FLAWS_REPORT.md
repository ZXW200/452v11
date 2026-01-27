# 实验逻辑漏洞和缺陷分析报告

## 分析范围
- `research.py` - 主实验脚本 (exp1-exp6)
- `game_theory/llm_strategy.py` - LLM 策略实现
- `game_theory/simulation.py` - 博弈仿真引擎
- `game_theory/strategies.py` - 经典策略库
- `game_theory/network.py` - 网络拓扑
- `game_theory/games.py` - 博弈定义

---

## 严重漏洞 (Critical)

### 1. exp5/exp5b 中 LLMStrategy 接口调用错误

**位置**: `simulation.py:184-186` + `research.py:1779`

**问题描述**: simulation.py 中调用策略的方式与 LLMStrategy 的接口不兼容。

```python
# simulation.py:184-186
def execute_decision(task):
    strategy, history, opponent_name = task
    return strategy.choose_action(history, opponent_name)  # 只传2个参数

# 但 LLMStrategy.choose_action 期望:
def choose_action(self, my_history: List, opponent_history: List, opponent_name: str = "Opponent", ...)
```

**后果**: 当调用 LLMStrategy 时：
- `my_history` = `[(Action, Action), ...]`（元组列表）
- `opponent_history` = `"Agent_2"`（字符串！）

在 `_format_history` 中会用 `zip(my_history, opponent_history)` 遍历，将字符串 "Agent_2" 当作字符序列 `['A','g','e','n','t'...]` 处理，导致**完全错误的历史格式化**！

**影响**: exp5/exp5b 的实验结果完全不可信。

---

### 2. exp4 三方对战的历史记录混淆

**位置**: `research.py:1131-1143`

**问题描述**: Round-Robin 模式下，每个 LLM 与两个不同对手对战，但历史记录按 provider 聚合，而非按对手区分。

```python
histories = {normalize_provider_name(p): [] for p in self.providers}
# 每轮对战后（与两个不同对手）
histories[d1].append(Action[match["p1_action"]])  # 只记录自己的action
```

**后果**:
1. LLM 只能看到自己的历史，看不到对手的历史
2. 与不同对手的交互记录混在一起
3. `choose_action(histories[d1], histories[d2])` 传递的是错误的"对手历史"

---

### 3. exp4 Cheap Talk 中消息与动作的时序问题

**位置**: `research.py:1088-1098`

```python
# 先生成消息（双方同时）
msg1 = llm1.generate_message(histories[d1], histories[d2], d2)
msg2 = llm2.generate_message(histories[d2], histories[d1], d1)

# 再选择动作（用对方的消息）
action1 = llm1.choose_action(..., opponent_message=msg2)
action2 = llm2.choose_action(..., opponent_message=msg1)
```

**后果**: 消息是在**本轮开始时**生成的，但传递给对手的是**对手也同时生成**的消息。这意味着双方都看不到对方的实际消息就已经决策了。正确的做法应该是：先 A 发消息，B 看到后回复，然后双方决策。

---

## 中等漏洞 (Medium)

### 4. exp5/exp5b 的 LLM 响应与 game_history 不匹配

**位置**: `research.py:1808-1811`

```python
responses = getattr(agent.strategy, 'raw_responses', [])
for r_idx, hist in enumerate(agent.game_history):
    llm_response = responses[r_idx] if r_idx < len(responses) else ""
```

**问题**: 在完全连接网络中，每轮每个 agent 可能与 N-1 个对手交互，产生 N-1 条 game_history 记录。但 LLM 每轮只调用一次（并行化后），`raw_responses` 长度与 `game_history` 长度不一致。

---

### 5. promise_keeping 分析逻辑缺陷

**位置**: `research.py:1675-1696`

```python
cooperation_keywords = ["cooperate", "trust", "promise", ...]
for msg, action in zip(messages, actions):
    if msg and any(kw in msg.lower() for kw in cooperation_keywords):
        promise_count += 1
        if action == Action.COOPERATE:
            kept_count += 1
```

**问题**:
1. 不考虑上下文，"I will **not** cooperate" 也被认为是承诺合作
2. "I don't trust you" 被计算为 trust 承诺
3. messages 和 actions 假设一一对应，但时序关系不明确

---

### 6. exp2 静默修改 rounds 参数

**位置**: `research.py:765-766`

```python
if self.rounds < 30:
    self.rounds = 30  # 记忆视窗实验至少30轮
```

**问题**: 用户指定 `--rounds 20`，但实际运行 30 轮，没有任何警告输出。

---

### 7. 置信区间计算使用 z 而非 t 分布

**位置**: `research.py:348-353`

```python
if n > 1:
    se = std / np.sqrt(n)
    ci_low = mean - 1.96 * se  # z=1.96 假设大样本
```

**问题**: 默认 `n_repeats=3`，样本量太小，应使用 t 分布而非正态近似。z=1.96 在 n=3 时会低估置信区间宽度。

---

### 8. Provider 双向名称映射可能导致调试困难

**位置**: `research.py:55-67`

```python
def normalize_provider_name(provider: str) -> str:
    name_map = {"moonshot": "gemini"}  # moonshot 显示为 gemini

def resolve_provider_name(provider: str) -> str:
    name_map = {"gemini": "moonshot"}  # gemini 实际调用 moonshot
```

**问题**: 用户输入 gemini → 实际调用 moonshot API → 结果显示为 gemini。API 出错时难以追踪真正调用的服务。

---

## 轻微漏洞 (Minor)

### 9. 异常记录器默认阈值不合理

**位置**: `research.py:391`

```python
threshold: float = 1.0  # 低于 100% 合作率就记录
```

**问题**: 在囚徒困境等博弈中，背叛是理性策略，合作率低于 100% 不应被视为"异常"。

---

### 10. exp5/exp5b 的 strategy_map 在循环内重置

**位置**: `research.py:1736-1737`

```python
for i in range(self.n_repeats):
    strategy_map = {}  # 每次 trial 重新创建
```

**问题**: 不同 trial 可能有不同的策略分配，但最终保存的 `strategy_map` 只反映最后一次的映射。

---

### 11. 解析失败时的随机回退影响可重复性

**位置**: `llm_strategy.py:237-251`

```python
def _random_fallback(self, response: str, reason: str) -> ParseResult:
    action = random.choice([self.Action.COOPERATE, self.Action.DEFECT])
```

**问题**: 没有设置随机种子，影响实验可重复性。虽然目的是消除偏差，但也引入了不可控随机性。

---

### 12. LLMStrategy 与传统 Strategy 的接口不一致

**位置**: `llm_strategy.py:360` vs `strategies.py:30`

```python
# LLMStrategy
def choose_action(self, my_history: List, opponent_history: List, ...)

# 传统 Strategy
def choose_action(self, history: List[Tuple[Action, Action]], ...)
```

**问题**: 需要使用 `make_history_tuples` 转换，增加出错风险。接口不统一违反了多态设计原则。

---

### 13. exp4b 的 coop_rate 统计可能产生无意义结果

**位置**: `research.py:1563-1568`

```python
avg_coop_mean = (mode_stats["player1_coop_rate"]["mean"] + mode_stats["player2_coop_rate"]["mean"]) / 2
avg_coop_std = (mode_stats["player1_coop_rate"]["std"] + mode_stats["player2_coop_rate"]["std"]) / 2
```

**问题**: 如果某个 player 的所有 trial 都失败（results 为空列表），`compute_statistics` 返回 mean=0, std=0，但这种"平均"操作在统计上是无意义的。

---

### 14. GrimTrigger 遍历全部历史检查背叛

**位置**: `strategies.py:146-150`

```python
for my_action, opp_action in history:
    if opp_action == Action.DEFECT:
        self.triggered = True
        return Action.DEFECT
```

**问题**: 每次决策都遍历全部历史，O(n) 复杂度。当 `self.triggered=True` 时仍然遍历是冗余的（虽然已在第 143 行提前返回）。

---

### 15. exp6 只测试 LLM vs 经典策略

**位置**: `research.py:2189-2322`

**问题**: Baseline 实验只测试 LLM 对战经典策略，无法评估 LLM 之间对战的表现差异。这可能遗漏重要的实验维度。

---

## 总结

| 严重程度 | 数量 | 主要影响 |
|---------|------|---------|
| 严重 | 3 | 实验结果不可靠，LLM 接收错误输入 |
| 中等 | 5 | 数据记录不准确，统计分析有偏 |
| 轻微 | 7 | 代码质量、可维护性问题 |

**最紧急需要修复**:
1. simulation.py 中 LLMStrategy 的调用接口问题（导致 exp5/exp5b 结果完全不可信）
2. exp4 的历史记录混淆问题（导致 LLM 看不到正确的对手历史）
