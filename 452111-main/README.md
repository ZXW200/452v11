# Game Theory LLM Multi-Agent Research

Game Theory LLM Multi-Agent Research Framework v0.6.0 (v17)

Research on LLM decision-making behavior in classic game theory scenarios (Prisoner's Dilemma, Snowdrift Game, Stag Hunt).

## Installation

```bash
pip install numpy matplotlib requests
# Optional: advanced network topology features
pip install networkx
```

## API Configuration

Set API keys via environment variables:

```bash
export DEEPSEEK_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

Or use the setup wizard:

```bash
python game_theory/llm_api.py setup
```

## Experiment List

| Experiment | Command | Description |
|------|------|------|
| exp1 | `python research.py exp1` | Pure vs Hybrid - LLM self-analysis vs code-assisted |
| exp2 | `python research.py exp2` | Memory window comparison - 5/10/20/full history |
| exp3 | `python research.py exp3` | Multi-LLM comparison - DeepSeek vs GPT vs Gemini |
| exp4 | `python research.py exp4` | Cheap Talk three-way - 3 LLM Round-Robin with language communication |
| exp4b | `python research.py exp4b` | Cheap Talk one-on-one - specify both LLMs for language communication game |
| exp5 | `python research.py exp5` | Group dynamics (3 LLM + 8 classic strategies = 11 agents) |
| exp5b | `python research.py exp5b` | Group dynamics (3 LLM + 8 classic strategies = 11 agents) |
| exp6 | `python research.py exp6` | Baseline comparison - LLM vs classic strategies |
| all | `python research.py all` | Run all experiments |

### Legacy Command Aliases

| Legacy Command | Corresponding Experiment |
|--------|----------|
| pure_hybrid | exp1 |
| window | exp2 |
| multi_llm | exp3 |
| cheap_talk | exp4 |
| cheap_talk_1v1 | exp4b |
| group_single | exp5 |
| group / group_multi | exp5b |
| baseline | exp6 |

## Command Line Options

| Option | Description | Default |
|------|------|--------|
| `--provider` | LLM provider (deepseek/openai/gemini) | deepseek |
| `--provider1` | Player1 model for exp4b | same as --provider |
| `--provider2` | Player2 model for exp4b | same as --provider |
| `--repeats` | Number of repeats | 3 |
| `--rounds` | Rounds per repeat | 20 |
| `--games` | Game type (pd/snowdrift/stag_hunt/all) | all |
| `--n_agents` | Number of agents for group dynamics (fixed at 11 for exp5/exp5b) | 11 |
| `-h, --help` | Show help information | - |

## Examples

```bash
# Show help
python research.py --help

# Run a single experiment
python research.py exp1
python research.py exp6

# Specify LLM provider
python research.py exp1 --provider openai

# Cheap Talk cross-model match
python research.py exp4b --provider1 openai --provider2 gemini

# Group dynamics experiment (15 agents, 30 rounds)
python research.py exp5b --n_agents 15 --rounds 30

# Run Prisoner's Dilemma only
python research.py exp1 --games pd

# Run all experiments, repeat 5 times
python research.py all --repeats 5
```

## Output Structure

```
results/{timestamp}/
├── config.json               # Experiment configuration
├── summary.json              # Summary
├── raw/                      # Raw trial data (JSON)
│   └── {exp}_{game}_{condition}_trial{N}.json
├── rounds/                   # Round data (CSV)
│   └── {exp}_{game}_{condition}_rounds.csv
├── stats/                    # Statistical summaries
│   └── {exp}_summary.csv
├── figures/                  # Charts
│   └── {exp}_{game}_{condition}.png
└── anomalies/                # Anomaly records
    └── {exp}_anomalies.csv
```

## Project Structure

```
452111-main/
├── research.py              # Main experiment script
├── game_theory/
│   ├── games.py             # Game definitions and payoff matrices
│   ├── strategies.py        # Classic game strategies (TitForTat, Pavlov, etc.)
│   ├── llm_strategy.py      # LLM decision strategy
│   ├── llm_api.py           # Unified LLM API interface
│   ├── simulation.py        # Group dynamics simulation engine
│   ├── network.py           # Network topology (fully connected, small-world, etc.)
│   └── prompts/             # Prompt templates
└── README.md
```

## Supported Game Types

| Game | Command Arg | Description |
|------|----------|------|
| Prisoner's Dilemma | pd | Prisoner's Dilemma |
| Snowdrift Game | snowdrift | Snowdrift Game |
| Stag Hunt | stag_hunt | Stag Hunt |

## Supported LLMs

- **DeepSeek** - Default, best cost-performance
- **OpenAI** - GPT-4o
- **Gemini** - Gemini 2.0 Flash
- **Ollama** - Local model, free

## Version History

### v0.6.0 (v17)
- Remove Harmony Game: dominant cooperation strategy has no analytical value; focus on three game types
- Remove moonshot proxy layer: Gemini connects directly to native API (gemini-2.0-flash)
- Unify exp5 agent configuration: match exp5b with 3 LLM + 8 classic strategies = 11 agents
- Clean up unused code: remove plot_comparison_bar(), _trigger_reflection() empty stub, unused imports
- Add bilingual comments (Chinese/English): add English comments to all modules

### v0.5.2 (v16)
- Refactor exp5b agent allocation: fixed 3 LLM (1 per provider) + 8 classic strategies = 11 agents
- Add new classic strategies: GenerousTitForTat (forgiving tit-for-tat), Extort2 (zero-determinant extortion strategy)
- Update exp5b classic strategy list: TitForTat, TitForTwoTats, GenerousTitForTat, Extort2, Pavlov, GrimTrigger, AlwaysDefect, RandomStrategy
- Fix exp5b crash: self.n_agents undefined after removing n_agents parameter
- Switch OpenAI base URL to hiapi.online proxy

### v0.5.1 (v15)
- Fix Exp4 division-by-zero error: guard for empty `coop_rate_dict`
- Add providers parameter validation: Exp3/Exp4/Exp5b prevent empty list input
- Enhance strategy robustness: GrimTrigger/GradualStrategy add automatic state reset

### v0.5.0 (v14)
- Fix LLMStrategy and GameSimulation parameter format mismatch
- Fix AnomalyRecorder output directory error
- Fix pure/hybrid mode history window inconsistency
- Add division-by-zero protection

### v0.4.0 (v13)
- Refactor output directory structure: raw/, rounds/, stats/, figures/, anomalies/
- Add unified save interface

### v0.3.0 (v12)
- Unify version number management
- Fix exception handling issues
- Update documentation

### v0.2.0 (v11)
- Fix exp5/exp5b strategy name leakage: rename from strategy names to Agent_N naming
- Add strategy_map for mapping during analysis
- Update strategy list
