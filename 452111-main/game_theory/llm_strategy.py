"""
LLM-based game strategy.
Modes: pure (LLM analyzes raw history) or hybrid (code pre-processes stats).
Includes ResponseParser for parsing LLM responses.
"""

import re
import random
import os
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field


# --- Template Loading ---

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_TEMPLATE_CACHE: Dict[str, str] = {}


def _load_template(name: str) -> str:
    """Load prompt template from prompts/ directory."""
    if name not in _TEMPLATE_CACHE:
        path = os.path.join(_TEMPLATE_DIR, f"{name}.txt")
        with open(path, "r", encoding="utf-8") as f:
            _TEMPLATE_CACHE[name] = f.read()
    return _TEMPLATE_CACHE[name]


# --- Parse Status ---

class ParseStatus(Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"

@dataclass
class ParseResult:
    """Result of parsing an LLM response."""
    action: Optional[Any] = None
    status: ParseStatus = ParseStatus.FAILED
    confidence: float = 0.0
    matched_pattern: str = ""
    raw_response: str = ""

    cooperate_signals: List[Tuple[str, float]] = field(default_factory=list)
    defect_signals: List[Tuple[str, float]] = field(default_factory=list)


# --- Response Parser ---

class ResponseParser:
    """Parse LLM responses into cooperate/defect actions."""

    # Cooperate patterns (ordered by confidence)
    COOPERATE_PATTERNS = [
        (r"ACTION:\s*COOPERATE", 1.0),
        (r"ACTION:\s*C\b", 0.95),

        (r"action:\s*cooperate", 0.9),
        (r"my\s+action\s*(?:is)?:?\s*cooperate", 0.85),
        (r"i\s+(?:will\s+)?(?:choose|select|pick)\s+(?:to\s+)?cooperate", 0.85),
        (r"decision:\s*cooperate", 0.85),
        (r"choice:\s*cooperate", 0.85),

        (r"i(?:'ll|\s+will)\s+cooperate", 0.8),
        (r"let(?:'s|s)\s+cooperate", 0.75),
        (r"cooperating\s+(?:is|seems)\s+(?:the\s+)?(?:best|better|right)", 0.7),
        (r"(?:strategy|best|optimal)\s+(?:is\s+)?to\s+cooperate", 0.7),
        (r"to\s+cooperate\s+(?:this|now|here)", 0.65),

        (r"(?:我)?(?:选择|决定)?合作", 0.85),
        (r"动作[：:]\s*合作", 0.9),
        (r"行动[：:]\s*合作", 0.9),
        (r"选择[：:]\s*合作", 0.85),

        (r"\b(?:action|choice|decision)\s*[：:=]\s*C\b", 0.7),
        (r"\bcooperate\b", 0.5),
    ]

    # Defect patterns (ordered by confidence)
    DEFECT_PATTERNS = [
        (r"ACTION:\s*DEFECT", 1.0),
        (r"ACTION:\s*D\b", 0.95),

        (r"action:\s*defect", 0.9),
        (r"my\s+action\s*(?:is)?:?\s*defect", 0.85),
        (r"i\s+(?:will\s+)?(?:choose|select|pick)\s+(?:to\s+)?defect", 0.85),
        (r"decision:\s*defect", 0.85),
        (r"choice:\s*defect", 0.85),

        (r"i(?:'ll|\s+will)\s+defect", 0.8),
        (r"i\s+(?:must|should|have\s+to)\s+defect", 0.75),
        (r"defecting\s+(?:is|seems)\s+(?:the\s+)?(?:best|better|right|optimal)", 0.7),
        (r"(?:have|need)\s+to\s+defect", 0.7),
        (r"(?:strategy|best|optimal)\s+(?:is\s+)?to\s+defect", 0.7),
        (r"to\s+defect\s+(?:this|now|here)", 0.65),

        (r"i(?:'ll|\s+will)\s+betray", 0.8),
        (r"i\s+(?:choose|select)\s+(?:to\s+)?betray", 0.8),

        (r"(?:我)?(?:选择|决定)?背叛", 0.85),
        (r"(?:我)?(?:选择|决定)?不合作", 0.8),
        (r"动作[：:]\s*背叛", 0.9),
        (r"行动[：:]\s*背叛", 0.9),
        (r"选择[：:]\s*背叛", 0.85),
        (r"动作[：:]\s*D\b", 0.85),

        (r"\b(?:action|choice|decision)\s*[：:=]\s*D\b", 0.7),
        (r"\bdefect\b(?!\s*(?:ion|ive|or))", 0.5),
    ]

    def __init__(self):
        self.stats = {
            "total": 0,
            "success": 0,
            "fallback": 0,
            "ambiguous": 0,
            "failed": 0,
            "forced_cooperate": 0,
            "forced_defect": 0,
        }
        self._Action = None

    @property
    def Action(self):
        if self._Action is None:
            try:
                from .games import Action
            except ImportError:
                from games import Action
            self._Action = Action
        return self._Action

    def parse(self, response: str) -> ParseResult:
        """Parse an LLM response into an action."""
        self.stats["total"] += 1

        if not response or not response.strip():
            self.stats["failed"] += 1
            return self._random_fallback("", "empty response")

        text = response.strip()

        coop_signals = self._scan_patterns(text, self.COOPERATE_PATTERNS)
        defect_signals = self._scan_patterns(text, self.DEFECT_PATTERNS)

        max_coop = max([conf for _, conf in coop_signals], default=0)
        max_defect = max([conf for _, conf in defect_signals], default=0)

        result = ParseResult(
            raw_response=response,
            cooperate_signals=coop_signals,
            defect_signals=defect_signals,
        )

        if max_coop == 0 and max_defect == 0:
            self.stats["failed"] += 1
            return self._random_fallback(response, "no signal")

        if max_coop > 0 and max_defect > 0:
            diff = abs(max_coop - max_defect)
            if diff < 0.2:
                self.stats["ambiguous"] += 1
                return self._random_fallback(response, "ambiguous")

        if max_coop > max_defect:
            result.action = self.Action.COOPERATE
            result.confidence = max_coop
            result.matched_pattern = coop_signals[0][0] if coop_signals else ""
            result.status = ParseStatus.SUCCESS if max_coop >= 0.7 else ParseStatus.FALLBACK
        else:
            result.action = self.Action.DEFECT
            result.confidence = max_defect
            result.matched_pattern = defect_signals[0][0] if defect_signals else ""
            result.status = ParseStatus.SUCCESS if max_defect >= 0.7 else ParseStatus.FALLBACK

        self.stats["success" if result.status == ParseStatus.SUCCESS else "fallback"] += 1
        return result

    def _scan_patterns(self, text: str, patterns: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Scan text for matching patterns, return sorted by confidence."""
        matches = []
        for pattern, confidence in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append((pattern, confidence))
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _random_fallback(self, response: str, reason: str) -> ParseResult:
        """Random choice when parsing fails (unbiased fallback)."""
        action = random.choice([self.Action.COOPERATE, self.Action.DEFECT])
        if action == self.Action.COOPERATE:
            self.stats["forced_cooperate"] += 1
        else:
            self.stats["forced_defect"] += 1

        return ParseResult(
            action=action,
            status=ParseStatus.FAILED,
            confidence=0.0,
            matched_pattern=f"random ({reason})",
            raw_response=response,
        )

    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        total = self.stats["total"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "success_rate": self.stats["success"] / total,
            "fallback_rate": self.stats["fallback"] / total,
            "ambiguous_rate": self.stats["ambiguous"] / total,
            "failure_rate": self.stats["failed"] / total,
        }


# --- LLM Strategy ---

class LLMStrategy:
    """LLM-based game strategy. Calls LLM API to decide cooperate/defect."""

    DEFAULT_MAX_TOKENS = 1000

    def __init__(self,
                 provider: str = "deepseek",
                 mode: str = "hybrid",
                 game_config = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.7,
                 persona_prompt: str = None,
                 history_window: int = None,
                 enable_cheap_talk: bool = False,
                 agent_name: str = "Player",
                 personality: str = "rational and analytical",
                 strategy_tendency: str = "balanced"):
        """Initialize LLM strategy with provider, mode, and game config."""
        self.provider = provider
        self.mode = mode
        self.game_config = game_config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.persona_prompt = persona_prompt
        self.history_window = history_window
        self.enable_cheap_talk = enable_cheap_talk
        self.agent_name = agent_name
        self.personality = personality
        self.strategy_tendency = strategy_tendency

        self.parser = ResponseParser()
        self._client = None

        self.raw_responses: List[str] = []
        self.total_payoff = 0.0
        self.current_round = 0
        self.last_message: str = ""

    @property
    def name(self) -> str:
        return f"LLM ({self.provider}/{self.mode})"

    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            try:
                from .llm_api import LLMClient
            except ImportError:
                from llm_api import LLMClient
            self._client = LLMClient(provider=self.provider)
        return self._client

    @property
    def Action(self):
        try:
            from .games import Action
        except ImportError:
            from games import Action
        return Action

    def choose_action(self,
                      history: List[Tuple[Any, Any]] = None,
                      opponent_name: str = "Opponent",
                      opponent_message: str = None) -> Any:
        """Choose cooperate or defect based on history."""
        if history is None:
            history = []
        my_history = [my_act for my_act, _ in history]
        opponent_history = [opp_act for _, opp_act in history]

        prompt = self._build_prompt(my_history, opponent_history, opponent_name, opponent_message)

        try:
            response = self.client.chat(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.raw_responses.append(response)

            result = self.parser.parse(response)
            return result.action

        except Exception as e:
            return random.choice([self.Action.COOPERATE, self.Action.DEFECT])

    def _build_prompt(self,
                      my_history: List,
                      opponent_history: List,
                      opponent_name: str,
                      opponent_message: str = None) -> str:
        """Build LLM prompt based on mode."""
        if self.game_config:
            try:
                from .games import get_payoff_description
            except ImportError:
                from games import get_payoff_description
            game_desc = get_payoff_description(self.game_config)
        else:
            game_desc = "Standard Prisoner's Dilemma"

        if self.mode == "pure":
            return self._build_pure_prompt(my_history, opponent_history, opponent_name, game_desc, opponent_message)
        else:
            return self._build_hybrid_prompt(my_history, opponent_history, opponent_name, game_desc, opponent_message)

    def _build_pure_prompt(self,
                           my_history: List,
                           opponent_history: List,
                           opponent_name: str,
                           game_desc: str,
                           opponent_message: str = None) -> str:
        """Pure mode: LLM analyzes raw history."""
        self.current_round = len(my_history) + 1
        history_str = self._format_history(my_history, opponent_history)

        # Apply history window
        window = self.history_window if self.history_window else len(my_history)
        windowed_my = my_history[-window:] if my_history else []
        windowed_opp = opponent_history[-window:] if opponent_history else []
        window_size = len(windowed_my)

        my_coop_rate = "0%" if window_size == 0 else f"{sum(1 for a in windowed_my if self._get_action_value(a) == 'cooperate') / window_size:.0%}"
        opp_coop_rate = "N/A" if window_size == 0 else f"{sum(1 for a in windowed_opp if self._get_action_value(a) == 'cooperate') / window_size:.0%}"

        template = _load_template("strategy_select")
        prompt = template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            current_round=self.current_round,
            total_payoff=f"{self.total_payoff:.1f}",
            my_coop_rate=my_coop_rate,
            opp_coop_rate=opp_coop_rate,
            opp_pattern=self._analyze_pattern(opponent_history),
            history_summary=history_str if history_str else "No history yet (first round)",
        )

        if opponent_message:
            message_section = f"\n=== OPPONENT MESSAGE ===\n{opponent_name} says: \"{opponent_message}\"\n"
            prompt = prompt.replace("=== YOUR TASK ===", message_section + "=== YOUR TASK ===")

        return prompt

    def _build_hybrid_prompt(self,
                             my_history: List,
                             opponent_history: List,
                             opponent_name: str,
                             game_desc: str,
                             opponent_message: str = None) -> str:
        """Hybrid mode: code pre-processes stats for LLM."""
        self.current_round = len(my_history) + 1
        rounds_played = len(opponent_history)

        if rounds_played == 0:
            my_coop_rate_str = "0%"
            opp_coop_rate_str = "N/A"
            history_summary = "No history yet (first round)"
        else:
            window = self.history_window if self.history_window else rounds_played
            windowed_opp = opponent_history[-window:]
            windowed_my = my_history[-window:]
            window_size = len(windowed_opp)

            opp_coop = sum(1 for a in windowed_opp if self._get_action_value(a) == "cooperate")
            opp_coop_rate = opp_coop / window_size if window_size > 0 else 0
            my_coop = sum(1 for a in windowed_my if self._get_action_value(a) == "cooperate")
            my_coop_rate = my_coop / window_size if window_size > 0 else 0

            my_coop_rate_str = f"{my_coop_rate:.0%}"
            opp_coop_rate_str = f"{opp_coop_rate:.0%} ({opp_coop}/{window_size})"

            window_info = f" (last {window} rounds)" if self.history_window else ""
            last_action = self._get_action_value(opponent_history[-1])
            history_summary = f"Rounds: {rounds_played}{window_info}, Last opponent action: {last_action}"

        template = _load_template("strategy_select")
        prompt = template.format(
            opponent_name=opponent_name,
            game_desc=game_desc,
            current_round=self.current_round,
            total_payoff=f"{self.total_payoff:.1f}",
            my_coop_rate=my_coop_rate_str,
            opp_coop_rate=opp_coop_rate_str,
            opp_pattern=self._analyze_pattern(opponent_history),
            history_summary=history_summary,
        )

        if opponent_message:
            message_section = f"\n=== OPPONENT MESSAGE ===\n{opponent_name} says: \"{opponent_message}\"\n"
            prompt = prompt.replace("=== YOUR TASK ===", message_section + "=== YOUR TASK ===")

        return prompt

    def _analyze_pattern(self, opponent_history: List) -> str:
        """Classify opponent behavior from recent history."""
        if len(opponent_history) < 3:
            return "Not enough data"

        recent = opponent_history[-10:] if len(opponent_history) >= 10 else opponent_history
        coop_count = sum(1 for a in recent if self._get_action_value(a) == "cooperate")
        coop_rate = coop_count / len(recent)

        if coop_rate >= 0.8:
            return "Highly cooperative"
        elif coop_rate >= 0.5:
            return "Mixed/Conditional cooperator"
        elif coop_rate >= 0.2:
            return "Mostly defects"
        else:
            return "Always defects"

    def _format_history(self, my_history: List, opponent_history: List) -> str:
        """Format history as readable text."""
        if not my_history:
            return ""

        # Apply history window
        window = self.history_window if self.history_window else len(my_history)

        lines = []
        for i, (my_act, opp_act) in enumerate(zip(my_history, opponent_history), 1):
            my_str = my_act.value if hasattr(my_act, 'value') else str(my_act)
            opp_str = opp_act.value if hasattr(opp_act, 'value') else str(opp_act)
            lines.append(f"Round {i}: You={my_str}, Opponent={opp_str}")

        return "\n".join(lines[-window:])

    def _get_action_value(self, action) -> str:
        """Get action string value (works with both enum and string)."""
        if hasattr(action, 'value'):
            return action.value
        return str(action).lower()

    def _get_payoffs(self) -> Dict[str, int]:
        """Extract payoff values from game config."""
        if not self.game_config:
            return {"cc": 3, "cd": 0, "dc": 5, "dd": 1}
        matrix = self.game_config.payoff_matrix
        return {
            "cc": matrix[(self.Action.COOPERATE, self.Action.COOPERATE)][0],
            "cd": matrix[(self.Action.COOPERATE, self.Action.DEFECT)][0],
            "dc": matrix[(self.Action.DEFECT, self.Action.COOPERATE)][0],
            "dd": matrix[(self.Action.DEFECT, self.Action.DEFECT)][0],
        }

    def generate_message(self,
                         history: List[Tuple[Any, Any]] = None,
                         opponent_name: str = "Opponent") -> str:
        """Generate a cheap talk message to send to opponent."""
        if not self.enable_cheap_talk:
            return ""

        if history is None:
            history = []
        rounds_played = len(history)

        prompt = f"""You are playing an iterated game against {opponent_name}.

Rounds played so far: {rounds_played}

You can send a short message to your opponent before making your decision.
This message could be used to signal your intentions, build trust, or strategize.

Generate a brief message (1-2 sentences max) to send to your opponent.
The message should be strategic and relevant to the game.

MESSAGE:"""

        try:
            response = self.client.chat(
                prompt,
                max_tokens=100,
                temperature=self.temperature,
            )

            message = response.strip()
            if "MESSAGE:" in message:
                message = message.split("MESSAGE:")[-1].strip()

            self.last_message = message
            return message

        except Exception as e:
            return ""

    def update_payoff(self, payoff: float):
        """Add payoff to running total."""
        self.total_payoff += payoff

    def get_debug_info(self) -> Dict:
        """Get debug info for diagnostics."""
        return {
            "provider": self.provider,
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "total_calls": len(self.raw_responses),
            "parse_quality": self.parser.get_stats(),
            "last_raw_response": self.raw_responses[-1] if self.raw_responses else None,
        }

    def reset(self):
        """Reset state for a new game."""
        self.raw_responses = []
        self.total_payoff = 0.0
        self.last_message = ""
        self.parser = ResponseParser()


# --- Test Code ---

if __name__ == "__main__":
    print("=" * 60)
    print("解析器测试")
    print("=" * 60)

    parser = ResponseParser()

    test_cases = [
        ("ACTION: COOPERATE", "COOPERATE", ParseStatus.SUCCESS),
        ("ACTION: DEFECT", "DEFECT", ParseStatus.SUCCESS),
        ("action: cooperate", "COOPERATE", ParseStatus.SUCCESS),
        ("action: defect", "DEFECT", ParseStatus.SUCCESS),
        ("I will cooperate.", "COOPERATE", ParseStatus.SUCCESS),
        ("I choose to defect.", "DEFECT", ParseStatus.SUCCESS),
        ("I'll defect this round.", "DEFECT", ParseStatus.SUCCESS),
        ("Let's cooperate for mutual benefit.", "COOPERATE", ParseStatus.SUCCESS),
        ("我选择合作", "COOPERATE", ParseStatus.SUCCESS),
        ("我选择背叛", "DEFECT", ParseStatus.SUCCESS),
        ("After careful analysis, I think the best strategy is to defect this round.",
         "DEFECT", ParseStatus.SUCCESS),
        ("The optimal strategy is to cooperate here.",
         "COOPERATE", ParseStatus.SUCCESS),
        ("Hello world", None, ParseStatus.FAILED),
        ("The weather is nice today.", None, ParseStatus.FAILED),
    ]

    passed = 0
    for text, expected_action, expected_status in test_cases:
        result = parser.parse(text)

        if expected_action is None:
            success = result.status == expected_status
        else:
            success = (result.action.value.upper() == expected_action and
                      result.status in [ParseStatus.SUCCESS, ParseStatus.FALLBACK])

        icon = "OK" if success else "FAIL"
        action_str = result.action.value.upper() if result.action else "None"
        status_str = "[success]" if success else "[failed]"

        print(f"{icon} '{text[:40]:40s}' -> {action_str:10s} {status_str}")

        if success:
            passed += 1

    print(f"通过: {passed}/{len(test_cases)}")
    print(f"统计: {parser.get_stats()}")
