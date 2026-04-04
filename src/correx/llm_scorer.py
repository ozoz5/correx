"""LLM-based reaction scorer.

Backends (tried in order when backend="auto"):
  1. ollama     — local HTTP service, no extra deps
  2. anthropic  — Anthropic API (Haiku), most accurate, ~$0.00006/turn
  3. mlx        — mlx-lm in-process inference (lazy-loaded, M-series Mac)
  4. rule       — rule-based fallback, always available

LLM scores are persisted to score_dictionary.json. Once a feedback pattern
is scored by the LLM, it's never scored again — the dictionary result is
returned instantly. Over time API calls approach zero.

Usage:
    scorer = LlmScorer()                          # auto-detect
    scorer = LlmScorer(backend="anthropic")       # force Anthropic API
    scorer = LlmScorer(backend="ollama")          # force ollama
    scorer = LlmScorer(backend="rule")            # rule-based only

    score = scorer.score("ここだけ直して、あとは最高")  # → 0.65
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Literal

from .reaction_scorer import score_reaction

Backend = Literal["auto", "ollama", "anthropic", "mlx", "rule"]

# Patterns that small models misread — intercept before LLM call
_RELUCTANT_ACCEPTANCE = (
    "まあいいか",
    "まあいい",
    "まあ良いか",
    "まあ良い",
    "一応ok",
    "一応OK",
    "一応いい",
    "とりあえずいい",
    "とりあえずok",
    "とりあえずOK",
)

_RELUCTANT_SCORE = 0.65


def _preprocess_score(feedback: str, corrections: list[str]) -> float | None:
    """
    Intercept known patterns that small LLMs consistently misread.
    Returns a score if matched, None if the LLM should decide.
    """
    if corrections:
        return None  # let LLM handle mixed signals
    lower = feedback.strip().lower()
    if any(pat.lower() in lower for pat in _RELUCTANT_ACCEPTANCE):
        return _RELUCTANT_SCORE
    return None

_SCORING_PROMPT = """\
ユーザーがAIとの対話中に与えた反応を評価してください。
これはAIの出力に対する「満足度」ではなく、「対話がうまく進んでいるか」の指標です。

重要な区別:
- 「世界のどこにもないだろ」→ 感嘆・絶賛（0.95）。否定ではない。
- 「いやまて、報酬だよ！」→ 新しい気づきの共有（0.70）。拒絶ではない。
- 「そうだね」→ 同意・承認（0.80）。
- 「進め！」→ 強い承認と指示（0.85）。
- 「うーん、それは内部的な枠から出てない」→ 方向修正（0.50）。否定ではなく改善要求。
- 「バグを直せよ！」→ 不満・修正要求（0.20）。
- 「まあまあじゃん」→ 渋い承認（0.65）。

修正点（corrections）がある場合:
- corrections + 称賛的なフィードバック → 学びの瞬間。高スコア（0.70-0.90）
- corrections + 否定的なフィードバック → 実際の失敗。低スコア（0.10-0.30）
- corrections + 中立的なフィードバック → 方向転換。中スコア（0.50-0.65）
- corrections + フィードバックなし → 学習記録。中スコア（0.60）

数値だけ返してください。

フィードバック: {feedback}
修正点: {corrections}"""


def _parse_float(text: str) -> float | None:
    """Extract the first float in [0.0, 1.0] from a string."""
    for token in re.findall(r"\d+\.?\d*", text.strip()):
        try:
            value = float(token)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _ollama_score(
    feedback: str,
    corrections: list[str],
    *,
    model: str = "qwen2.5:1.5b",
    endpoint: str = "http://127.0.0.1:11434",
    timeout: float = 8.0,
) -> float | None:
    prompt = _SCORING_PROMPT.format(
        feedback=feedback or "（なし）",
        corrections="、".join(corrections) if corrections else "（なし）",
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return _parse_float(data.get("response", ""))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        import sys
        print(f"[correx] Ollama scoring failed: {exc}", file=sys.stderr)
        return None


def _ollama_available(endpoint: str = "http://127.0.0.1:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=2.0):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# MLX-LM backend (lazy in-process, cached per model_id)
# ---------------------------------------------------------------------------

_mlx_cache: dict[str, tuple] = {}


def _mlx_score(
    feedback: str,
    corrections: list[str],
    *,
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_tokens: int = 8,
) -> float | None:
    try:
        from mlx_lm import generate, load  # type: ignore[import]
    except ImportError:
        import sys
        print("[correx] MLX scoring skipped: mlx_lm not installed", file=sys.stderr)
        return None

    if model not in _mlx_cache:
        try:
            _mlx_cache[model] = load(model)
        except Exception as exc:
            import sys
            print(f"[correx] MLX model load failed: {exc}", file=sys.stderr)
            return None

    mlx_model, tokenizer = _mlx_cache[model]
    prompt = _SCORING_PROMPT.format(
        feedback=feedback or "（なし）",
        corrections="、".join(corrections) if corrections else "（なし）",
    )
    try:
        # Build chat-style prompt if tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(
            mlx_model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )
        return _parse_float(response)
    except Exception as exc:
        import sys
        print(f"[correx] MLX scoring failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Anthropic API backend
# ---------------------------------------------------------------------------

def _anthropic_score(
    feedback: str,
    corrections: list[str],
    *,
    model: str = "claude-sonnet-4-20250514",
    timeout: float = 10.0,
) -> float | None:
    """Score using Anthropic API. Most accurate, lowest cost ($0.00006/turn)."""
    try:
        import anthropic  # type: ignore[import]
    except ImportError:
        return None

    prompt = _SCORING_PROMPT.format(
        feedback=feedback or "（なし）",
        corrections="、".join(corrections) if corrections else "（なし）",
    )
    try:
        client = anthropic.Anthropic(timeout=timeout)
        response = client.messages.create(
            model=model,
            max_tokens=8,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return _parse_float(text)
    except Exception as exc:
        import sys
        print(f"[correx] Anthropic scoring failed: {exc}", file=sys.stderr)
        return None


def _anthropic_available() -> bool:
    """Check if Anthropic SDK is installed and API key is configured."""
    try:
        import anthropic  # type: ignore[import]
        client = anthropic.Anthropic()
        return bool(client.api_key)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LlmScorer
# ---------------------------------------------------------------------------

class LlmScorer:
    """
    Score user reactions via LLM with automatic fallback to rule-based scoring.

    LLM scores are persisted to score_dictionary.json so the same pattern
    is never scored twice by the LLM. Over time the dictionary grows and
    API calls approach zero.

    Args:
        backend:  "auto" | "ollama" | "anthropic" | "mlx" | "rule"
        model:    Model name/path for ollama or mlx.
                  Defaults: ollama → "qwen2.5:1.5b"
                            mlx    → "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        endpoint: Ollama API base URL (default: http://127.0.0.1:11434)
        score_dict_path: Path to persistent score dictionary JSON.
    """

    def __init__(
        self,
        backend: Backend = "auto",
        model: str | None = None,
        endpoint: str = "http://127.0.0.1:11434",
        score_dict_path: str | None = None,
    ):
        self.backend: Backend = backend
        self.endpoint = endpoint

        # Default models per backend
        if backend == "mlx":
            self.model = model or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        else:
            self.model = model or "qwen2.5:1.5b"

        # Resolved backend after auto-detection
        self._resolved: Backend | None = None

        # Instance-level score cache (avoids lru_cache memory leak from holding self)
        self._cache: dict[tuple[str, str], float] = {}

        # Persistent score dictionary — LLM results are stored here
        self._score_dict_path = score_dict_path
        self._score_dict: dict[str, float] = {}
        self._load_score_dict()

    def _load_score_dict(self) -> None:
        """Load persistent score dictionary from disk."""
        if not self._score_dict_path:
            return
        from pathlib import Path
        path = Path(self._score_dict_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._score_dict = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, OSError):
                pass

    def _persist_score(self, feedback: str, corrections_key: str, score: float, backend: str) -> None:
        """Save an LLM-scored result to the persistent dictionary.

        Writes are batched: only flushes to disk every 10 new entries
        to avoid excessive I/O. Call flush_score_dict() to force.
        """
        if not self._score_dict_path:
            return
        dict_key = f"{feedback}||{corrections_key}"
        self._score_dict[dict_key] = round(score, 4)
        self._dirty_count = getattr(self, "_dirty_count", 0) + 1
        if self._dirty_count >= 10:
            self.flush_score_dict()

    def teach(self, feedback: str, corrections: list[str], score: float) -> None:
        """Feed an externally-determined score into the dictionary cache.

        Called when reaction_score_override is used, so the dictionary
        continues learning from client LLM judgments. Without this, the
        dictionary would starve when overrides are always provided.
        """
        corrections_key = "|".join(sorted(set(corrections)))[:200]
        self._persist_score(feedback.strip().lower(), corrections_key, score, "client_override")

    def flush_score_dict(self) -> None:
        """Flush buffered score dictionary to disk."""
        if not self._score_dict_path or not self._score_dict:
            return
        from pathlib import Path
        path = Path(self._score_dict_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._score_dict, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            self._dirty_count = 0
        except OSError:
            pass

    def _dict_lookup(self, feedback: str, corrections_key: str) -> float | None:
        """Check if this feedback pattern was already scored by LLM."""
        dict_key = f"{feedback}||{corrections_key}"
        return self._score_dict.get(dict_key)

    def _resolve_backend(self) -> Backend:
        if self._resolved is not None:
            return self._resolved
        if self.backend != "auto":
            self._resolved = self.backend
            import sys
            print(f"[correx] Scorer backend: {self._resolved} (explicit)", file=sys.stderr)
            return self._resolved

        import sys

        # Try Anthropic API first (remote, most accurate, very cheap)
        if _anthropic_available():
            self._resolved = "anthropic"
            print("[correx] Scorer backend: anthropic (API)", file=sys.stderr)
            return self._resolved

        # Rule-based is more accurate than small local models (1.5B)
        # for Japanese mixed-signal scoring. Use it as default.
        # Ollama/MLX are available for explicit backend selection.
        self._resolved = "rule"
        print("[correx] Scorer backend: rule (improved rule-based)", file=sys.stderr)
        return self._resolved

    def _cached_score(self, feedback: str, corrections_key: str) -> float:
        cache_key = (feedback, corrections_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        corrections = [c for c in corrections_key.split("\x00") if c]

        # 1. Check persistent score dictionary (free, instant)
        dict_result = self._dict_lookup(feedback, corrections_key)
        if dict_result is not None:
            self._cache[cache_key] = dict_result
            return dict_result

        # 2. Intercept patterns that small LLMs consistently misread
        pre = _preprocess_score(feedback, corrections)
        if pre is not None:
            self._cache[cache_key] = pre
            return pre

        resolved = self._resolve_backend()

        if resolved == "ollama":
            result = _ollama_score(
                feedback, corrections, model=self.model, endpoint=self.endpoint
            )
            if result is not None:
                self._cache[cache_key] = result
                self._persist_score(feedback, corrections_key, result, "ollama")
                return result

        if resolved == "anthropic":
            result = _anthropic_score(feedback, corrections)
            if result is not None:
                self._cache[cache_key] = result
                self._persist_score(feedback, corrections_key, result, "anthropic")
                return result

        if resolved == "mlx":
            result = _mlx_score(feedback, corrections, model=self.model)
            if result is not None:
                self._cache[cache_key] = result
                self._persist_score(feedback, corrections_key, result, "mlx")
                return result

        # Fallback: rule-based (not persisted — rule-based is deterministic)
        result = score_reaction(feedback, corrections)
        self._cache[cache_key] = result
        return result

    def __del__(self) -> None:
        """Flush any buffered scores on garbage collection."""
        try:
            self.flush_score_dict()
        except Exception:
            pass

    def score(
        self,
        user_feedback: str,
        extracted_corrections: list[str] | None = None,
    ) -> float:
        """
        Score a user reaction. Returns 0.0 (worst) to 1.0 (best).

        Silence (empty feedback + no corrections) → 0.75 (silent approval).
        Uses LLM if available, falls back to rule-based automatically.
        """
        feedback = (user_feedback or "").strip()
        corrections = [c for c in (extracted_corrections or []) if c]
        corrections_key = "\x00".join(corrections)
        return self._cached_score(feedback, corrections_key)

    @property
    def active_backend(self) -> Backend:
        """Which backend is currently active."""
        return self._resolve_backend()


# ---------------------------------------------------------------------------
# Module-level convenience (uses auto-detect, shared instance)
# ---------------------------------------------------------------------------

_default_scorer: LlmScorer | None = None


def get_default_scorer() -> LlmScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = LlmScorer(backend="auto")
    return _default_scorer


def score_with_llm(
    user_feedback: str,
    extracted_corrections: list[str] | None = None,
    *,
    scorer: LlmScorer | None = None,
) -> float:
    """Score a user reaction using the best available backend."""
    return (scorer or get_default_scorer()).score(user_feedback, extracted_corrections)
