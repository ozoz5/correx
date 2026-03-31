"""LLM-based reaction scorer.

Backends (tried in order when backend="auto"):
  1. ollama  — local HTTP service, no extra deps
  2. mlx     — mlx-lm in-process inference (lazy-loaded, M-series Mac)
  3. rule    — rule-based fallback, always available

Usage:
    scorer = LlmScorer()                          # auto-detect
    scorer = LlmScorer(backend="ollama")          # force ollama
    scorer = LlmScorer(backend="mlx", model="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
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

Backend = Literal["auto", "ollama", "mlx", "rule"]

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
ユーザーがAIの出力に対して与えたフィードバックを評価してください。
満足度を 0.0 〜 1.0 の数値で答えてください。

目安:
0.05 = 完全否定・全部やり直し（「違う」「最悪」「ひどい」）
0.20 = 大きな修正が必要
0.35 = 修正が必要
0.55 = 部分的に良い、一部修正あり
0.65 = 渋い承認（「まあいいか」「一応OK」「とりあえずいい」など、修正はないが消極的）
0.70 = ほぼ良い、軽い修正あり
0.75 = 沈黙・承認（何も言わない = 受け入れた）
0.85 = 良い・満足（「いいね」「よかった」）
0.95 = 非常に良い・絶賛（「完璧」「最高」「これだ」）

「まあ」「一応」「とりあえず」「まあまあ」「悪くない」「まあいいか」「まあいい」が含まれ修正がない場合は 0.65 にしてください。

数値だけ返してください。説明不要です。

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
        print(f"[pseudo-intelligence] Ollama scoring failed: {exc}", file=sys.stderr)
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
        print("[pseudo-intelligence] MLX scoring skipped: mlx_lm not installed", file=sys.stderr)
        return None

    if model not in _mlx_cache:
        try:
            _mlx_cache[model] = load(model)
        except Exception as exc:
            import sys
            print(f"[pseudo-intelligence] MLX model load failed: {exc}", file=sys.stderr)
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
        print(f"[pseudo-intelligence] MLX scoring failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# LlmScorer
# ---------------------------------------------------------------------------

class LlmScorer:
    """
    Score user reactions via LLM with automatic fallback to rule-based scoring.

    Args:
        backend:  "auto" | "ollama" | "mlx" | "rule"
        model:    Model name/path for ollama or mlx.
                  Defaults: ollama → "qwen2.5:1.5b"
                            mlx    → "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        endpoint: Ollama API base URL (default: http://127.0.0.1:11434)
    """

    def __init__(
        self,
        backend: Backend = "auto",
        model: str | None = None,
        endpoint: str = "http://127.0.0.1:11434",
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

    def _resolve_backend(self) -> Backend:
        if self._resolved is not None:
            return self._resolved
        if self.backend != "auto":
            self._resolved = self.backend
            import sys
            print(f"[pseudo-intelligence] Scorer backend: {self._resolved} (explicit)", file=sys.stderr)
            return self._resolved

        import sys

        # Try ollama first
        if _ollama_available(self.endpoint):
            self._resolved = "ollama"
            print(f"[pseudo-intelligence] Scorer backend: ollama ({self.endpoint})", file=sys.stderr)
            return self._resolved

        # Try mlx-lm
        try:
            import importlib.util
            if importlib.util.find_spec("mlx_lm") is not None:
                self._resolved = "mlx"
                print(f"[pseudo-intelligence] Scorer backend: mlx ({self.model})", file=sys.stderr)
                return self._resolved
        except Exception:
            pass

        self._resolved = "rule"
        print("[pseudo-intelligence] Scorer backend: rule (fallback — no LLM available)", file=sys.stderr)
        return self._resolved

    def _cached_score(self, feedback: str, corrections_key: str) -> float:
        cache_key = (feedback, corrections_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        corrections = [c for c in corrections_key.split("\x00") if c]

        # Intercept patterns that small LLMs consistently misread
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
                return result

        if resolved == "mlx":
            result = _mlx_score(feedback, corrections, model=self.model)
            if result is not None:
                self._cache[cache_key] = result
                return result

        # Fallback: rule-based
        result = score_reaction(feedback, corrections)
        self._cache[cache_key] = result
        return result

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
