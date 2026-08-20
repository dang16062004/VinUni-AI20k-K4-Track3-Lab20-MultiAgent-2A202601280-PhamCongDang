"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# USD per 1,000,000 tokens: (input_price, output_price). Update if OpenAI pricing changes.
_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Best-effort cost estimate. Returns None for unknown models rather than guessing."""

    if input_tokens is None or output_tokens is None:
        return None
    pricing = _PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


class LLMClient:
    """OpenAI-backed LLM client.

    Retries transient errors (connection issues, rate limits, 5xx) with exponential
    backoff. Timeout is bounded by `settings.timeout_seconds`. Does NOT retry on
    authentication/validation errors (e.g. bad API key) since those never succeed on retry.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env before calling LLMClient.complete()."
            )
        self._model = settings.openai_model
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.timeout_seconds)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, InternalServerError)),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Token usage and estimated cost are logged on the returned `LLMResponse` so callers
        (agents, benchmark) don't need to touch the OpenAI SDK directly.
        """

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost_usd(self._model, input_tokens, output_tokens),
        )
