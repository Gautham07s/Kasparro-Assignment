"""
OpenAI LLM client (alternate provider).

Uses the OpenAI API with structured JSON output and Pydantic validation.
Includes repair retry on validation failure and exponential backoff on transient errors.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings
from .base import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OpenAIClient:
    """OpenAI API client implementing the LLMClient protocol."""

    def __init__(self, settings: Settings) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        self._model = settings.llm_model or "gpt-4o"

    async def extract(self, system: str, user: str, schema: type[T]) -> T:
        """Send prompt to OpenAI and parse structured response."""
        json_schema = schema.model_json_schema()

        try:
            response_text = await self._call_api(system, user, json_schema)
            return self._parse_response(response_text, schema, system, user, json_schema)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI extraction failed: {exc}") from exc

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_api(self, system: str, user: str, json_schema: dict) -> str:
        """Make the API call with retry logic."""
        import openai

        schema_instruction = (
            f"\n\nRespond with a JSON object matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system + schema_instruction},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except openai.RateLimitError as exc:
            raise LLMError(f"OpenAI rate limited: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ConnectionError(f"OpenAI server error: {exc}") from exc
            raise LLMError(f"OpenAI API error: {exc}") from exc
        except openai.APITimeoutError as exc:
            raise TimeoutError(f"OpenAI timeout: {exc}") from exc

    def _parse_response(
        self, text: str, schema: type[T],
        system: str, user: str, json_schema: dict,
    ) -> T:
        """Parse and validate the LLM response. One repair retry on validation failure."""
        cleaned = self._strip_code_fences(text)

        try:
            data = json.loads(cleaned)
            return schema.model_validate(data)
        except Exception as first_error:
            logger.warning("First parse attempt failed: %s", first_error)

            try:
                import asyncio
                repair_text = asyncio.get_event_loop().run_until_complete(
                    self._repair_call(system, user, text, str(first_error), json_schema)
                )
                cleaned = self._strip_code_fences(repair_text)
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except Exception as repair_error:
                raise LLMError(
                    f"OpenAI response validation failed after repair: {repair_error}"
                ) from repair_error

    async def _repair_call(
        self, system: str, user: str, original: str, error: str, json_schema: dict,
    ) -> str:
        """Send the validation error back for one repair attempt."""
        repair_prompt = (
            f"Your previous response had a validation error:\n{error}\n\n"
            f"Original response:\n{original}\n\n"
            f"Please fix the JSON to match the schema and return only valid JSON."
        )
        return await self._call_api(system, repair_prompt, json_schema)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip markdown code fences if present."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
