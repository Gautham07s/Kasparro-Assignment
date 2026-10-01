"""
Gemini client adapter using google-genai.
"""

from __future__ import annotations

import json
import logging
import time

from google import genai
from google.genai import types
from pydantic import ValidationError

from ..config import Settings
from .base import LLMClient, LLMError

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Adapter for Google Gemini using structured output."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.llm_model or "gemini-2.0-flash"
        self.timeout = settings.llm_timeout_seconds

    async def extract(self, system: str, user: str, schema: type) -> object:
        """
        Uses Gemini's structured output (response_schema).
        We attempt to parse the JSON and retry on validation errors.
        """
        max_attempts = 2
        messages = [
            types.Content(role="user", parts=[types.Part.from_text(text=f"{system}\n\n{user}")])
        ]
        
        # Pydantic schema to JSON schema conversion
        json_schema = schema.model_json_schema()
        
        def remove_additional_properties(d):
            if isinstance(d, dict):
                d.pop("additionalProperties", None)
                for v in d.values():
                    remove_additional_properties(v)
            elif isinstance(d, list):
                for item in d:
                    remove_additional_properties(item)
                    
        remove_additional_properties(json_schema)
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=0.0,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                # google-genai doesn't have an async client in standard docs yet, so we use sync 
                # inside an executor or just run it synchronously (which is okay with max_concurrency 
                # but better if we had true async). Since this is a quick adapter, we run it directly.
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=messages,
                        config=config,
                    )
                )

                if not response.text:
                    raise LLMError("Empty response from Gemini")
                
                # Try parsing as JSON
                try:
                    data = json.loads(response.text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON string from LLM: {exc}")

                # Validate against Pydantic schema
                parsed_obj = schema.model_validate(data)
                return parsed_obj

            except ValidationError as exc:
                logger.warning("Gemini validation error (attempt %d/%d): %s", attempt, max_attempts, exc)
                if attempt == max_attempts:
                    raise LLMError(f"Failed to validate Gemini response after {max_attempts} attempts") from exc
                
                # Append repair message
                messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
                repair_prompt = f"Your JSON response failed schema validation:\n{exc}\nPlease fix it and return only valid JSON."
                messages.append(types.Content(role="user", parts=[types.Part.from_text(text=repair_prompt)]))
                # Add delay
                await asyncio.sleep(2 * attempt)
                
            except Exception as exc:
                logger.warning("Gemini API error (attempt %d/%d): %s", attempt, max_attempts, exc)
                if attempt == max_attempts:
                    raise LLMError(f"Gemini API failed after {max_attempts} attempts") from exc
                # Wait and retry for rate limits/server errors
                import asyncio
                await asyncio.sleep(4 * attempt)

        raise LLMError("Exhausted Gemini attempts")
