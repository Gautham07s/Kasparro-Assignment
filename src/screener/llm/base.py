"""
LLM client protocol and shared errors.

Defines the interface that all LLM providers must implement.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""
    pass


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM provider clients."""

    async def extract(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        """
        Send a prompt to the LLM and parse the structured response.

        Args:
            system: System prompt
            user: User prompt (resume content)
            schema: Pydantic model class to parse the response into

        Returns:
            An instance of the schema type

        Raises:
            LLMError: If the LLM call fails after all retries
        """
        ...
