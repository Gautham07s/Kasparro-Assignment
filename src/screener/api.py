"""
FastAPI application for the AI Resume Screening System.

Endpoints:
  GET  /health   -> {"status": "ok"}
  POST /screen   -> Run screening pipeline
  GET  /results  -> Latest screening results
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_settings
from .models import ScreeningRun
from .pipeline import run_screening

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Screening & Ranking System",
    description="Screen and rank candidate resumes based on Python and AI/agentic evidence",
    version="1.0.0",
)

# ── In-memory state ──────────────────────────────────────────

_latest_run: Optional[ScreeningRun] = None
_screening_lock = asyncio.Lock()


# ── Request/Response Models ──────────────────────────────────


class ScreenRequest(BaseModel):
    """Request body for POST /screen."""
    input_dir: Optional[str] = None
    use_llm: bool = True
    output_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "ok"


# ── Endpoints ────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/screen")
async def screen_resumes(request: ScreenRequest):
    """
    Run the screening pipeline.

    Uses an asyncio.Lock to prevent concurrent runs (returns 409 if busy).
    """
    global _latest_run

    settings = load_settings()

    input_dir = request.input_dir or settings.default_input_dir
    output_path = request.output_path or settings.default_output_path

    # Validate input directory
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Input directory does not exist or is not a directory: {input_dir}",
        )

    # Guard against concurrent runs
    if _screening_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A screening run is already in progress. Please wait.",
        )

    async with _screening_lock:
        run = await run_screening(
            input_dir=input_path,
            settings=settings,
            use_llm=request.use_llm,
        )

        # Store latest run
        _latest_run = run

        # Write results file
        from .report import write_json
        write_json(run, output_path)

    return run.model_dump()


@app.get("/results")
async def get_results():
    """Return the latest screening results, or 404 if no run has been performed."""
    if _latest_run is None:
        raise HTTPException(
            status_code=404,
            detail="No screening results available. Run POST /screen first.",
        )
    return _latest_run.model_dump()
