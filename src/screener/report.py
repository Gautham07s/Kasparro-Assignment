"""
Report generation: JSON, CSV, and terminal output.

Writes results atomically (temp file + rename).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path

from .models import ScreeningRun

logger = logging.getLogger(__name__)


def write_json(run: ScreeningRun, output_path: str | Path) -> Path:
    """
    Write results.json atomically (temp file then rename).

    Creates the output directory if it doesn't exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the output dict
    data = {
        "batch_summary": run.batch_summary.model_dump(),
        "ranked_candidates": [
            _candidate_to_dict(c) for c in run.ranked_candidates
        ],
        "rejected_candidates": [
            {
                "candidate": c.candidate_name,
                "eligible": c.eligible,
                "rejection_reasons": c.rejection_reasons,
                "matched_skills": c.matched_skills,
            }
            for c in run.rejected_candidates
        ],
        "failed_files": [
            {"file": f.file_name, "reason": f.reason}
            for f in run.failed_files
        ],
        "duplicates": [
            {"file": d.file_name, "duplicate_of": d.duplicate_of}
            for d in run.duplicates
        ],
    }

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent),
        suffix=".json.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # On Windows, we need to remove the target first if it exists
        if output_path.exists():
            output_path.unlink()
        os.rename(tmp_path, str(output_path))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Results written to %s", output_path)
    return output_path


def write_csv(run: ScreeningRun, output_path: str | Path) -> Path:
    """Write a flat CSV with ranked and rejected candidates."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_path.with_suffix(".csv")

    headers = [
        "rank", "candidate_name", "email", "eligible", "total_score",
        "ai_project_depth", "python_backend", "cloud_fullstack",
        "github", "engineering_depth", "rejection_reasons",
    ]

    rows = []
    for c in run.ranked_candidates:
        bd = c.score_breakdown
        rows.append({
            "rank": c.rank,
            "candidate_name": c.candidate_name,
            "email": c.email or "",
            "eligible": "yes",
            "total_score": c.total_score,
            "ai_project_depth": bd.ai_project_depth if bd else 0,
            "python_backend": bd.python_backend if bd else 0,
            "cloud_fullstack": bd.cloud_fullstack if bd else 0,
            "github": bd.github if bd else 0,
            "engineering_depth": bd.engineering_depth if bd else 0,
            "rejection_reasons": "",
        })

    for c in run.rejected_candidates:
        rows.append({
            "rank": "",
            "candidate_name": c.candidate_name,
            "email": c.email or "",
            "eligible": "no",
            "total_score": 0,
            "ai_project_depth": 0,
            "python_backend": 0,
            "cloud_fullstack": 0,
            "github": 0,
            "engineering_depth": 0,
            "rejection_reasons": "; ".join(c.rejection_reasons),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV written to %s", csv_path)
    return csv_path


def print_terminal_summary(run: ScreeningRun) -> None:
    """Print a summary table and batch stats to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        _print_rich_summary(run)
    except ImportError:
        _print_plain_summary(run)


def _print_rich_summary(run: ScreeningRun) -> None:
    """Print a rich terminal table of top 10 + batch summary."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Batch summary panel
    bs = run.batch_summary
    summary_text = (
        f"[bold]Total Resumes:[/bold] {bs.total_resumes}  |  "
        f"[green]Parsed:[/green] {bs.successfully_parsed}  |  "
        f"[green]Eligible:[/green] {bs.eligible}  |  "
        f"[red]Rejected:[/red] {bs.rejected}  |  "
        f"[yellow]Failed:[/yellow] {bs.failed_unreadable}  |  "
        f"[dim]Duplicates:[/dim] {bs.duplicates}  |  "
        f"[yellow]LLM Fallbacks:[/yellow] {bs.llm_fallbacks}  |  "
        f"[yellow]GitHub Failures:[/yellow] {bs.github_failures}"
    )
    console.print(Panel(summary_text, title="Batch Summary", border_style="blue"))

    # Top 10 table
    table = Table(title="Top 10 Ranked Candidates", show_lines=True)
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("Candidate", style="bold", width=22)
    table.add_column("Score", style="bold green", width=6)
    table.add_column("AI", width=4)
    table.add_column("Py/BE", width=6)
    table.add_column("Cloud", width=6)
    table.add_column("GH", width=4)
    table.add_column("Eng", width=4)
    table.add_column("Source", width=8)

    for c in run.ranked_candidates[:10]:
        bd = c.score_breakdown
        table.add_row(
            str(c.rank),
            c.candidate_name[:20],
            str(c.total_score),
            str(bd.ai_project_depth) if bd else "-",
            str(bd.python_backend) if bd else "-",
            str(bd.cloud_fullstack) if bd else "-",
            str(bd.github) if bd else "-",
            str(bd.engineering_depth) if bd else "-",
            c.analysis_source or "-",
        )

    console.print(table)

    # Failed files
    if run.failed_files:
        console.print(f"\n[yellow]Failed files ({len(run.failed_files)}):[/yellow]")
        for f in run.failed_files:
            console.print(f"  • {f.file_name}: {f.reason}")


def _print_plain_summary(run: ScreeningRun) -> None:
    """Fallback plain-text summary."""
    bs = run.batch_summary
    print(f"\n{'=' * 60}")
    print("BATCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {bs.total_resumes} | Parsed: {bs.successfully_parsed} | "
          f"Eligible: {bs.eligible} | Rejected: {bs.rejected}")
    print(f"Failed: {bs.failed_unreadable} | Duplicates: {bs.duplicates} | "
          f"LLM Fallbacks: {bs.llm_fallbacks} | GitHub Failures: {bs.github_failures}")
    print(f"\n{'=' * 60}")
    print("TOP 10 RANKED CANDIDATES")
    print(f"{'=' * 60}")
    print(f"{'Rank':<5} {'Candidate':<25} {'Score':<7} {'AI':<5} {'Py':<5} "
          f"{'Cloud':<6} {'GH':<4} {'Eng':<4}")
    print("-" * 60)

    for c in run.ranked_candidates[:10]:
        bd = c.score_breakdown
        print(f"{c.rank:<5} {c.candidate_name[:23]:<25} {c.total_score:<7} "
              f"{bd.ai_project_depth if bd else 0:<5} "
              f"{bd.python_backend if bd else 0:<5} "
              f"{bd.cloud_fullstack if bd else 0:<6} "
              f"{bd.github if bd else 0:<4} "
              f"{bd.engineering_depth if bd else 0:<4}")


def _candidate_to_dict(c) -> dict:
    """Convert a CandidateResult to the spec's output dict format."""
    result = {
        "rank": c.rank,
        "candidate_name": c.candidate_name,
        "email": c.email,
        "source_file": c.source_file,
        "eligible": c.eligible,
        "total_score": c.total_score,
    }
    if c.score_breakdown:
        result["score_breakdown"] = c.score_breakdown.model_dump()
    result.update({
        "matched_skills": c.matched_skills,
        "project_summary": c.project_summary,
        "github_summary": c.github_summary,
        "strengths": c.strengths,
        "concerns": c.concerns,
        "evidence": c.evidence,
        "penalties_applied": c.penalties_applied,
        "analysis_source": c.analysis_source,
    })
    return result
