# AI Resume Screening & Ranking System

An automated resume screening system that ingests a folder of resumes, applies rule-based eligibility filtering for Python and AI/agentic evidence, scores eligible candidates on a 100-point rubric, enriches with GitHub activity, and outputs a ranked shortlist.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLI / FastAPI                              │
│            main.py (argparse)  │  api.py (POST/GET)               │
└────────────────────────┬───────┴──────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                      pipeline.py                                   │
│               Orchestration + Bounded Concurrency                  │
└───┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Ingest  ││Eligib- ││LLM /   ││GitHub  ││Scoring │
│        ││ility   ││Fallback││Enrich  ││        │
│PDF/DOCX││Python  ││Facts + ││Events +││100-pt  │
│/TXT    ││+ AI    ││Evidence││Repos   ││Rubric  │
│read    ││rules   ││extract ││2 calls ││        │
│dedupe  ││lexicons││        ││/user   ││Rank    │
└────────┘└────────┘└────────┘└────────┘└────────┘
    │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┘
                         │
                         ▼
               ┌──────────────────┐
               │    report.py     │
               │ JSON/CSV/Table   │
               └──────────────────┘
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in API keys (optional for --no-llm mode)
```

## Usage

### CLI

```bash
# Run with deterministic fallback (no LLM needed)
python main.py --no-llm --input ./resumes --output ./output/results.json

# Run with LLM analysis (requires API key in .env)
python main.py --input ./resumes --output ./output/results.json

# With CSV output and verbose logging
python main.py --no-llm --input ./resumes --csv --verbose

# Custom concurrency
python main.py --no-llm --max-concurrency 10
```

### FastAPI

```bash
# Start the server
uvicorn screener.api:app --app-dir src --reload

# Health check
curl http://localhost:8000/health

# Run screening
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{"input_dir": "./resumes", "use_llm": false}'

# Get latest results
curl http://localhost:8000/results
```

## Output Format

Results are written to `output/results.json`:

```json
{
  "batch_summary": {
    "total_resumes": 50,
    "successfully_parsed": 50,
    "eligible": 33,
    "rejected": 17,
    "failed_unreadable": 0,
    "duplicates": 0,
    "llm_fallbacks": 0,
    "github_failures": 3
  },
  "ranked_candidates": [
    {
      "rank": 1,
      "candidate_name": "...",
      "eligible": true,
      "total_score": 81,
      "score_breakdown": {
        "ai_project_depth": 30,
        "python_backend": 27,
        "cloud_fullstack": 12,
        "github": 7,
        "engineering_depth": 5
      },
      "matched_skills": ["Python", "FastAPI", "LangChain", ...],
      "project_summary": "...",
      "github_summary": "Recently active (12 pushes in 90 days)...",
      "strengths": ["Strong RAG project", ...],
      "concerns": [],
      "evidence": ["Python (direct): python", "AI (direct): langchain"],
      "analysis_source": "fallback"
    }
  ],
  "rejected_candidates": [
    {
      "candidate": "...",
      "eligible": false,
      "rejection_reasons": ["No evidence of Python stack"],
      "matched_skills": ["Java", "React"]
    }
  ],
  "failed_files": [],
  "duplicates": []
}
```

## Testing

```bash
# Run all tests (no network, no real LLM)
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_eligibility.py -v
python -m pytest tests/test_scoring.py -v
python -m pytest tests/test_github.py -v
```

## Design Decisions

### Filtering Strategy

Eligibility is **rule-based and outside the LLM**, using configurable lexicons in `config/settings.yaml`. This ensures:

- **Deterministic, auditable** decisions — every rejection has an exact reason string
- **Cost control** — rejected candidates never reach the LLM or GitHub APIs
- **ML-only rejection** — Classic ML/CV/NLP terms (CNN, scikit-learn, YOLO, etc.) alone do NOT qualify for the AI/agentic requirement. The system detects this case and produces a specific rejection reason: "ML only, no LLM/agentic evidence"
- **Ambiguous term handling** — Terms like "embeddings" are checked for context; "word embeddings" in a TF-IDF/NLP pipeline doesn't qualify, but "vector embeddings" with RAG/retrieval does
- **Extensibility** — All lexicons (Python-implied, AI-frameworks, AI-concepts, ML-only) live in YAML and can be extended without code changes
- **No false negatives from stack mixing** — JS/Java/React presence never causes rejection; only the absence of Python AND AI evidence does

### Scoring Strategy

The LLM extracts **FACTS and verified evidence** — code then computes points. This separation ensures:

- **Reproducibility** — Same facts always produce same score
- **All weights in config** — `config/settings.yaml` holds every point value, cap, and threshold
- **Penalties for thin wrappers** — Projects classified as `thin_wrapper` (just an API call + UI) receive -5 to -15 penalties; `tutorial_style` projects receive -5 to -8
- **Skills-only at 40%** — A skill listed only in the skills section (not used in a project/work context) earns 40% of its points
- **No-AI-project cap** — Candidates without a substantive or basic AI project are capped at 55 total points, preventing strong-backend-only candidates from ranking near the top
- **Deterministic tie-break** — Ties broken by ai_project_depth desc → python_backend desc → candidate_name asc

### LLM Usage

- **Structured output via Pydantic** — The LLM returns JSON matching the `ResumeAnalysis` schema; response is validated with Pydantic v2
- **Provider adapter pattern** — `LLMClient` Protocol with Anthropic (default) and OpenAI implementations; no provider specifics leak outside `llm/`
- **One call per eligible resume** — Returns facts and evidence, not scores
- **Repair retry** — On validation failure, the error is sent back for one repair attempt
- **Deterministic fallback** — When `--no-llm`, no API key, or an LLM call fails, the keyword-based `fallback.py` produces the same output schema
- **Prompt-injection guard** — Resume content is placed inside `<RESUME>` delimiters with explicit instructions to ignore embedded commands
- **Verbatim-evidence check** — After LLM response, evidence quotes are fuzzy-matched against the resume text; unverifiable quotes are dropped

### GitHub Scoring

- **0-5 activity** (PushEvents in 90 days) + **0-5 repos** (maintained + relevant) = **max 10**
- **2 API calls per user** — events + repos, bounded by `asyncio.Semaphore`
- **Rate-limit math** — Unauthenticated limit is 60 req/hr; 50 candidates × 2 = 100 calls can exceed this. A warning is logged when no token is set, and after a 429/403 the system stops making new calls (marks remaining candidates `rate_limited`)
- **Token via env** — `GITHUB_TOKEN` env var enables authenticated 5000 req/hr limit
- **Run-level cache** — `dict` keyed by lowercase username prevents duplicate fetches
- **Graceful statuses** — `ok`, `no_github`, `not_found`, `rate_limited`, `error` — failure is recorded, never crashes the batch

### Reliability

- **Per-file isolation** — Every file read/parse is wrapped in try/except; failures produce `FailedResume`, never crash the batch
- **Bounded concurrency** — `asyncio.Semaphore(MAX_CONCURRENCY)` limits parallel LLM + GitHub calls
- **Counts reconcile** — Pipeline asserts: `total = parsed + failed + duplicates` and `parsed = eligible + rejected`
- **asyncio.gather with return_exceptions** — Unexpected exceptions during processing still produce a fallback result for that candidate

## If I Had More Time

1. **OCR for scanned PDFs** — Use Tesseract or a cloud OCR service to handle image-only PDFs that currently fail with "no extractable text"
2. **On-disk cache with TTL** — Cache LLM and GitHub responses to disk with configurable TTL, reducing API costs on re-runs
3. **Calibration set** — Build a set of human-ranked resumes and tune weights/thresholds to minimize rank divergence
4. **Background-job API with persistence** — Replace in-memory state with a task queue (Celery/Redis) and database for production use; support long-running screening via `POST /screen` returning a job ID with `GET /jobs/{id}` for status polling

## Known Limitations

- **No OCR** — Scanned/image-only PDFs cannot be processed
- **Fallback analyzer is keyword-based** — Without the LLM, project classification relies on simple keyword heuristics, which may mis-classify nuanced projects
- **GitHub rate limits** — Without a `GITHUB_TOKEN`, the 60 req/hr limit will be hit with ~30+ candidates, leaving some without GitHub scores
- **Name extraction heuristic** — Candidate names are extracted from the first few lines of the resume using heuristics; unusual formatting may cause incorrect names
- **No persistent storage** — API results are in-memory only; restarting the server loses previous results
