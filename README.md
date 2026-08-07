# ShotFit

ShotFit is a public-data NBA decision brief that asks:

> Does this player's shooting appear likely to translate, where does the evidence come from, and what role should basketball staff investigate next?

The application leads with basketball language and keeps technical validation in a separate tab. Staff can narrow the 2025–26 player pool by team and primary roster position. ShotFit compares a zone baseline, logistic regression, and XGBoost with chronological validation, then stabilizes player results for sample size.

## Quickstart

```bash
uv sync --all-groups
uv run python -m shotfit.cli ingest
uv run python -m shotfit.cli build-features
uv run python -m shotfit.cli train
uv run python -m shotfit.cli export-app
uv run streamlit run streamlit_app.py
```

Or run the complete offline pipeline:

```bash
uv run python -m shotfit.cli all
```

## Design principles

- Team-batched API retrieval avoids the observed 102,400-row silent truncation.
- Every response is cached and checksummed before transformation.
- Player and team identity are excluded from shot difficulty.
- 2025–26 is the untouched chronological test and public eligibility season; 2024–25 supplies supporting validation evidence.
- The public app uses precomputed artifacts and makes no runtime network calls.
- Automatic roles are descriptive scouting prompts, not causal projections.

## Repository map

- `src/shotfit/`: ingestion, features, modeling, evaluation, and CLI
- `streamlit_app.py`: Basketball Brief and Model & Validation tabs
- `tests/`: deterministic unit and contract tests
- `docs/`: architecture and data dictionary
- `MODEL_CARD.md`: intended use, methodology, and limitations

## Validation and limitations

The generated application reports real validation and untouched-test metrics after the pipeline runs. See [MODEL_CARD.md](MODEL_CARD.md) for the full methodology.

Public NBA shot records do not contain shot-level defender distance, pass quality, movement, balance, exact play design, health, or internal scouting context. ShotFit should be used with film, tracking, and scouting evidence.

## Data policy

Raw NBA responses, the local DuckDB database, and full shot-level feature files are not committed. The public repository contains reproducible ingestion code, schemas, tests, model metadata, and a compact derived app bundle. No OKC hiring-project material is used.
