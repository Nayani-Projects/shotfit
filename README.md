# ShotFit — 2025–26 Shot-Making Evidence Board

ShotFit asks one descriptive question:

> Which NBA players produced the strongest evidence of shot-making above or below expectation in 2025–26, and where did the difference originate?

The app compares actual makes with calibrated expected makes from the same observable public shot context, adjusts player and area results for sample size, and reports an 80% range. It does **not** forecast future shooting, recommend a role or acquisition, or prescribe shot selection.

## Quickstart

```bash
uv sync --all-groups
uv run python -m shotfit.cli ingest
uv run python -m shotfit.cli build-features
uv run python -m shotfit.cli train
uv run python -m shotfit.cli export-app
uv run streamlit run streamlit_app.py
```

Or run the complete offline pipeline with `uv run python -m shotfit.cli all`. On macOS, `run_local.command` starts the prebuilt app locally.

## Analytical contract

- 2023–24 trains the shot-context model.
- 2024–25 selects and calibrates the model and sets public evidence standards.
- Untouched 2025–26 shots exclusively produce public player, area, and court results.
- Player and team identity are excluded from shot difficulty.
- Players need at least 250 test-season attempts.
- Strong positive evidence requires the entire 80% adjusted range above zero.
- Strong negative evidence requires the entire range below zero.
- Every other result is inconclusive.
- Shot-profile labels use validation-season positional frequency benchmarks, not performance.

The 250-attempt standard was chosen before applying labels to 2025–26. On 2024–25 validation data it produced a 0.62 deterministic split-half correlation while retaining 306 players; higher thresholds improved stability but removed substantially more of the population.

## App structure

- **Evidence Board:** filter and compare all qualified players.
- **Player Brief:** inspect the adjusted estimate, range, shot distribution, court map, area evidence, and review flags.
- **Model & Validation:** audit model quality, calibration, threshold selection, interval sensitivity, and limitations.

## Repository map

- `src/shotfit/`: ingestion, features, modeling, evidence generation, court rendering, and CLI
- `data/app/`: compact precomputed Parquet and JSON bundle used at runtime
- `tests/`: deterministic unit, contract, and artifact-reconciliation tests
- `docs/`: architecture and data dictionary
- `MODEL_CARD.md`: methodology, intended use, and limitations

## Data policy and limitations

Raw responses, DuckDB, and full shot-level files remain out of Git. The public app makes no runtime NBA.com calls.

Public shot records omit shot-level defender distance, pass quality, movement, balance, screen quality, play design, health, and internal role context. ShotFit describes performance relative to observable context in this sample; it does not establish future performance, team fit, or the shots a player should take. No OKC hiring-project material is used.
