# Steel Design Challenge (2024–2025)

A cleaned, recruiter-friendly machine learning project for alloy design.

## Project goal
Predict steel mechanical properties from composition and generate candidate alloys that balance high strength with low nickel/cobalt content.

## Repository structure

- `src/steel_pipeline/` — production-ready pipeline modules
- `scripts/` — simple command entry points
- `data/raw/` — original dataset inputs
- `data/interim/` — normalized/intermediate artifacts
- `data/processed/` — final model-ready datasets
- `models/` — trained model artifacts (`.pkl`)
- `reports/figures/` — confusion matrices and visuals
- `reports/summary/` — metrics and result summaries
- `notebooks/` — lecture and challenge notebooks
- `archive/` — legacy scripts and old exports retained for traceability

## Quick start

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m steel_pipeline preprocess
python -m steel_pipeline train
python -m steel_pipeline optimize --samples 1000
```

Or run all steps:

```bash
python -m steel_pipeline all
```

## Clean-code improvements applied

- Removed hardcoded personal paths and interactive user prompts
- Replaced monolithic scripts with single-responsibility modules
- Added reproducible, path-portable CLI workflow
- Consolidated artifacts into explicit data/model/report directories
- Archived legacy scripts and duplicate outputs separately

## Validation commands

```bash
python -m compileall src scripts
python -m steel_pipeline preprocess
python -m steel_pipeline train
python -m steel_pipeline optimize --samples 250
```

## Notes

- Canonical raw source: `data/raw/database_steel_properties.csv`
- Canonical final dataset: `data/processed/final_steel_data.xlsx`
- Legacy exploratory material is preserved under `archive/` and `notebooks/`
