# Project Architecture

## Pipeline stages

1. **Preprocessing** (`steel_pipeline.preprocess`)
   - load raw steel compositions
   - standardize feature columns
   - impute mechanical-property gaps
   - derive `strength_rating`

2. **Training** (`steel_pipeline.train`)
   - train/evaluate regression models for yield/tensile/elongation
   - train/evaluate classifiers for strength class
   - save best artifacts and metrics

3. **Optimization** (`steel_pipeline.optimize`)
   - generate synthetic composition candidates
   - predict performance with trained models
   - filter candidates by engineering constraints

## Key outputs

- `data/interim/normalized_file_complete.xlsx`
- `data/processed/imputed_materials_data.xlsx`
- `data/processed/final_steel_data.xlsx`
- `models/*.pkl`
- `reports/summary/model_metrics.json`
- `reports/summary/optimized_candidates.csv`
