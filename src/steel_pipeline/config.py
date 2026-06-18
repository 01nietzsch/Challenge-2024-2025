from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_DIR = REPORTS_DIR / "summary"

FEATURE_COLUMNS = ["fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
MECHANICAL_COLUMNS = ["yield strength", "tensile strength", "elongation"]

RAW_DATASET = RAW_DATA_DIR / "database_steel_properties.csv"
INTERIM_NORMALIZED = INTERIM_DATA_DIR / "normalized_file_complete.xlsx"
IMPUTED_DATASET = PROCESSED_DATA_DIR / "imputed_materials_data.xlsx"
FINAL_DATASET = PROCESSED_DATA_DIR / "final_steel_data.xlsx"
METRICS_JSON = SUMMARY_DIR / "model_metrics.json"
CANDIDATES_CSV = SUMMARY_DIR / "optimized_candidates.csv"
