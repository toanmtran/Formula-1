from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ERGAST_DIR = REPO_ROOT / "data" / "raw"

# Large telemetry files live on E: to save space on C:/D: (the raw 2018-2025
# FastF1 download is ~10 GB).  Fall back to the in-repo location if E: is not
# available.
_E_DATA_DIR = Path("E:/f1_pipeline_data/pit_stop_pipeline")
PIPELINE_DATA_DIR = _E_DATA_DIR if _E_DATA_DIR.parent.exists() \
                   else REPO_ROOT / "data" / "pit_stop_pipeline"
RAW_FASTF1_DIR    = PIPELINE_DATA_DIR / "f1_data_csv_export"
PROCESSED_DIR     = PIPELINE_DATA_DIR / "preprocessed"
AGGREGATED_CSV    = PIPELINE_DATA_DIR / "all_training_data.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "pit_stop_strategy"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

BEST_MODEL_PT   = MODELS_DIR / "best_f1_model.pt"
BILSTM_MODEL_PT = MODELS_DIR / "f1_bilstm_model.pt"

for _d in (OUTPUT_DIR, PLOTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
