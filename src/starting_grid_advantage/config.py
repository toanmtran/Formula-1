import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR     = os.path.join(REPO_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'starting_grid_processed')

OUTPUT_DIR  = os.path.join(REPO_ROOT, 'outputs', 'starting_grid_advantage')
PLOT_DIR    = os.path.join(OUTPUT_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
MODEL_DIR   = os.path.join(OUTPUT_DIR, 'models')


RANDOM_SEED = 42
TEST_SIZE = 0.2

STREET_CIRCUITS = [
    'monaco',
    'baku',
    'marina_bay',
    'jeddah',
    'vegas',
    'miami',
    'albert_park',
]


RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8

DL_EPOCHS = 15
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.01


for directory in [PROCESSED_DATA_DIR, PLOT_DIR, METRICS_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)
