import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'raw')

OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs', 'home_advantage')
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figures')
TAB_DIR    = os.path.join(OUTPUT_DIR, 'tables')

RESULTS_PATH       = os.path.join(DATA_DIR, 'results.csv')
RACES_PATH         = os.path.join(DATA_DIR, 'races.csv')
DRIVERS_PATH       = os.path.join(DATA_DIR, 'drivers.csv')
CIRCUITS_PATH      = os.path.join(DATA_DIR, 'circuits.csv')
STATUS_PATH        = os.path.join(DATA_DIR, 'status.csv')
COUNTRIES_MAP_PATH = os.path.join(DATA_DIR, 'countries.csv')

for _d in (OUTPUT_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)
