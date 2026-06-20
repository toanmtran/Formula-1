import pandas as pd
from .config import *


def load_and_merge_data() -> pd.DataFrame:
    print("Loading data...")
    results   = pd.read_csv(RESULTS_PATH)
    races     = pd.read_csv(RACES_PATH)
    drivers   = pd.read_csv(DRIVERS_PATH)
    circuits  = pd.read_csv(CIRCUITS_PATH)
    circuits['country'] = circuits['country'].replace({'United States': 'USA'})

    countries_map = pd.read_csv(COUNTRIES_MAP_PATH)

    print("Merging datasets...")
    df = results.merge(races[['raceId', 'year', 'circuitId']], on='raceId')
    df = df.merge(drivers[['driverId', 'driverRef', 'nationality']], on='driverId')
    df = df.merge(circuits[['circuitId', 'circuitRef', 'country']], on='circuitId')
    df = df.merge(countries_map[['nationality', 'en_short_name']], on='nationality', how='left')

    return df
