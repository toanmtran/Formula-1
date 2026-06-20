import fastf1
import pandas as pd
import time

from .paths import PIPELINE_DATA_DIR, RAW_FASTF1_DIR

PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
cache_dir = PIPELINE_DATA_DIR / "f1_raw_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

csv_base_dir = RAW_FASTF1_DIR
csv_base_dir.mkdir(parents=True, exist_ok=True)


def export_season_to_csv(year):
    year_dir = csv_base_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"!!! Critical: Could not load schedule for {year}. Waiting 60s...")
        time.sleep(60)
        return

    for _, event in schedule.iterrows():
        if event['EventFormat'] == 'testing':
            continue

        gp = event['EventName']
        if (year_dir / f"{gp}_Laps.csv").exists():
            print(f"Skipping {year} {gp} - Already exists.")
            continue

        print(f"Processing: {year} {gp}")

        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')

            load_telemetry = True if year >= 2018 else False
            session.load(telemetry=load_telemetry)

            session.results.to_csv(year_dir / f"{gp}_Results.csv")
            session.laps.to_csv(year_dir / f"{gp}_Laps.csv")

            if load_telemetry:
                all_tel = []
                for driver in session.drivers:
                    tel = session.laps.pick_driver(driver).get_telemetry()
                    tel['Driver'] = driver
                    all_tel.append(tel)
                pd.concat(all_tel).to_csv(year_dir / f"{gp}_Telemetry.csv")

            # Polite delay to avoid HTTP 429 from the FastF1 API
            time.sleep(3)

        except Exception as e:
            print(f"Error on {year} {gp}: {e}")
            time.sleep(10)


if __name__ == "__main__":
    for y in range(2018, 2026):
        export_season_to_csv(y)
