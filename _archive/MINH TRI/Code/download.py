import fastf1
import pandas as pd
import os
import time

# 1. Setup Cache (Absolute necessity to avoid re-downloading)
cache_dir = './f1_raw_cache'
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

csv_base_dir = './f1_data_csv_export'
if not os.path.exists(csv_base_dir): os.makedirs(csv_base_dir)

def export_season_to_csv(year):
    year_dir = os.path.join(csv_base_dir, str(year))
    if not os.path.exists(year_dir): os.makedirs(year_dir)
    
    # Try-except for the schedule itself
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"!!! Critical: Could not load schedule for {year}. Waiting 60s...")
        time.sleep(60) # Long wait if the API is angry
        return

    for _, event in schedule.iterrows():
        if event['EventFormat'] == 'testing': continue
        
        gp = event['EventName']
        # Skip if already downloaded (Checkpoint logic)
        if os.path.exists(f"{year_dir}/{gp}_Laps.csv"):
            print(f"Skipping {year} {gp} - Already exists.")
            continue
            
        print(f"Processing: {year} {gp}")
        
        try:
            session = fastf1.get_session(year, event['RoundNumber'], 'R')
            
            # Telemetry only for 2018-2025 range
            load_telemetry = True if year >= 2018 else False
            session.load(telemetry=load_telemetry)

            session.results.to_csv(f"{year_dir}/{gp}_Results.csv")
            session.laps.to_csv(f"{year_dir}/{gp}_Laps.csv")

            if load_telemetry:
                all_tel = []
                for driver in session.drivers:
                    tel = session.laps.pick_driver(driver).get_telemetry()
                    tel['Driver'] = driver
                    all_tel.append(tel)
                pd.concat(all_tel).to_csv(f"{year_dir}/{gp}_Telemetry.csv")

            # BE POLITE: Wait 3-5 seconds between races to avoid 429 errors
            time.sleep(3) 

        except Exception as e:
            print(f"Error on {year} {gp}: {e}")
            time.sleep(10) # Brief cooling period after an error

# Run the range
for y in range(2018, 2026):
    export_season_to_csv(y)