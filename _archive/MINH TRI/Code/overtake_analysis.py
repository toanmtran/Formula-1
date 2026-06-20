import os
import pandas as pd
import numpy as np

def load_preprocessed_data(data_path="C:/Nguyen Tri/Code/Statisanalyss/all_training_data.csv"):
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        if 'RaceID' in df.columns and 'Race' not in df.columns:
            df['Race'] = df['RaceID']
        return df

    df_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith("_processed.csv"):
                file_path = os.path.join(root, file)
                df_temp = pd.read_csv(file_path)
                if 'Driver' in df_temp.columns and 'LapNumber' in df_temp.columns:
                    df_temp['Race'] = file.replace("_processed.csv", "")
                    df_list.append(df_temp)
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def analyze_overtakes(df):
    print("\n--- Overtake Classification ---")
    
    total_strategic = 0
    total_on_track = 0
    
    # Process race by race
    for race, race_df in df.groupby('Race'):
        # Pivot positions to trace position changes lap by lap
        try:
            pos_df = race_df.pivot(index='LapNumber', columns='Driver', values='Position')
            prob_col = 'PitStopProbability' if 'PitStopProbability' in race_df.columns else 'Pit_Probability'
            pit_prob_df = race_df.pivot(index='LapNumber', columns='Driver', values=prob_col)
            has_pit_df = race_df.pivot(index='LapNumber', columns='Driver', values='HasPitStop')
        except Exception:
            continue
            
        drivers = pos_df.columns
        for lap in pos_df.index[:-1]:
            current_pos = pos_df.loc[lap]
            next_pos = pos_df.loc[lap + 1]
            
            # Find overtakes between driver A and driver B
            for i, driver_A in enumerate(drivers):
                for driver_B in drivers[i+1:]:
                    pos_A_curr, pos_B_curr = current_pos[driver_A], current_pos[driver_B]
                    pos_A_next, pos_B_next = next_pos[driver_A], next_pos[driver_B]
                    
                    if pd.isna(pos_A_curr) or pd.isna(pos_B_curr) or pd.isna(pos_A_next) or pd.isna(pos_B_next):
                        continue
                        
                    # Check for intersect (position swap)
                    # Ignore swaps where someone DNF'd or if it's outside top 10 as per prompt (optional, but let's check top 10)
                    if min(pos_A_curr, pos_B_curr) > 10 and min(pos_A_next, pos_B_next) > 10:
                        continue
                        
                    if (pos_A_curr > pos_B_curr and pos_A_next < pos_B_next) or \
                       (pos_A_curr < pos_B_curr and pos_A_next > pos_B_next):
                        
                        # We have an overtake
                        # Check Strategic vs On-track
                        # A window of +/- 2 laps for either driver A or B
                        window_start = max(1, lap - 2)
                        window_end = lap + 2
                        
                        strategic = False
                        
                        # 1. Did anyone pit nearby?
                        if has_pit_df.loc[window_start:window_end, driver_A].sum() > 0 or \
                           has_pit_df.loc[window_start:window_end, driver_B].sum() > 0:
                            strategic = True
                            
                        # 2. Was Pit_Probability > 0.8?
                        elif pit_prob_df.loc[lap, driver_A] > 0.8 or pit_prob_df.loc[lap, driver_B] > 0.8:
                            strategic = True
                            
                        if strategic:
                            total_strategic += 1
                        else:
                            total_on_track += 1

    total_overtakes = total_strategic + total_on_track
    if total_overtakes > 0:
        print(f"Total Top-10 Position Changes: {total_overtakes}")
        print(f"Strategic Overtakes (Pit/Probability driven): {total_strategic} ({(total_strategic/total_overtakes)*100:.2f}%)")
        print(f"On-Track Overtakes: {total_on_track} ({(total_on_track/total_overtakes)*100:.2f}%)")
    else:
        print("No overtakes detected.")

def analyze_vsc_time_save(df):
    print("\n--- VSC vs Green Flag Time-Save ---")
    
    # Isolate IN laps (HasPitStop == 1)
    in_laps = df[df['HasPitStop'] == 1].copy()
    
    if len(in_laps) == 0:
        print("No pit stops found.")
        return
        
    if 'Stochastic_Shock_VSC_SC' not in in_laps.columns:
        print("VSC shock data not available in this dataset. Skipping VSC vs Green Flag comparison.")
        return

    green_flag_pits = in_laps[in_laps['Stochastic_Shock_VSC_SC'] == 0]['LapTime_Seconds'].dropna()
    vsc_pits = in_laps[in_laps['Stochastic_Shock_VSC_SC'] == 1]['LapTime_Seconds'].dropna()
    
    print(f"Green Flag Pit IN Laps: {len(green_flag_pits)}")
    print(f"VSC/SC Pit IN Laps: {len(vsc_pits)}")
    
    if len(green_flag_pits) > 0 and len(vsc_pits) > 0:
        gf_mean = green_flag_pits.median()
        vsc_mean = vsc_pits.median()
        time_saved = gf_mean - vsc_mean
        
        print(f"Median Lap Time (Green Flag Pit): {gf_mean:.2f} seconds")
        print(f"Median Lap Time (VSC Pit): {vsc_mean:.2f} seconds")
        # Under VSC, the *field* slows down. So pitting under VSC relative to green flag 
        # means you lose LESS time relative to the field. 
        # Actually, the Lap Time itself during a pit is long. 
        # We need the delta to the median field pace.
        print(f"Note: This measures absolute pit-lap duration, but the true strategic value is relative to the field.")
        
    else:
        print("Not enough data to compare VSC vs Green Flag pit stops in 2023 subset.")

if __name__ == "__main__":
    df = load_preprocessed_data()
    if not df.empty:
        analyze_overtakes(df)
        analyze_vsc_time_save(df)
    else:
        print("No data loaded.")
