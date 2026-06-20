import os
import pandas as pd
import numpy as np
from scipy import stats
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

ERGAST_DIR = "C:/Nguyen Tri/Code/Statisanalyss/Data/formula-1-race-data/versions/116"
FASTF1_DIR = "C:/Nguyen Tri/Code/Statisanalyss/all_training_data.csv"

def load_ergast():
    pit_stops = pd.read_csv(f"{ERGAST_DIR}/pit_stops.csv")
    results = pd.read_csv(f"{ERGAST_DIR}/results.csv")
    constructors = pd.read_csv(f"{ERGAST_DIR}/constructors.csv")
    driver_standings = pd.read_csv(f"{ERGAST_DIR}/driver_standings.csv")
    return pit_stops, results, constructors, driver_standings

def analyze_pit_stop_correlation():
    print("\n--- Strategic Regression: Pit Duration vs Championship Points ---")
    try:
        pit_stops, results, constructors, _ = load_ergast()
    except FileNotFoundError:
        print("Ergast data not found.")
        return

    # Clean pit stop durations (some might be strings like '1:15.2')
    # Filter out abnormally long pit stops > 40 seconds (likely repairs/penalties)
    pit_stops['duration_sec'] = pd.to_numeric(pit_stops['duration'], errors='coerce')
    pit_stops = pit_stops[pit_stops['duration_sec'] < 40].copy()

    # Merge to get constructorId for each pit stop
    # results table maps raceId and driverId to constructorId and points
    res_map = results[['raceId', 'driverId', 'constructorId', 'points']].drop_duplicates()
    
    merged = pd.merge(pit_stops, res_map, on=['raceId', 'driverId'], how='inner')
    
    # Calculate median pit stop duration by constructor per race, then average across all races
    team_pits = merged.groupby('constructorId')['duration_sec'].median().reset_index()
    
    # Total points by constructor (historical)
    team_points = results.groupby('constructorId')['points'].sum().reset_index()
    
    team_stats = pd.merge(team_pits, team_points, on='constructorId')
    team_stats = pd.merge(team_stats, constructors[['constructorId', 'name']], on='constructorId')
    
    # Keep only active/modern teams with significant points to avoid outliers from 1950s
    team_stats = team_stats[team_stats['points'] > 100]
    
    correlation, p_value = stats.pearsonr(team_stats['duration_sec'], team_stats['points'])
    
    print("Top 5 Teams by Fastest Median Pit Stop:")
    print(team_stats.sort_values('duration_sec').head()[['name', 'duration_sec', 'points']].to_string(index=False))
    
    print(f"\nPearson Correlation (Median Pit Duration vs Total Points): {correlation:.4f}")
    print(f"P-Value: {p_value:.4e}")
    if p_value < 0.05:
        if correlation < 0:
            print("Definitive Statistical Correlation: Faster pit stops are significantly correlated with more points.")
        else:
            print("Definitive Statistical Correlation: Slower pit stops are correlated with more points (Unexpected).")
    else:
        print("No definitive statistical correlation found between median pit stop duration and total points.")


def analyze_tactical_agility():
    print("\n--- Agility Prediction: VSC Reaction Time vs Finishing Position ---")
    
    if os.path.isfile(FASTF1_DIR):
        df = pd.read_csv(FASTF1_DIR)
        if 'RaceID' in df.columns and 'Race' not in df.columns:
            df['Race'] = df['RaceID']
    else:
        df_list = []
        for root, _, files in os.walk(FASTF1_DIR):
            for file in files:
                if file.endswith("_processed.csv"):
                    df_temp = pd.read_csv(os.path.join(root, file))
                    df_temp['Race'] = file.replace("_processed.csv", "")
                    df_list.append(df_temp)
                    
        if not df_list:
            print("No FastF1 preprocessed data found.")
            return
            
        df = pd.concat(df_list, ignore_index=True)
    
    # Build dataset per driver per race
    driver_race_stats = []
    
    for (race, driver), group in df.groupby(['Race', 'Driver']):
        group = group.sort_values('LapNumber')
        if len(group) < 10:
            continue
            
        start_pos = group.iloc[0]['Position']
        end_pos = group.iloc[-1]['Position']
        finished_higher = 1 if end_pos < start_pos else 0
        
        # Calculate Tactical Agility (reaction to VSC)
        # Find VSC/SC deployments
        if 'Stochastic_Shock_VSC_SC' in group.columns:
            vsc_laps = group[group['Stochastic_Shock_VSC_SC'] == 1]['LapNumber'].values
        else:
            vsc_laps = []
        pit_laps = group[group['HasPitStop'] == 1]['LapNumber'].values
        
        reaction_times = []
        for vsc in vsc_laps:
            # Did they pit within 3 laps AFTER the VSC?
            future_pits = pit_laps[pit_laps >= vsc]
            if len(future_pits) > 0:
                reaction = future_pits[0] - vsc
                if reaction <= 3:
                    reaction_times.append(reaction)
                    
        # Feature: mean reaction time to VSC (lower is more agile). If they never pitted under VSC, give penalty.
        agility_score = np.mean(reaction_times) if len(reaction_times) > 0 else 5.0
        
        # Other features to help the model
        if 'Stationary_PaceDecay' in group.columns:
            avg_pace_decay = group['Stationary_PaceDecay'].mean()
        elif 'pace_degradation_slope' in group.columns:
            avg_pace_decay = group['pace_degradation_slope'].mean()
        else:
            avg_pace_decay = 0
        prob_col = 'PitStopProbability' if 'PitStopProbability' in group.columns else 'Pit_Probability'
        mean_pit_prob = group[prob_col].mean() if prob_col in group.columns else 0
        
        driver_race_stats.append({
            'Driver': driver,
            'Race': race,
            'GridPos': start_pos,
            'Agility_Score': agility_score,
            'PaceDecay': avg_pace_decay,
            'Mean_Pit_Prob': mean_pit_prob,
            'Finished_Higher': finished_higher
        })
        
    dataset = pd.DataFrame(driver_race_stats).dropna()
    
    if len(dataset) < 10:
        print("Not enough agility data to train a model.")
        return
        
    print(f"Dataset compiled: {len(dataset)} driver-race entries.")
    print(f"Average Agility Score (Laps to react): {dataset['Agility_Score'].mean():.2f}")
    
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed. Skipping XGBoost/RandomForest model training.")
        return
        
    X = dataset[['GridPos', 'Agility_Score', 'PaceDecay', 'Mean_Pit_Prob']]
    y = dataset['Finished_Higher']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"\nRandom Forest Classifier Accuracy (Predicting 'Finished Higher Than Qualified'): {acc*100:.2f}%")
    
    importances = clf.feature_importances_
    features = X.columns
    print("\nFeature Importances:")
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp:.4f}")
        
    if importances[1] > 0.15:
        print("\nConclusion: Tactical Agility (Reaction to VSC) is a highly significant predictor of finishing above qualifying position.")
    else:
        print("\nConclusion: Tactical Agility was less important than baseline pace/grid position in this subset.")

if __name__ == "__main__":
    analyze_pit_stop_correlation()
    analyze_tactical_agility()
