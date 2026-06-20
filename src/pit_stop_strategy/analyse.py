import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

from src.pit_stop_strategy.paths import PROCESSED_DIR, PLOTS_DIR, PIPELINE_DATA_DIR

file_pattern = str(PROCESSED_DIR)

df_list = []
for root, _, files in os.walk(file_pattern):
    for file in files:
        if file.endswith("_processed.csv"):
            file_path = os.path.join(root, file)
            df_temp = pd.read_csv(file_path)
            df_temp['SourceFile'] = file
            df_list.append(df_temp)

df_all = pd.concat(df_list, ignore_index=True)

_out_csv = PIPELINE_DATA_DIR / "All_Races_Aggregated.csv"
df_all.to_csv(_out_csv, index=False)
print(f"Aggregated {len(df_list)} files into '{_out_csv}'")


lap_time  = df_all['LapTime_Seconds'].dropna()
tyre_life = df_all['TyreLife'].dropna()
pit_laps  = df_all[df_all['HasPitStop'] == True]['LapNumber'].dropna()

df_sorted = df_all.sort_values(by=['SourceFile', 'DriverNumber', 'LapNumber'])

pos_gains = []
for (race, driver), group in df_sorted.groupby(['SourceFile', 'DriverNumber']):
    if not group.empty:
        start_pos = group.iloc[0]['Position']
        end_pos   = group.iloc[-1]['Position']
        pos_gains.append(start_pos - end_pos)

pos_gains = np.array(pos_gains)
pos_gains_clean = pos_gains[~np.isnan(pos_gains)]

stats = [
    {'Variable': 'Lap Time (s)',        'Mean': lap_time.mean(),         'SD': lap_time.std(),         'Min': lap_time.min(),         'Max': lap_time.max()},
    {'Variable': 'Tyre Life (laps)',    'Mean': tyre_life.mean(),        'SD': tyre_life.std(),        'Min': tyre_life.min(),        'Max': tyre_life.max()},
    {'Variable': 'Pit Lap',             'Mean': pit_laps.mean(),         'SD': pit_laps.std(),         'Min': pit_laps.min(),         'Max': pit_laps.max()},
    {'Variable': 'Total Position Gain', 'Mean': pos_gains_clean.mean(),  'SD': pos_gains_clean.std(),  'Min': pos_gains_clean.min(),  'Max': pos_gains_clean.max()},
]

stats_df = pd.DataFrame(stats)
print("\n--- Descriptive Statistics ---")
print(stats_df.to_string(index=False))


sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(lap_time, bins=50, kde=True, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('Lap Time Distribution (s)')
axes[0, 0].set_xlabel('Lap Time (Seconds)')

sns.histplot(pit_laps, bins=30, kde=True, ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Pit-Stop Lap Distribution')
axes[0, 1].set_xlabel('Lap Number')

sns.histplot(tyre_life, bins=30, kde=True, ax=axes[1, 0], color='green')
axes[1, 0].set_title('Tyre Life Distribution')
axes[1, 0].set_xlabel('Tyre Life (Laps)')

sns.histplot(pos_gains_clean, bins=20, kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Position Gain Distribution per Driver')
axes[1, 1].set_xlabel('Positions Gained (+)')

plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'Aggregated_Distributions.png'))
plt.close()


df_sorted['PosChange']    = df_sorted.groupby(['SourceFile', 'DriverNumber'])['Position'].diff().fillna(0)
df_sorted['AbsPosChange'] = df_sorted['PosChange'].abs()

changes_per_lap = df_sorted.groupby('LapNumber')['AbsPosChange'].sum()

early_changes = changes_per_lap[changes_per_lap.index <= 15].sum()
mid_changes   = changes_per_lap[(changes_per_lap.index > 15) & (changes_per_lap.index <= 40)].sum()
late_changes  = changes_per_lap[changes_per_lap.index > 40].sum()

print("\n--- Position Gain Timing ---")
print(f"Early Laps (1-15) Total Changes: {early_changes}")
print(f"Mid Laps (16-40) Total Changes: {mid_changes}")
print(f"Late Laps (41+) Total Changes: {late_changes}")


stints_per_driver = df_all.groupby('DriverNumber')['Stint'].nunique()
stint_counts      = df_all['Stint'].value_counts().sort_index()
deg_per_compound  = df_all.groupby('Compound')['tire_performance_decay'].mean()

print("\n--- Tyre Strategy Analysis ---")
print("Stints per Driver (All Races):")
print(stints_per_driver.value_counts())
print("\nAverage Tyre Degradation by Compound:")
print(deg_per_compound)

if 'Team' in df_all.columns:
    pit_stops = df_all[df_all['HasPitStop'] == 1]
    if len(pit_stops) > 0:
        team_variance = pit_stops.groupby('Team')['TyreLife'].agg(['mean', 'var']).dropna()
        team_variance.columns = ['E[X] (Mean)', 'Var(X) (Variance)']
        print("\nTyrelife Distribution by Team (Risk Profile):")
        print(team_variance.sort_values(by='Var(X) (Variance)', ascending=True))


corr_lap_wear = df_all['LapTime_Seconds'].corr(df_all['TyreLife'])
print(f"\nCorrelation between Lap Time and Tyre Life: {corr_lap_wear:.3f}")

corr_dist_speed = df_all['DistanceToDriverAhead_mean'].corr(df_all['Speed_mean'])
print(f"Correlation between Distance to Leader and Speed: {corr_dist_speed:.3f}")


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_all,
    x='TyreLife',
    y='tire_performance_decay',
    hue='Compound',
    errorbar='ci',
    marker='o',
)
plt.title('Tyre Performance Decay Over Tyre Life (All Races)')
plt.xlabel('Tyre Life (Laps)')
plt.ylabel('Tyre Performance Decay (s)')
plt.legend(title='Compound')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'Tyre_Degradation_Curve.png'))
plt.close()
