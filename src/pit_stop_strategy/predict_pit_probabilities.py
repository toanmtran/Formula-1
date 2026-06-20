import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.pit_stop_strategy.paths import AGGREGATED_CSV, BEST_MODEL_PT

DATA_PATH = str(AGGREGATED_CSV)
OUTPUT_PATH = str(AGGREGATED_CSV)
MODEL_PATH = str(BEST_MODEL_PT)

SEQ_LENGTH = 20
BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class F1PitStopPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=256, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return x


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    original_df = df.copy()

    drop_cols = [
        'PitInTime', 'PitOutTime', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'SessionTime', 'Time', 'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate'
    ]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols, errors='ignore')

    print("Engineering features...")
    historical_pit = (
        df[df['HasPitStop'] == 1]
        .groupby(['Compound', 'Team'])['LapNumber']
        .mean()
        .reset_index()
    )
    historical_pit.columns = ['Compound', 'Team', 'historical_pit_lap']
    df = df.drop(columns=['historical_pit_lap'], errors='ignore')
    df = df.merge(historical_pit, on=['Compound', 'Team'], how='left')

    global_pit_mean = historical_pit['historical_pit_lap'].mean()
    df['historical_pit_lap'] = df['historical_pit_lap'].fillna(global_pit_mean)
    df['pit_window_delta'] = df['LapNumber'] - df['historical_pit_lap']

    categorical_cols = ['Compound', 'Team']
    continuous_cols = [
        'LapTime_Seconds', 'Position', 'LapNumber', 'TyreLife', 'TrackStatus',
        'delta_laptime', 'CumulativeTimeStint', 'race_progress_fraction',
        'relative_tire_age', 'tire_compound_age_delta', 'tire_performance_decay',
        'rolling_pace_mean_5', 'pace_std_5', 'pace_degradation_slope',
        'historical_pit_lap', 'pit_window_delta',
        'Speed_mean', 'Speed_max', 'Speed_std', 'RPM_mean', 'RPM_max',
        'Throttle_mean', 'Throttle_std', 'Brake_mean', 'Brake_sum',
        'DRS_mean', 'DRS_sum', 'DistanceToDriverAhead_mean', 'DistanceToDriverAhead_min',
        'nGear_mean', 'nGear_max', 'traffic_pressure', 'drs_dependency',
        'thermal_stress_proxy', 'high_speed_stress', 'brake_aggression', 'pace_vs_ahead',
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    encoded_cat_cols = [col for col in df.columns
                        if col.startswith("Compound_") or col.startswith("Team_")]
    feature_cols = continuous_cols + encoded_cat_cols

    print("Normalizing data...")
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    df = df.fillna(0)

    print("Building sequences...")
    df['original_row_id'] = df.index

    X = []
    predicted_indices = []
    grouped = df.groupby(['Year', 'RaceID', 'DriverNumber'])

    for _, group in grouped:
        group = group.sort_values('LapNumber')
        feat = group[feature_cols].values
        orig_ids = group['original_row_id'].values

        if len(group) < SEQ_LENGTH:
            continue

        for i in range(len(group) - SEQ_LENGTH + 1):
            X.append(feat[i:i + SEQ_LENGTH])
            predicted_indices.append(orig_ids[i + SEQ_LENGTH - 1])

    X_tensor = torch.tensor(np.array(X, dtype=np.float32))
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model from {MODEL_PATH}...")
    model = F1PitStopPredictor(input_size=X_tensor.shape[2]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running inference across entire dataset...")
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(DEVICE)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                all_probs.append(probs.item())
            else:
                all_probs.extend(probs.tolist())

    print("Merging probabilities back into original dataset...")
    # First (SEQ_LENGTH - 1) laps of every stint have no prediction window.
    original_df['PitStopProbability'] = 0.0
    original_df.loc[predicted_indices, 'PitStopProbability'] = all_probs

    print(f"Saving updated data to {OUTPUT_PATH}...")
    original_df.to_csv(OUTPUT_PATH, index=False)
    print("Success.")


if __name__ == "__main__":
    main()
