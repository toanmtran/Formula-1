import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve
)


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "C:/Nguyen Tri/Code/Statisanalyss/all_training_data.csv"

SEQ_LENGTH = 20

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)


# =========================================================
# REMOVE LEAKAGE
# =========================================================

drop_cols = [

    'PitInTime',
    'PitOutTime',

    'LapTime',
    'Sector1Time',
    'Sector2Time',
    'Sector3Time',

    'SessionTime',
    'Time',

    'Deleted',
    'DeletedReason',

    'FastF1Generated',
    'IsAccurate'
]

existing_drop_cols = [

    col for col in drop_cols
    if col in df.columns
]

df = df.drop(
    columns=existing_drop_cols,
    errors='ignore'
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

if 2025 in df['Year'].values:

    train_df = df[
        df['Year'] < 2025
    ].copy()

    test_df = df[
        df['Year'] == 2025
    ].copy()

else:

    latest_year = df['Year'].max()

    train_df = df[
        df['Year'] < latest_year
    ].copy()

    test_df = df[
        df['Year'] == latest_year
    ].copy()


# =========================================================
# FIX HISTORICAL PIT LEAKAGE
# =========================================================

historical_pit = (

    train_df[
        train_df['HasPitStop'] == 1
    ]

    .groupby(
        ['Compound', 'Team']
    )['LapNumber']

    .mean()

    .reset_index()
)

historical_pit.columns = [

    'Compound',
    'Team',
    'historical_pit_lap'
]

train_df = train_df.drop(
    columns=['historical_pit_lap'],
    errors='ignore'
)

test_df = test_df.drop(
    columns=['historical_pit_lap'],
    errors='ignore'
)

train_df = train_df.merge(
    historical_pit,
    on=['Compound', 'Team'],
    how='left'
)

test_df = test_df.merge(
    historical_pit,
    on=['Compound', 'Team'],
    how='left'
)

global_pit_mean = (
    historical_pit[
        'historical_pit_lap'
    ].mean()
)

train_df['historical_pit_lap'] = (
    train_df['historical_pit_lap']
    .fillna(global_pit_mean)
)

test_df['historical_pit_lap'] = (
    test_df['historical_pit_lap']
    .fillna(global_pit_mean)
)

train_df['pit_window_delta'] = (
    train_df['LapNumber']
    -
    train_df['historical_pit_lap']
)

test_df['pit_window_delta'] = (
    test_df['LapNumber']
    -
    test_df['historical_pit_lap']
)


# =========================================================
# FEATURES
# =========================================================

categorical_cols = [
    'Compound',
    'Team'
]

continuous_cols = [

    # race state
    'LapTime_Seconds',
    'Position',
    'LapNumber',
    'TyreLife',
    'TrackStatus',

    # pace evolution
    'delta_laptime',
    'CumulativeTimeStint',
    'race_progress_fraction',

    # tire degradation
    'relative_tire_age',
    'tire_compound_age_delta',
    'tire_performance_decay',

    # rolling pace metrics
    'rolling_pace_mean_5',
    'pace_std_5',
    'pace_degradation_slope',

    # pit strategy features
    'historical_pit_lap',
    'pit_window_delta',

    # telemetry
    'Speed_mean',
    'Speed_max',
    'Speed_std',

    'RPM_mean',
    'RPM_max',

    'Throttle_mean',
    'Throttle_std',

    'Brake_mean',
    'Brake_sum',

    'DRS_mean',
    'DRS_sum',

    # traffic
    'DistanceToDriverAhead_mean',
    'DistanceToDriverAhead_min',

    # gearbox
    'nGear_mean',
    'nGear_max',

    # engineered strategy indicators
    'traffic_pressure',
    'drs_dependency',
    'thermal_stress_proxy',
    'high_speed_stress',
    'brake_aggression',
    'pace_vs_ahead'
]

# =========================================================
# ENCODE CATEGORICALS
# =========================================================

combined_df = pd.concat(
    [train_df, test_df],
    axis=0
)

combined_df = pd.get_dummies(
    combined_df,
    columns=categorical_cols,
    drop_first=False
)

train_df = combined_df[
    combined_df['Year'] < 2025
].copy()

test_df = combined_df[
    combined_df['Year'] == 2025
].copy()

encoded_cat_cols = [

    col for col in combined_df.columns

    if (
        col.startswith("Compound_")
        or
        col.startswith("Team_")
    )
]

feature_cols = (
    continuous_cols +
    encoded_cat_cols
)


# =========================================================
# NORMALIZATION
# =========================================================

scaler = StandardScaler()

train_df[continuous_cols] = (
    scaler.fit_transform(
        train_df[continuous_cols]
    )
)

test_df[continuous_cols] = (
    scaler.transform(
        test_df[continuous_cols]
    )
)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)


# =========================================================
# SEQUENCE CREATION
# =========================================================

def create_driver_sequences(
    data,
    feature_columns,
    target_column='HasPitStop',
    seq_length=20
):

    X = []
    y = []

    grouped = data.groupby(
        ['RaceID', 'DriverNumber']
    )

    for _, group in grouped:

        group = group.sort_values(
            'LapNumber'
        )

        feat = group[
            feature_columns
        ].values

        targ = group[
            target_column
        ].values

        if len(group) < seq_length:
            continue

        for i in range(
            len(group) - seq_length + 1
        ):

            X.append(
                feat[
                    i:i+seq_length
                ]
            )

            y.append(
                targ[
                    i+seq_length-1
                ]
            )

    return (

        np.array(
            X,
            dtype=np.float32
        ),

        np.array(
            y,
            dtype=np.float32
        )
    )


X_train, y_train = create_driver_sequences(
    train_df,
    feature_cols,
    seq_length=SEQ_LENGTH
)

X_test, y_test = create_driver_sequences(
    test_df,
    feature_cols,
    seq_length=SEQ_LENGTH
)

print(X_train.shape)
print(X_test.shape)


# =========================================================
# PYTORCH DATASET
# =========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# =========================================================
# MODEL
# =========================================================

class F1PitStopPredictor(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.bn1 = nn.BatchNorm1d(
            512
        )

        self.dropout1 = nn.Dropout(
            0.2
        )

        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.bn2 = nn.BatchNorm1d(
            256
        )

        self.dropout2 = nn.Dropout(
            0.3
        )

        self.lstm3 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.bn3 = nn.BatchNorm1d(
            128
        )

        self.dropout3 = nn.Dropout(
            0.3
        )

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.dropout4 = nn.Dropout(
            0.2
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.output = nn.Linear(
            32,
            1
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        x
    ):

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


model = F1PitStopPredictor(
    input_size=X_train.shape[2]
).to(DEVICE)


# =========================================================
# LOSS
# =========================================================

positive_weight = (
    len(y_train[y_train == 0])
    /
    len(y_train[y_train == 1])
)

positive_weight_tensor = torch.tensor(
    [positive_weight], 
    dtype=torch.float32
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight_tensor)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=4
)


# =========================================================
# TRAINING LOOP
# =========================================================

best_auc_pr = 0

patience = 8

counter = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)

        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    all_probs = []

    all_targets = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

            probs = (
                torch.sigmoid(outputs)
                .squeeze()
                .cpu()
                .numpy()
            )

            all_probs.extend(probs)

            all_targets.extend(
                batch_y.numpy().flatten()
            )

    auc_pr = average_precision_score(
        all_targets,
        all_probs
    )

    roc_auc = roc_auc_score(
        all_targets,
        all_probs
    )

    precision = precision_score(
        all_targets,
        (np.array(all_probs) >= 0.5).astype(int)
    )

    recall = recall_score(
        all_targets,
        (np.array(all_probs) >= 0.5).astype(int)
    )

    f1 = f1_score(
        all_targets,
        (np.array(all_probs) >= 0.5).astype(int)
    )

    scheduler.step(auc_pr)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"ROC-AUC: {roc_auc:.4f} | "
        f"AUC-PR: {auc_pr:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-Score: {f1:.4f}"
    )

    if auc_pr > best_auc_pr:

        best_auc_pr = auc_pr

        counter = 0

        torch.save(
            model.state_dict(),
            "best_f1_model.pt"
        )

# =========================================================
# LOAD BEST MODEL
# =========================================================

# model.load_state_dict(
#     torch.load(
#         "best_f1_model.pt"
#     )
# )


# =========================================================
# FINAL EVALUATION
# =========================================================

model.eval()

all_probs = []

all_targets = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(DEVICE)

        outputs = model(batch_x)

        probs = (
            torch.sigmoid(outputs)
            .squeeze()
            .cpu()
            .numpy()
        )

        all_probs.extend(probs)

        all_targets.extend(
            batch_y.numpy().flatten()
        )

y_pred_probs = np.array(
    all_probs
)

y_test_np = np.array(
    all_targets
)


# =========================================================
# BEST THRESHOLD
# =========================================================

precision_vals, recall_vals, thresholds = (
    precision_recall_curve(
        y_test_np,
        y_pred_probs
    )
)

f1_scores = (

    2
    *
    precision_vals
    *
    recall_vals

    /

    (
        precision_vals
        +
        recall_vals
        +
        1e-8
    )
)

best_idx = np.argmax(
    f1_scores[:-1]
)

best_threshold = thresholds[
    best_idx
]

print(
    f"Best Threshold: "
    f"{best_threshold:.4f}"
)


# =========================================================
# FINAL PREDICTIONS
# =========================================================

y_pred = (
    y_pred_probs >= best_threshold
).astype(int)

precision = precision_score(
    y_test_np,
    y_pred
)

recall = recall_score(
    y_test_np,
    y_pred
)

f1 = f1_score(
    y_test_np,
    y_pred
)

roc_auc = roc_auc_score(
    y_test_np,
    y_pred_probs
)

auc_pr = average_precision_score(
    y_test_np,
    y_pred_probs
)

balanced_acc = balanced_accuracy_score(
    y_test_np,
    y_pred
)

tn, fp, fn, tp = confusion_matrix(
    y_test_np,
    y_pred
).ravel()

specificity = tn / (tn + fp)


# =========================================================
# RESULTS
# =========================================================

print("\n--- Evaluation Metrics ---")

print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"AUC-PR:            {auc_pr:.4f}")

print("\n--- Confusion Matrix ---")

print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "f1_bilstm_model.pt"
)

print("\nModel Saved")