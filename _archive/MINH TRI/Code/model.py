import torch
import torch.nn as nn


class F1PitStopPredictor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden1=256,
        hidden2=128,
        hidden3=64,
        dropout=0.3
    ):

        super(F1PitStopPredictor, self).__init__()

        # =====================================================
        # BiLSTM 1
        # =====================================================

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.bn1 = nn.BatchNorm1d(
            hidden1 * 2
        )

        self.dropout1 = nn.Dropout(
            0.2
        )

        # =====================================================
        # BiLSTM 2
        # =====================================================

        self.lstm2 = nn.LSTM(
            input_size=hidden1 * 2,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.bn2 = nn.BatchNorm1d(
            hidden2 * 2
        )

        self.dropout2 = nn.Dropout(
            0.3
        )

        # =====================================================
        # BiLSTM 3
        # =====================================================

        self.lstm3 = nn.LSTM(
            input_size=hidden2 * 2,
            hidden_size=hidden3,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.bn3 = nn.BatchNorm1d(
            hidden3 * 2
        )

        self.dropout3 = nn.Dropout(
            0.3
        )

        # =====================================================
        # Dense Layers
        # =====================================================

        self.fc1 = nn.Linear(
            hidden3 * 2,
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # =====================================================
        # LSTM 1
        # =====================================================

        x, _ = self.lstm1(x)

        x = self.dropout1(x)

        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)

        # =====================================================
        # LSTM 2
        # =====================================================

        x, _ = self.lstm2(x)

        x = self.dropout2(x)

        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)

        # =====================================================
        # LSTM 3
        # =====================================================

        x, _ = self.lstm3(x)

        x = x[:, -1, :]

        x = self.dropout3(x)

        x = self.bn3(x)

        # =====================================================
        # Dense
        # =====================================================

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout4(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x