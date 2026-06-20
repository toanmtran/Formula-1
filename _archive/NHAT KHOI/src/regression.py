"""
src/regression.py
Mô hình Hồi quy (Regression).
"""

import os
import joblib
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from src.config import RANDOM_SEED, TEST_SIZE, DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE, MODEL_DIR

def prove_street_interaction(df):
    print("\n  [1] Chứng minh tương tác (OLS Modeling):")
    features = ['grid', 'is_street', 'grid_x_street']
    X = sm.add_constant(df[features])
    y = df['positionOrder']
    
    model = sm.OLS(y, X).fit()
    grid_coef = model.params['grid']
    inter_coef = model.params['grid_x_street']
    p_value = model.pvalues['grid_x_street']
    
    print(f"      -> Hệ số Grid          : {grid_coef:.4f}")
    print(f"      -> Hệ số Grid x Street : {inter_coef:.4f}")
    print(f"      -> P-value             : {p_value:.6f}")
    if p_value < 0.05:
        print("      [+] KẾT LUẬN: P-value < 0.05, tương tác dương hợp lệ. "
              "Đường phố phạt lỗi lùi vị trí nặng hơn!")

def run_linear_explainable_model(df):
    print("\n  [2] Giải thích trọng số (Linear Regression):")
    df_test = df[df['positionOrder'] <= 15].copy()
    df_test['random_noise_1'] = np.random.randn(len(df_test))
    df_test['random_noise_2'] = np.random.randn(len(df_test))
    
    features = ['grid', 'is_street', 'car_strength_index', 'random_noise_1', 'random_noise_2']
    X = df_test[features]
    y = df_test['positionOrder']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    model_path = os.path.join(MODEL_DIR, 'linear_regression.pkl')
    if os.path.exists(model_path):
        print("      -> Nạp mô hình có sẵn...")
        model = joblib.load(model_path)
    else:
        print("      -> Huấn luyện mô hình mới...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"      -> Sai số MAE: {mae:.2f}")

def run_lstm_predictive_model(df):
    print("\n  [3] Mạng nơ-ron dự đoán (LSTM):")
    df_dl = df[df['positionOrder'] <= 15].copy()
    df_dl['random_noise_1'] = np.random.randn(len(df_dl))
    
    features = ['grid', 'is_street', 'car_strength_index', 'random_noise_1']
    X = df_dl[features].values
    y = df_dl['positionOrder'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    # --- ĐỔI ĐUÔI TỪ .h5 SANG .keras TẠI ĐÂY ---
    lstm_path = os.path.join(MODEL_DIR, 'lstm_f1_model.keras')
    scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')
    
    if os.path.exists(lstm_path) and os.path.exists(scaler_path):
        print("      -> Đã tìm thấy Scaler và Mô hình LSTM. Đang nạp...")
        scaler = joblib.load(scaler_path)
        
        # Load mô hình bằng chuẩn mới cực gọn
        model = load_model(lstm_path)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    else:
        print("      -> Đang chuẩn hóa và huấn luyện LSTM...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        tf.random.set_seed(RANDOM_SEED)
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(1, len(features))),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=DL_LEARNING_RATE), loss='mse', metrics=['mae'])
        model.fit(X_train_3d, y_train, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, validation_split=0.2, verbose=0)
        
        model.save(lstm_path)
        joblib.dump(scaler, scaler_path)
        print(f"      -> Đã lưu mô hình LSTM (.keras) và Scaler (.pkl).")

    # Đánh giá chung
    _, mae = model.evaluate(X_test_3d, y_test, verbose=0)
    print(f"      -> Sai số Test MAE: {mae:.2f} vị trí")

def execute_regression_pipeline(df):
    prove_street_interaction(df)
    run_linear_explainable_model(df)
    run_lstm_predictive_model(df)