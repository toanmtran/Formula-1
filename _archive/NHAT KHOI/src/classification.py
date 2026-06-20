"""
src/classification.py
Mô hình Phân loại (Classification).
Nhiệm vụ: Dự đoán khả năng lọt vào Podium (Top 3) của tay đua.
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_SEED, TEST_SIZE, RF_N_ESTIMATORS, RF_MAX_DEPTH, MODEL_DIR

def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("  [+] Logistic Regression:")
    model_path = os.path.join(MODEL_DIR, 'lr_podium_classifier.pkl')
    
    if os.path.exists(model_path):
        print("      -> Đã tìm thấy file trọng số. Đang nạp mô hình...")
        model = joblib.load(model_path)
    else:
        print("      -> Đang huấn luyện mô hình mới...")
        model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"      -> Đã lưu mô hình tại: {model_path}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("      " + "-"*45)
    print(f"      | KẾT QUẢ ĐÁNH GIÁ (Accuracy): {acc:.4f} |")
    print("      " + "-"*45)
    return model

def run_random_forest(X_train, X_test, y_train, y_test):
    print("\n  [+] Random Forest Classifier:")
    model_path = os.path.join(MODEL_DIR, 'rf_podium_classifier.pkl')
    
    if os.path.exists(model_path):
        print("      -> Đã tìm thấy file trọng số. Đang nạp mô hình...")
        model = joblib.load(model_path)
    else:
        print("      -> Đang huấn luyện mô hình mới...")
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED,
            class_weight='balanced' 
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"      -> Đã lưu mô hình tại: {model_path}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("      " + "-"*45)
    print(f"      | KẾT QUẢ ĐÁNH GIÁ (Accuracy): {acc:.4f} |")
    print("      " + "-"*45)
    return model

def execute_classification_pipeline(df):
    print("--- KHỞI TẠO DỮ LIỆU PHÂN LOẠI ---")
    features = ['grid', 'is_street', 'car_strength_index']
    X = df[features]
    y = df['is_podium']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)