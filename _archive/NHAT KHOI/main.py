"""
main.py
ĐIỂM KHỞI CHẠY HỆ THỐNG (ENTRY POINT)
Dự án: Hệ thống Phân tích & Dự đoán Lợi thế xuất phát F1 (H2 Hypothesis)
"""

import os
import warnings
import logging

# ==============================================================================
# TẮT TOÀN BỘ CẢNH BÁO TỪ TRƯỚC KHI IMPORT TENSORFLOW VÀ CÁC MODULE KHÁC
# ==============================================================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt cảnh báo oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Tắt log C++ (GPU warning) của TensorFlow
warnings.filterwarnings('ignore')          # Tắt cảnh báo Deprecation của Python

# Tắt log cấp độ Python của thư viện nền absl và tensorflow
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
# ==============================================================================

# Bây giờ mới import các module cốt lõi từ package src/
from src.utils import enforce_determinism
from src.data_pipeline import execute_data_pipeline
from src.features import construct_feature_space
from src.eda_analyzer import perform_eda_analysis
from src.classification import execute_classification_pipeline
from src.regression import execute_regression_pipeline

def main():
    print("\n" + "╔" + "═"*60 + "╗")
    print("║ BẮT ĐẦU HỆ THỐNG PHÂN TÍCH F1 - H2 HYPOTHESIS".ljust(61) + "║")
    print("╚" + "═"*60 + "╝\n")
    
    # Khóa hoàn toàn tính ngẫu nhiên của phần cứng
    enforce_determinism()
    
    try:
        print("\n>>> BƯỚC 1: XỬ LÝ DỮ LIỆU THÔ (DATA PIPELINE)")
        df_raw = execute_data_pipeline()
        
        print("\n>>> BƯỚC 2: TẠO ĐẶC TRƯNG TOÁN HỌC (FEATURE ENGINEERING)")
        df_ready = construct_feature_space(df_raw)
        
        print("\n>>> BƯỚC 3: PHÂN TÍCH KHÁM PHÁ VÀ XUẤT BIỂU ĐỒ (EDA)")
        perform_eda_analysis(df_ready)
        
        print("\n>>> BƯỚC 4: MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)")
        execute_classification_pipeline(df_ready)
        
        print("\n>>> BƯỚC 5: MÔ HÌNH HỒI QUY VÀ DEEP LEARNING (REGRESSION)")
        execute_regression_pipeline(df_ready)
        
        print("\n" + "="*62)
        print(" THÀNH CÔNG! HỆ THỐNG ĐÃ HOÀN TẤT TOÀN BỘ CHU TRÌNH.")
        print(" Vui lòng kiểm tra thư mục 'output/' để lấy biểu đồ và mô hình.")
        print("="*62 + "\n")
        
    except FileNotFoundError as fnf_err:
        print(f"\n[LỖI DỮ LIỆU]: {fnf_err}")
        print("Vui lòng đảm bảo các file CSV đã được đưa vào đúng thư mục data/raw/")
    except Exception as e:
        print(f"\n[LỖI HỆ THỐNG KHÔNG XÁC ĐỊNH]: {e}")
        raise e

if __name__ == "__main__":
    main()