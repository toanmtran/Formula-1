"""
src/config.py
Quản lý cấu hình toàn cục, đường dẫn hệ thống và siêu tham số cho F1 Predictive System.
"""

import os

# =====================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (PATHS)
# =====================================================================
# Lấy đường dẫn tuyệt đối của thư mục gốc dự án (F1_Advantage_Project)
# Điều này giúp code chạy mượt trên cả Windows, Mac, Linux mà không bị lỗi đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Định nghĩa các thư mục dữ liệu
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')             # Dữ liệu gốc chưa xử lý
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') # Dữ liệu đã làm sạch

# Định nghĩa các thư mục đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')             # Lưu biểu đồ EDA
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')        # Lưu bảng số liệu/tương quan
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')           # Lưu trọng số AI (.pkl, .h5)


# =====================================================================
# 2. HẰNG SỐ & TÍNH NGẪU NHIÊN (CONSTANTS)
# =====================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Danh sách các mã đường đua được phân loại là "Đường phố" (Street Circuits)
# Đây là Domain Knowledge dùng cho Interaction Modeling
STREET_CIRCUITS = [
    'monaco', 
    'baku', 
    'marina_bay', 
    'jeddah', 
    'vegas', 
    'miami', 
    'albert_park'
]


# =====================================================================
# 3. SIÊU THAM SỐ MÔ HÌNH (HYPERPARAMETERS)
# =====================================================================
# Machine Learning (Random Forest)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8

# Deep Learning (LSTM / Advanced Regression)
DL_EPOCHS = 15
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.01


# =====================================================================
# 4. KHỞI TẠO HỆ THỐNG THƯ MỤC TỰ ĐỘNG
# =====================================================================
# Đoạn code này đảm bảo khi hệ thống chạy lần đầu, 
# nó sẽ tự động tạo ra các thư mục trống nếu chúng chưa tồn tại.
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOT_DIR, METRICS_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)