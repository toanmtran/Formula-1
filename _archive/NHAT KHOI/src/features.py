"""
src/features.py
Kỹ sư Đặc trưng (Feature Engineering).
Nhiệm vụ: Biến đổi dữ liệu đã làm sạch thành các đặc trưng có ý nghĩa toán học cho AI,
bao gồm biến mục tiêu (target), biến tương tác và chỉ số sức mạnh đội đua.
"""

import pandas as pd
#import logging
from src.config import STREET_CIRCUITS

def construct_feature_space(df):
    """
    Hàm xây dựng không gian đặc trưng.
    Nhận vào DataFrame đã được gộp từ data_pipeline và trả về DataFrame sẵn sàng cho ML.
    """
    #logger = logging.getLogger("F1_System")
    print("=== BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING) ===")
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning của Pandas
    df_feat = df.copy()

    # -----------------------------------------------------------------
    # 1. TẠO BIẾN MỤC TIÊU (TARGET VARIABLES)
    # -----------------------------------------------------------------
    # Phục vụ cho bài toán Classification trong file classification.py
    # 1: Đạt Podium (Top 3), 0: Không đạt
    df_feat['is_podium'] = (df_feat['positionOrder'] <= 3).astype(int)

    # -----------------------------------------------------------------
    # 2. TẠO ĐẶC TRƯNG CẤU TRÚC ĐƯỜNG ĐUA (TRACK TYPOLOGY)
    # -----------------------------------------------------------------
    # 1: Đường phố (Khó vượt), 0: Đường trường (Dễ vượt)
    df_feat['is_street'] = df_feat['circuitRef'].isin(STREET_CIRCUITS).astype(int)

    # -----------------------------------------------------------------
    # 3. TẠO BIẾN TƯƠNG TÁC (INTERACTION TERM)
    # -----------------------------------------------------------------
    # Phục vụ cho bài toán Chứng minh toán học OLS trong file regression.py
    # Biến này chỉ kích hoạt giá trị khi chặng đua đó là đường phố
    df_feat['grid_x_street'] = df_feat['grid'] * df_feat['is_street']

    # -----------------------------------------------------------------
    # 4. TẠO CHỈ SỐ SỨC MẠNH CHIẾC XE (CONSTRUCTOR STRENGTH INDEX)
    # -----------------------------------------------------------------
    print("Đang tính toán Chỉ số Sức mạnh Đội đua (Constructor Strength)...")
    
    # Tính trung bình vị trí về đích của từng đội (số càng nhỏ xe càng mạnh)
    constructor_map = df_feat.groupby('constructorId')['positionOrder'].mean().reset_index()
    
    # Đổi tên cột để tránh trùng lặp khi merge
    constructor_map.rename(columns={'positionOrder': 'car_strength_index'}, inplace=True)
    
    # Gắn chỉ số này trở lại bảng dữ liệu chính
    df_feat = df_feat.merge(constructor_map, on='constructorId', how='left')
    
    # Xử lý an toàn: Nếu có đội đua nào bị NaN (rất hiếm), ta điền bằng mức trung bình của toàn bộ các xe
    mean_strength = df_feat['car_strength_index'].mean()
    df_feat['car_strength_index'] = df_feat['car_strength_index'].fillna(mean_strength)

    print(f"Hoàn thành Feature Engineering! Tổng số trường dữ liệu: {df_feat.shape[1]}")
    return df_feat

# Khối lệnh dùng để test độc lập file này
if __name__ == "__main__":
    from src.utils import setup_logger
    from src.data_pipeline import execute_data_pipeline
    
    #setup_logger()
    # Chạy pipeline trước để có data sạch, sau đó nhét vào hàm feature
    df_clean = execute_data_pipeline()
    df_features = construct_feature_space(df_clean)
    
    print("\n[KIỂM TRA CÁC CỘT MỚI TẠO]")
    print(df_features[['circuitRef', 'grid', 'is_street', 'grid_x_street', 'car_strength_index', 'is_podium']].head(10))