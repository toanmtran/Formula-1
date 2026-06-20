"""
src/data_pipeline.py
Đường ống xử lý dữ liệu (Data Pipeline).
Nhiệm vụ: Nạp dữ liệu thô từ CSV, ép kiểu, làm sạch nhiễu và gộp (merge) các bảng quan hệ.
"""

import pandas as pd
import os
#import logging
from src.config import RAW_DATA_DIR

def load_and_clean_results():
    """
    Nạp và làm sạch bảng cốt lõi: results.csv
    Xử lý đặc thù của dữ liệu F1 (Ký tự rỗng được ghi là '\\N')
    """
    #logger = logging.getLogger("F1_System")
    print("Đang nạp file results.csv...")
    
    file_path = os.path.join(RAW_DATA_DIR, 'results.csv')
    if not os.path.exists(file_path):
        print(f"Không tìm thấy {file_path}. Vui lòng kiểm tra lại thư mục data/raw/")
        raise FileNotFoundError(f"Missing file: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Ép kiểu dữ liệu (Numeric Casting)
    # Tham số errors='coerce' sẽ tự động biến các chuỗi lỗi như '\\N' thành NaN (Not a Number)
    cols_to_numeric = ['grid', 'positionOrder', 'points']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # LÀM SẠCH (CLEANING):
    # 1. Loại bỏ các dòng bị khuyết vị trí xuất phát hoặc vị trí về đích
    original_len = len(df)
    df = df.dropna(subset=['grid', 'positionOrder', 'points'])
    
    # 2. Loại bỏ các tay đua xuất phát từ đường pit (grid = 0)
    # Vì phân tích của chúng ta tập trung vào lợi thế của vị trí xuất phát trên vạch (grid advantage)
    df = df[df['grid'] > 0]
    
    print(f"Đã làm sạch results.csv. Giữ lại {len(df)}/{original_len} bản ghi hợp lệ.")
    return df

def load_context_data():
    """Nạp các bảng dữ liệu bối cảnh (races và circuits)"""
    #logger = logging.getLogger("F1_System")
    print("Đang nạp file races.csv và circuits.csv...")
    
    try:
        races = pd.read_csv(os.path.join(RAW_DATA_DIR, 'races.csv'))
        circuits = pd.read_csv(os.path.join(RAW_DATA_DIR, 'circuits.csv'))
    except FileNotFoundError as e:
        print("Thiếu file races.csv hoặc circuits.csv trong thư mục data/raw/")
        raise e
        
    return races, circuits

def execute_data_pipeline():
    """
    Hàm điều phối toàn bộ đường ống dữ liệu (Data Pipeline).
    Sẽ được gọi từ main.py.
    """
    #logger = logging.getLogger("F1_System")
    print("=== BẮT ĐẦU ĐƯỜNG ỐNG XỬ LÝ DỮ LIỆU (DATA PIPELINE) ===")
    
    # 1. Nạp và làm sạch
    df_results = load_and_clean_results()
    df_races, df_circuits = load_context_data()
    
    # 2. Hợp nhất dữ liệu (Merge DataFrames)
    print("Đang tiến hành kết nối (Merge) các thực thể dữ liệu...")
    
    # Nối với races để lấy thông tin: Năm đua (year) và Mã trường đua (circuitId)
    df_merged = df_results.merge(
        df_races[['raceId', 'year', 'circuitId']], 
        on='raceId', 
        how='inner'
    )
    
    # Nối với circuits để lấy thông tin: Tên trường đua và Mã tham chiếu (circuitRef)
    df_final = df_merged.merge(
        df_circuits[['circuitId', 'name', 'circuitRef']], 
        on='circuitId', 
        how='inner'
    )
    
    print(f"Hoàn thành Pipeline! Kích thước dữ liệu sau khi merge: {df_final.shape}")
    
    return df_final

if __name__ == "__main__":
    #from src.utils import setup_logger
    #setup_logger()
    df_test = execute_data_pipeline()
    print(df_test.head())