import pandas as pd
import os
from .config import RAW_DATA_DIR


def load_and_clean_results():
    print("Đang nạp file results.csv...")

    file_path = os.path.join(RAW_DATA_DIR, 'results.csv')
    if not os.path.exists(file_path):
        print(f"Không tìm thấy {file_path}. Vui lòng kiểm tra lại thư mục data/raw/")
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    cols_to_numeric = ['grid', 'positionOrder', 'points']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    original_len = len(df)
    df = df.dropna(subset=['grid', 'positionOrder', 'points'])
    # grid == 0 means a pit-lane start; exclude because it isn't a "grid" advantage
    df = df[df['grid'] > 0]

    print(f"Đã làm sạch results.csv. Giữ lại {len(df)}/{original_len} bản ghi hợp lệ.")
    return df


def load_context_data():
    print("Đang nạp file races.csv và circuits.csv...")

    try:
        races    = pd.read_csv(os.path.join(RAW_DATA_DIR, 'races.csv'))
        circuits = pd.read_csv(os.path.join(RAW_DATA_DIR, 'circuits.csv'))
    except FileNotFoundError as e:
        print("Thiếu file races.csv hoặc circuits.csv trong thư mục data/raw/")
        raise e

    return races, circuits


def execute_data_pipeline():
    print("=== BẮT ĐẦU ĐƯỜNG ỐNG XỬ LÝ DỮ LIỆU (DATA PIPELINE) ===")

    df_results = load_and_clean_results()
    df_races, df_circuits = load_context_data()

    print("Đang tiến hành kết nối (Merge) các thực thể dữ liệu...")

    df_merged = df_results.merge(
        df_races[['raceId', 'year', 'circuitId']],
        on='raceId',
        how='inner',
    )

    df_final = df_merged.merge(
        df_circuits[['circuitId', 'name', 'circuitRef']],
        on='circuitId',
        how='inner',
    )

    print(f"Hoàn thành Pipeline! Kích thước dữ liệu sau khi merge: {df_final.shape}")
    return df_final


if __name__ == "__main__":
    df_test = execute_data_pipeline()
    print(df_test.head())
