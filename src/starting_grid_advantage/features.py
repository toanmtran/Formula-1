import pandas as pd
from .config import STREET_CIRCUITS


def construct_feature_space(df):
    print("=== BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING) ===")

    df_feat = df.copy()

    df_feat['is_podium']     = (df_feat['positionOrder'] <= 3).astype(int)
    df_feat['is_street']     = df_feat['circuitRef'].isin(STREET_CIRCUITS).astype(int)
    df_feat['grid_x_street'] = df_feat['grid'] * df_feat['is_street']

    print("Đang tính toán Chỉ số Sức mạnh Đội đua (Constructor Strength)...")

    # Average finishing position per constructor — lower means a faster car
    constructor_map = df_feat.groupby('constructorId')['positionOrder'].mean().reset_index()
    constructor_map.rename(columns={'positionOrder': 'car_strength_index'}, inplace=True)
    df_feat = df_feat.merge(constructor_map, on='constructorId', how='left')

    mean_strength = df_feat['car_strength_index'].mean()
    df_feat['car_strength_index'] = df_feat['car_strength_index'].fillna(mean_strength)

    print(f"Hoàn thành Feature Engineering! Tổng số trường dữ liệu: {df_feat.shape[1]}")
    return df_feat


if __name__ == "__main__":
    from .data_pipeline import execute_data_pipeline

    df_clean = execute_data_pipeline()
    df_features = construct_feature_space(df_clean)
    print("\n[KIỂM TRA CÁC CỘT MỚI TẠO]")
    print(df_features[
        ['circuitRef', 'grid', 'is_street', 'grid_x_street', 'car_strength_index', 'is_podium']
    ].head(10))
