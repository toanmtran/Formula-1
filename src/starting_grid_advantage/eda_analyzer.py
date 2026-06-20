import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import PLOT_DIR, METRICS_DIR


def calculate_pole_win_probability(df):
    print("Đang phân tích Q1: Xác suất thắng từ Pole...")

    pole_starts = df[df['grid'] == 1]
    pole_wins   = pole_starts[pole_starts['positionOrder'] == 1]

    if len(pole_starts) == 0:
        print("Không có dữ liệu xuất phát ở vị trí Pole.")
        return

    p_win = len(pole_wins) / len(pole_starts)

    print("\n" + "-" * 50)
    print(f"[Q1] XÁC SUẤT CHIẾN THẮNG TỪ POLE P(Win|Pole): {p_win*100:.2f}%")
    print(f"     (Thắng {len(pole_wins)} lần trên tổng số {len(pole_starts)} lần Pole)")
    print("-" * 50)


def compare_eras(df):
    print("Đang phân tích Q2: So sánh nhóm xuất phát (Từ 2010 đến nay)...")

    df_modern = df[df['year'] >= 2010]
    top5     = df_modern[df_modern['grid'].isin([1, 2, 3, 4, 5])]
    midfield = df_modern[df_modern['grid'].isin([6, 7, 8, 9, 10])]

    print("\n" + "-" * 50)
    print(f"[Q2] SO SÁNH TOP 5 vs MIDFIELD (KỶ NGUYÊN >= 2010):")
    print(f"-> NHÓM TOP 5 (P1-P5): Vị trí đích TB = {top5['positionOrder'].mean():.2f} | Điểm TB = {top5['points'].mean():.2f}")
    print(f"-> NHÓM MIDFIELD (P6-P10): Vị trí đích TB = {midfield['positionOrder'].mean():.2f} | Điểm TB = {midfield['points'].mean():.2f}")
    print("-" * 50)


def analyze_circuit_correlation(df):
    print("Đang phân tích Q3: Tương quan vị trí (Monaco vs Monza)...")

    corr_df = (
        df.groupby(['circuitRef', 'name'])[['grid', 'positionOrder']]
        .corr().iloc[0::2, -1].reset_index()
    )
    corr_df = corr_df.rename(columns={'positionOrder': 'grid_finish_correlation'}) \
                     .sort_values('grid_finish_correlation', ascending=False)

    metrics_path = os.path.join(METRICS_DIR, 'circuit_grid_correlations.csv')
    corr_df.to_csv(metrics_path, index=False)
    print(f"Đã lưu bảng tương quan các trường đua tại: {metrics_path}")

    print("\n" + "-" * 50)
    print(f"[Q3] HỆ SỐ TƯƠNG QUAN (Càng gần 1, Grid càng quan trọng):")
    try:
        monaco_r = corr_df[corr_df['circuitRef'] == 'monaco']['grid_finish_correlation'].values[0]
        monza_r  = corr_df[corr_df['circuitRef'] == 'monza']['grid_finish_correlation'].values[0]
        print(f"-> Monaco (Đường phố, khó vượt): r = {monaco_r:.4f}")
        print(f"-> Monza (Đường tốc độ, dễ vượt): r = {monza_r:.4f}")
    except IndexError:
        print("Không đủ dữ liệu cho Monaco hoặc Monza để so sánh.")
    print("-" * 50)


def plot_advantage_decay(df):
    print("Đang phân tích Q4: Vẽ biểu đồ Advantage Decay...")

    grid_stats = df[df['grid'] <= 20].groupby('grid').agg(
        total_starts=('resultId', 'count'),
        podiums=('positionOrder', lambda x: (x <= 3).sum()),
        mean_finish=('positionOrder', 'mean'),
    ).reset_index()

    grid_stats['p_podium'] = grid_stats['podiums'] / grid_stats['total_starts']

    plt.figure(figsize=(12, 6))
    sns.barplot(x='grid', y='p_podium', data=grid_stats, color='#1f77b4', edgecolor='black')
    plt.title('Decrease in Podium Probability Based on Starting Position',
              fontsize=14, fontweight='bold')
    plt.ylabel('Probability (0 - 1)', fontsize=12)
    plt.xlabel('Grid Position', fontsize=12)

    for idx, row in grid_stats.iterrows():
        plt.text(row.name, row.p_podium + 0.01, f"{row.p_podium*100:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_podium_decay.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='grid', y='mean_finish', data=grid_stats, marker='o',
                 color='#d62728', linewidth=2.5, markersize=8)
    plt.title('Average Finishing Position by Starting Position',
              fontsize=14, fontweight='bold')
    plt.ylabel('Finishing rank (the lower the better)', fontsize=12)
    plt.xlabel('Grid Position', fontsize=12)
    # Lower finishing rank = better position, so invert Y
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_mean_finish_decay.png'), dpi=300)
    plt.close()

    print(f"Đã xuất bản thành công 2 biểu đồ vào thư mục: {PLOT_DIR}")


def perform_eda_analysis(df):
    print("=== BẮT ĐẦU PHÂN TÍCH KHÁM PHÁ (EDA) ===")

    calculate_pole_win_probability(df)
    compare_eras(df)
    analyze_circuit_correlation(df)
    plot_advantage_decay(df)

    print("Hoàn thành module EDA Analyzer.")


if __name__ == "__main__":
    from .data_pipeline import execute_data_pipeline
    from .features import construct_feature_space

    df_raw = execute_data_pipeline()
    df_ready = construct_feature_space(df_raw)
    perform_eda_analysis(df_ready)
