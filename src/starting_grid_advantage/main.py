"""Entry point: H2 hypothesis pipeline for starting-grid advantage."""

import os
import sys
import warnings
import logging

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from .utils import enforce_determinism
from .data_pipeline import execute_data_pipeline
from .features import construct_feature_space
from .eda_analyzer import perform_eda_analysis
from .classification import execute_classification_pipeline
from .regression import execute_regression_pipeline


def main():
    print("\n" + "╔" + "═"*60 + "╗")
    print("║ BẮT ĐẦU HỆ THỐNG PHÂN TÍCH F1 - H2 HYPOTHESIS".ljust(61) + "║")
    print("╚" + "═"*60 + "╝\n")

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
        print(" Vui lòng kiểm tra thư mục 'outputs/starting_grid_advantage/' để lấy biểu đồ và mô hình.")
        print("="*62 + "\n")

    except FileNotFoundError as fnf_err:
        print(f"\n[LỖI DỮ LIỆU]: {fnf_err}")
        print("Vui lòng đảm bảo các file CSV đã được đưa vào đúng thư mục data/raw/")
    except Exception as e:
        print(f"\n[LỖI HỆ THỐNG KHÔNG XÁC ĐỊNH]: {e}")
        raise e


if __name__ == "__main__":
    main()
