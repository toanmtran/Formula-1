"""
src/utils.py
Chứa các hàm tiện ích dùng chung cho toàn bộ hệ thống:
1. Thiết lập Logging chuẩn hóa.
2. Khóa tính ngẫu nhiên (Determinism) để đảm bảo kết quả có thể tái lập (Reproducibility).
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf

# Import giá trị Hạt giống (Seed) từ file cấu hình
from src.config import RANDOM_SEED

# def setup_logger():
#     """
#     Khởi tạo và cấu hình hệ thống ghi nhật ký (Logging).
#     Giúp in ra màn hình console các thông báo với định dạng chuyên nghiệp:
#     [Thời gian] - [Cấp độ thông báo] - [Nội dung]
#     """
#     # Xóa các cấu hình log cũ nếu có để tránh in trùng lặp
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
        
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - [%(levelname)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     # Tạo một đối tượng logger riêng cho dự án
#     logger = logging.getLogger("F1_System")
#     return logger

def enforce_determinism():
    """
    Hàm cực kỳ quan trọng đối với Machine/Deep Learning:
    Khóa toàn bộ các nguồn phát sinh ngẫu nhiên ở cấp độ luồng phần cứng.
    Đảm bảo mô hình chạy 100 lần đều ra chung 1 kết quả sai số MAE.
    """
    #logger = logging.getLogger("F1_System")
    
    # 1. Khóa bộ băm (hash) của Python
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    
    # 2. Khóa module random cốt lõi của Python
    random.seed(RANDOM_SEED)
    
    # 3. Khóa các hàm sinh số ngẫu nhiên của Numpy (dùng cho mảng)
    np.random.seed(RANDOM_SEED)
    
    # 4. Khóa hạt giống ngẫu nhiên của mạng nơ-ron TensorFlow
    tf.random.set_seed(RANDOM_SEED)
    
    # 5. Ép buộc phần cứng (CPU/GPU) chạy đơn luồng cho các toán tử của TensorFlow
    # Đây là "liều thuốc đặc trị" giải quyết tận gốc việc kết quả Deep Learning bị nhảy số
    try:
        tf.config.experimental.enable_op_determinism()
        print(f"Đã kích hoạt chế độ Đơn luồng (Op Determinism) cho TensorFlow.")
    except AttributeError:
        # Đề phòng trường hợp máy bạn cài bản TensorFlow quá cũ (< 2.8) chưa có hàm này
        print("Phiên bản TensorFlow hiện tại không hỗ trợ enable_op_determinism(). Hãy cập nhật TF >= 2.8 nếu kết quả bị lệch.")

    print(f"Đã khóa toàn bộ tính ngẫu nhiên hệ thống với Seed = {RANDOM_SEED}")