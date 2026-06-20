import os
import random
import numpy as np
import tensorflow as tf

from .config import RANDOM_SEED


def enforce_determinism():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
        print("Đã kích hoạt chế độ Đơn luồng (Op Determinism) cho TensorFlow.")
    except AttributeError:
        print("Phiên bản TensorFlow hiện tại không hỗ trợ enable_op_determinism(). Hãy cập nhật TF >= 2.8 nếu kết quả bị lệch.")

    print(f"Đã khóa toàn bộ tính ngẫu nhiên hệ thống với Seed = {RANDOM_SEED}")
