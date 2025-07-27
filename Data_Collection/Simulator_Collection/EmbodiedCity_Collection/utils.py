"""
放一些辅助函数
"""

import os
import time
from datetime import datetime

__all__ = [
    "ensure_dir",
    "timestamp",
    "log",
]

def ensure_dir(path: str) -> None:
    """如果目录不存在，创建该目录"""
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """返回一个紧凑的'年月日_时分秒'格式的字符串，用于创建带有时间戳的文件名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str, level: str = "INFO") -> None:
    """简单日志记录器"""
    print(f"[{level}] {message}")