#!/usr/bin/env python3
"""公共基础件：日志格式、装饰器与设备解析。"""

import time
import functools

import torch

col_dic = {
    'd': 30, 'r': 31, 'g': 32, 'y': 33,
    'b': 34, 'p': 35, 'c': 36, 'w': 37
}

def string_decorator(basic:str, / , col:str='w', b:bool=False) -> str:
    """ANSI 颜色包装：b=True 时加粗。"""
    return f"\033[{'1;' if b else '0;'}{col_dic[col]}m{basic}\033[0m"


import logging
logging.basicConfig(level=logging.INFO,
                    format=string_decorator("[%(asctime)s %(levelname)s]", 'c') + " %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


def time_log(func):
    """打印函数耗时的装饰器。"""
    @functools.wraps(func)
    def wrapper(*args, **kw):
        _time_str = time.time()
        res = func(*args, **kw)
        _time_stp = time.time()
        print(string_decorator(f'[Function {func.__name__}() Finished] cost time: {_time_stp - _time_str:6.3f} s.', 'p', True))
        return res
    return wrapper


def resolve_torch_device(gpu_id=None) -> torch.device:
    """按配置的 GPU id 返回可用设备；id 非法或无 CUDA 时回退 CPU。"""
    try:
        gpu_id = int(gpu_id)
    except (TypeError, ValueError):
        gpu_id = -1
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")
