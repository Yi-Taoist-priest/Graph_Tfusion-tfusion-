"""pytest 根配置：把 src-layout 的库代码与脚本目录加入 sys.path。

项目把库代码放在 ``src/``、入口脚本放在 ``scripts/``，但模块名保持
``data_pipeline`` / ``detect`` / ``model`` / ``runner`` / ``common`` /
``feature_scaling`` / ``run_graph_tfusion`` / ``convert_cicids_pcap`` 不变，
因此测试里直接 ``import data_pipeline`` 这类写法仍然有效。

生产入口（``scripts/*.py``）有各自的 bootstrap，
不依赖本文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _relative in ("src", "scripts", "scripts/pcap"):
    _candidate = ROOT / _relative
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
