#!/usr/bin/env python3
"""
run_all.py - Graph_Tfusion 批量 train+test 运行器
遍历 config/<mode>/ 下的全部数据集配置，逐个执行训练+测试
结果保存到 log/all/<mode>/<数据集名>/
Usage:
    cd Graph_Tfusion
    python scripts/tools/run_all.py                    # 预训练+微调（config/train-test/）
    python scripts/tools/run_all.py --mode train-scratch  # 从头训练对照
    python scripts/tools/run_all.py --datasets IDS-17,DoH  # 指定数据集（输出位置不变）
    python scripts/tools/run_all.py --config-dir Kitsune   # 跑子目录里的逐类配置
    python scripts/tools/run_all.py --config-dir IDS2017 --only Bot,DoS_Hulk
    # 下游检测器对比：只改 detection_model，其余字段与逐类配置逐字相同
    python scripts/tools/run_all.py --config-dir IDS2017 --detectors rforest,kmeans,iforest

--detectors 说明：kmeans / iforest / ocsvm 为单类（无监督）检测器，只用训练集良性
样本拟合，以异常分数选阈值；rforest / logistic / knn / bayes 为有监督检测器；
graph_classifier 是端到端微调路径，不冻结 backbone，与上述冻结特征口径不可混比。
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
for _candidate in (repo_root, repo_root / "src", repo_root / "scripts"):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from common import *

RUNNER = os.path.join(str(repo_root / "scripts"), "run_graph_tfusion.py")


def select_configs(config_dir, only=None, include_shared=False):
    """Pick the per-class configs to run from one config directory.

    逐类配置由 script/make_class_configs.py 生成，特征就是 data_tag 只有一条；
    数据集级共享配置（all.json 之类）含多条 data_tag，除非显式 --include-shared
    否则跳过，避免同一次批跑里把每个类跑两遍。
    """
    selected = []
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith('.json') or name.startswith('.'):
            continue
        stem = os.path.splitext(name)[0]
        if only is not None and stem not in only:
            continue
        path = os.path.join(config_dir, name)
        with open(path) as handle:
            config = json.load(handle)
        tags = config.get('data_tag') or {}
        if not tags:
            logging.warning(f'[{stem}] 没有 data_tag，跳过')
            continue
        if len(tags) > 1 and not include_shared:
            logging.info(f'[{stem}] 含 {len(tags)} 个类（数据集级共享配置），跳过；'
                         '如需运行请加 --include-shared')
            continue
        selected.append((stem, path))
    return selected


def main():
    parser = argparse.ArgumentParser(description='Graph_Tfusion batch runner')
    parser.add_argument('--mode', default='train-test',
                        choices=['train-test', 'train-scratch'],
                        help='配置组：预训练+微调 / 从头训练对照（默认 train-test）')
    parser.add_argument('--datasets', default=None,
                        help='逗号分隔的数据集子集（默认全部）')
    parser.add_argument('--config-dir', default=None,
                        help='跑某个数据集子目录下的逐类配置，如 Kitsune / IDS2017 / DDOS_19')
    parser.add_argument('--only', default=None,
                        help='仅跑这些配置名（逗号分隔，配合 --config-dir 使用）')
    parser.add_argument('--detectors', default=None,
                        help='下游检测器对比（逗号分隔，如 rforest,kmeans,iforest）；'
                             '不填则沿用各配置自带的 detection_model。'
                             '每个检测器独立落盘，除 detection_model 外不改动任何字段')
    parser.add_argument('--include-shared', action='store_true',
                        help='配合 --config-dir：连同含多个 data_tag 的共享配置一起跑')
    args = parser.parse_args()

    base_dir = os.path.join(str(repo_root), 'config', args.mode)
    if not os.path.isdir(base_dir):
        logging.error(f'Config directory not found: {base_dir}')
        return -1

    only = {name.strip() for name in args.only.split(',')} if args.only else None

    if args.config_dir:
        config_dir = args.config_dir if os.path.isabs(args.config_dir) \
            else os.path.join(base_dir, args.config_dir)
        if not os.path.isdir(config_dir):
            logging.error(f'Config subdirectory not found: {config_dir}')
            return -1
        targets = select_configs(config_dir, only=only,
                                 include_shared=args.include_shared)
        if not targets:
            logging.error(f'{config_dir} 下没有可运行的配置。')
            return -1
        # 子目录名进输出路径，避免不同数据集的同名类（如 MSSQL / LDAP）互相覆盖
        label_prefix = os.path.relpath(config_dir, base_dir)
    else:
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(base_dir)
                       if f.endswith('.json'))
        if args.datasets:
            wanted = {d.strip() for d in args.datasets.split(',') if d.strip()}
            names = [t for t in names if t in wanted]
            missing = wanted - set(names)
            if missing:
                logging.error(f'Configs not found for: {sorted(missing)}')
                return -1
        targets = [(tag, os.path.join(base_dir, tag + '.json')) for tag in names]
        label_prefix = None

    detectors = None
    if args.detectors:
        detectors = [name.strip() for name in args.detectors.split(',') if name.strip()]
        if not detectors:
            logging.error('--detectors 解析为空。')
            return -1
        # 集中校验检测器名，避免跑到一半才发现拼写错误
        from detect.constants import ALL_DETECTION_MODELS
        unknown = [name for name in detectors if name not in ALL_DETECTION_MODELS]
        if unknown:
            logging.error(f'不支持的检测器 {unknown}；可选 {sorted(ALL_DETECTION_MODELS)}')
            return -1

    failures = []
    for tag, cfg_path in targets:
        with open(cfg_path) as f:
            cfg = json.load(f)
        # 结果输出位置统一为 log/all/<mode>/<数据集名>；子目录配置再下钻一层，
        # 保证不同数据集的同名类不会互相覆盖。
        base_label = os.path.join(label_prefix, tag) if label_prefix else tag
        for detector in detectors or [cfg.get('detection_model')]:
            run_cfg = json.loads(json.dumps(cfg))
            out_label = base_label
            if detectors:
                # 只替换检测器本身，其余字段逐字沿用配置；输出再下钻一层避免互相覆盖
                run_cfg['detection_model'] = detector
                out_label = f'{base_label}/{detector}'
                for key in ('fig_path', 'detection_model_path'):
                    if run_cfg.get(key):
                        run_cfg[key] = os.path.join(run_cfg[key], detector)
            run_cfg['log_path'] = os.path.join('./log/all', args.mode, out_label)
            cfg_dir = os.path.dirname(cfg_path)
            tmp_cfg = os.path.join(cfg_dir, f'.{tag}.tmp.json')
            with open(tmp_cfg, 'w') as f:
                json.dump(run_cfg, f, indent=2)
            logging.info(f'[{out_label}] Starting {args.mode} train+test ...')
            result = subprocess.run(
                [sys.executable, RUNNER, '-c', tmp_cfg],
                cwd=str(repo_root),
            )
            os.remove(tmp_cfg)
            if result.returncode != 0:
                logging.error(f'[{out_label}] FAILED (exitcode={result.returncode})')
                failures.append(out_label)

    if failures:
        logging.error(f'失败配置 ({len(failures)}): {failures}')
        return 1
    logging.info(f'全部 {len(targets)} 个配置运行成功。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
