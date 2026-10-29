"""配置校验与训练进程的失败校验。

从入口脚本抽出，供主线与各实验变体共用。
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from detect.constants import ALL_DETECTION_MODELS, THRESHOLD_STRATEGIES


# 预切分模式下 data_construct_param 中已被 loader 忽略的旧划分键。
# 留在配置里会让人误以为仍在生效，因此显式拒绝而非静默忽略。
# validation_ratio 不在此列：data_split.val 关闭时训练内部仍按它切分。
PRE_SPLIT_LEGACY_KEYS = ('train_ratio', 'train_number', 'test_samp_ratio', 'split_strategy')


def _validate_data_split_section(jin: Dict) -> bool:
    """校验 data_split（预切分 train/test/val 文件）配置段。

    正式运行只有一条切分路径：``command=train`` 必须启用预切分；
    ``command=pre-train`` 可读单文件，也可选预切分 train。运行期不再做
    内存切分，旧划分键与 clean_traffic_* 均已删除。
    """
    section = jin.get('data_split')
    command = jin.get('command')
    enabled = isinstance(section, dict) and bool(section.get('use_pre_split'))

    if command == 'train' and not enabled:
        logging.error(
            'command=train requires data_split.use_pre_split=true; '
            'in-memory splitting has been removed.'
        )
        return False
    if command == 'pre-train' and section is not None and not enabled:
        logging.error('data_split.use_pre_split must be true when data_split is present.')
        return False
    if not enabled:
        return _reject_legacy_split_keys(jin)

    if not isinstance(section, dict):
        logging.error('data_split must be a JSON object.')
        return False
    if len(jin.get('data_tag', {})) != 1:
        logging.error('data_split requires exactly one data_tag entry per config.')
        return False
    if jin.get('clean_traffic_tag') or jin.get('clean_traffic_path'):
        logging.error(
            'clean_traffic_tag/clean_traffic_path have been removed; '
            'benign flows live inside the pre-split files.'
        )
        return False

    if not _reject_legacy_split_keys(jin):
        return False

    params = jin.get('data_construct_param') or {}
    if command == 'train':
        if params.get('graph_context_mode', 'full_stream_causal') != 'full_stream_causal':
            logging.error(
                'command=train only supports graph_context_mode="full_stream_causal".'
            )
            return False
        if params.get('host_feature_mode', 'causal') != 'causal':
            logging.error('command=train only supports host_feature_mode="causal".')
            return False

    for name in ('train', 'test'):
        entry = section.get(name)
        if not isinstance(entry, dict) or entry.get('enable') is not True:
            logging.error(
                f"data_split.{name} must be enabled (enable=true) when "
                "use_pre_split is on; it cannot be disabled."
            )
            return False
        path = entry.get('path')
        if not isinstance(path, str) or not path:
            logging.error(f'data_split.{name}.path must be a non-empty string.')
            return False
        if not os.path.isfile(path):
            logging.error(f'data_split.{name}.path does not exist: {path}')
            return False

    val_entry = section.get('val')
    if val_entry is None:
        # val 缺省视为关闭，训练内部按 validation_ratio 切分。
        return True
    if not isinstance(val_entry, dict):
        logging.error('data_split.val must be a JSON object.')
        return False
    if val_entry.get('enable'):
        path = val_entry.get('path')
        if not isinstance(path, str) or not path:
            logging.error('data_split.val.path must be a non-empty string when val is enabled.')
            return False
        if not os.path.isfile(path):
            logging.error(f'data_split.val.path does not exist: {path}')
            return False
    return True


def _reject_legacy_split_keys(jin: Dict) -> bool:
    params = jin.get('data_construct_param') or {}
    legacy = [key for key in PRE_SPLIT_LEGACY_KEYS if key in params]
    if not legacy:
        return True
    logging.error(
        'data_construct_param contains removed split keys: %s; the split is '
        'defined entirely by data_split train/test/val files.',
        ', '.join(legacy),
    )
    return False


def _validate_threshold_and_sampling(jin: Dict) -> bool:
    """校验训练集抽样比例与阈值策略。

    ``data_construct_param.train_sample_ratio`` 取值 (0, 1]（仅预切分模式生效）；
    ``train_param.threshold_strategy`` 只允许 ``fixed_0.5`` / ``validation``。
    旧 ``optimize_threshold`` 仍接受，并自动映射为这两种策略。
    """
    params = jin.get('data_construct_param') or {}
    ratio = params.get('train_sample_ratio')
    if ratio is not None:
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            logging.error(f'train_sample_ratio must be a number, got {ratio!r}.')
            return False
        if not 0.0 < ratio <= 1.0:
            logging.error(f'train_sample_ratio must be in (0, 1], got {ratio!r}.')
            return False

    stage = params.get('train_sample_stage')
    if stage is not None and stage not in ('before_context', 'after_context'):
        logging.error(
            "train_sample_stage must be 'before_context' or 'after_context', "
            f"got {stage!r}."
        )
        return False

    train_param = jin.get('train_param') or {}
    strategy = train_param.get('threshold_strategy')
    if strategy is not None and strategy not in THRESHOLD_STRATEGIES:
        logging.error(
            f'threshold_strategy must be one of {THRESHOLD_STRATEGIES}, '
            f'got {strategy!r}.'
        )
        return False

    optimize = train_param.get('optimize_threshold')
    if optimize is not None and not isinstance(optimize, bool):
        logging.error(f'optimize_threshold must be a boolean, got {optimize!r}.')
        return False
    return True


def validate_json_config_pp(jin: Dict) -> bool:
    """校验 tFusion++ 配置"""
    common_keys = [
        'command', 'data_path', 'log_path', 'fig_path',
        'gpu_enable', 'data_construct_param', 'train_param', 'detection_model',
    ]
    for k in common_keys:
        if k not in jin:
            logging.error(f'Key {k} missing in configuration.')
            return False

    if jin['command'] == 'pre-train':
        if 'traffic_model_path' not in jin:
            return False
    elif jin['command'] == 'train':
        for k in ['traffic_model_tag', 'traffic_model_path', 'detection_model_path']:
            if k not in jin:
                logging.error(f'Key {k} missing for training.')
                return False
        if jin.get('detection_model') not in ALL_DETECTION_MODELS:
            logging.error(
                f"Unsupported detection_model={jin.get('detection_model')!r}; "
                f"choose from {ALL_DETECTION_MODELS}."
            )
            return False
        if jin.get('graph_feature_mode', 'structural') != 'structural':
            logging.error('Only graph_feature_mode="structural" is currently supported.')
            return False
    if jin.get('use_graph', True) is not True:
        logging.error('This entrypoint only supports the Graph-Tfusion++ path; set use_graph=true or remove it.')
        return False

    if not _validate_data_split_section(jin):
        return False
    if not _validate_threshold_and_sampling(jin):
        return False

    # 创建必要目录
    try:
        for p in [jin['log_path'], jin['fig_path'], jin['traffic_model_path']]:
            if not os.path.exists(p):
                os.makedirs(p)
        if 'detection_model_path' in jin and not os.path.exists(jin['detection_model_path']):
            os.makedirs(jin['detection_model_path'])
    except Exception as e:
        logging.error(f'Exception in create folder: {e}')
        return False

    return True


def ensure_processes_succeeded(processes) -> None:
    failures = [process for process in processes if process.exitcode not in (0, None)]
    if failures:
        detail = ", ".join(f"{process.name}(exitcode={process.exitcode})" for process in failures)
        raise RuntimeError(f"Training worker failed: {detail}")


def validate_formal_threshold_contract(jin: Dict) -> bool:
    """Formal runs only allow the absolute validation threshold.

    Returns True when the contract holds; logs an error and returns False
    otherwise. ``prior``/``quantile`` re-express the cutoff using the
    evaluated population (its prevalence or score distribution) and are
    rejected so a formal configuration can never reach the training path.
    """
    train_param = jin.get("train_param", {}) or {}
    transfer = train_param.get("threshold_transfer", "absolute")
    if transfer != "absolute":
        logging.error(
            "Formal runs only allow threshold_transfer='absolute'; "
            "got %r. 'prior' and test-score 'quantile' transfer read the "
            "evaluated population and are forbidden.",
            transfer,
        )
        return False
    return True
