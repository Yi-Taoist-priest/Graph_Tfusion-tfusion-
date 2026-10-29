#!/usr/bin/env python3
"""生成 R3-V 全量消融的实时进度看板（自包含 HTML，30 秒自动刷新）。

数据源：save/ablations_r3v_20261026/**/metrics.json
输出：dashboard/r3v_progress.html
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE = os.path.join(ROOT, "save", "ablations_r3v_20261026")
RUNS = os.path.join(ROOT, "runs", "r3v_full_20261026")
OUT = os.path.join(ROOT, "dashboard", "r3v_progress.html")

DATASETS = [
    ("DDOS_19", 17),
    ("IDS2017", 12),
    ("UNSW_NB15", 9),
    ("IoT-23", 8),
    ("Kitsune", 3),
    ("CSECICIDS2018", 1),
]
ARMS = (
    "full", "no_history_context", "no_retrieval",
    "no_gated_residual", "no_modal_attn", "no_pre_train",
)
CST = timezone(timedelta(hours=8))
RUNS_DIRS = [
    os.path.join(ROOT, "runs", "r3v_full_20261026"),
    os.path.join(ROOT, "runs", "r3v_extra_20261026"),
]
RATIO_ROOT = os.path.join(ROOT, "save", "r3v_val_ratio_20261026")
RATIO_VALUES = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]
RATIO_EXPECTED_PER = 50 * 3 * len(ARMS)


def load_metrics():
    rows = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(SAVE, "*", "*", "*", "seed_*", "metrics.json")):
        parts = path.split(os.sep)
        arm, dataset, cls = parts[-5], parts[-4], parts[-3]
        if arm not in ARMS:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        rows[dataset][cls, arm].append(payload)
    return rows


def load_ratio_metrics():
    """读取 validation 阈值 + 比例 sweep：ratio_*/arm/dataset/class/seed/metrics.json。"""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.isdir(RATIO_ROOT):
        return rows
    pattern = os.path.join(RATIO_ROOT, "ratio_*", "*", "*", "*", "seed_*", "metrics.json")
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        arm, dataset, cls = parts[-5], parts[-4], parts[-3]
        ratio = parts[-6].replace("ratio_", "")
        if arm not in ARMS:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        rows[ratio][dataset][cls, arm].append(payload)
    return rows


def load_ratio_status(path):
    started, current, finished = None, None, False
    if not os.path.isfile(path):
        return started, current, finished
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "start validation ratio sweep" in line:
                started = line.split("]")[0].lstrip("[")
            elif "start ratio=" in line:
                rest = line.split("start ratio=")[1].strip()
                ratio = rest.split()[0] if rest else ""
                dataset = ""
                if "dataset=" in rest:
                    dataset = rest.split("dataset=")[1].split()[0]
                try:
                    label = f"{float(ratio) * 100:.0f}%"
                except ValueError:
                    label = ratio
                current = f"{label} {dataset}"
            elif "end ratio=" in line:
                current = None
            elif "finished" in line:
                finished = True
    return started, current, finished


def load_status(path):
    started, current, finished = None, None, False
    if not os.path.isfile(path):
        return started, current, finished
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if " start " in line and "dataset=" not in line and "arms=" in line:
                started = line.split("]")[0].lstrip("[")
            elif "start dataset=" in line:
                current = line.split("start dataset=")[1].split()[0]
            elif "end dataset=" in line:
                if current:
                    current = None
            elif "finished" in line:
                finished = True
    return started, current, finished


def mean(values):
    return st.mean(values) if values else float("nan")


def bootstrap_ci(values, seed=777, iterations=2000):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(iterations):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def arm_mean(entries, key):
    return mean([entry.get(key) for entry in entries if entry.get(key) is not None])


def build_payload():
    rows = load_metrics()
    main_status = load_status(os.path.join(RUNS_DIRS[0], "status.log"))
    started_raw = main_status[0]
    now = datetime.now(timezone.utc)
    started = None
    if started_raw:
        try:
            started = datetime.strptime(started_raw, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        except ValueError:
            started = None

    datasets_out = []
    per_class = []
    summaries = []
    pooled_d_auc, pooled_d_macro = [], []
    total_done = total_expected = 0
    class_means = {}
    arm_done = {arm: 0 for arm in ARMS}
    arm_expected = 50 * 3

    for name, n_classes in DATASETS:
        expected = n_classes * 3 * len(ARMS)
        done = sum(len(v) for v in rows.get(name, {}).values())
        total_done += done
        total_expected += expected
        for (cls, arm), entries in rows.get(name, {}).items():
            if arm in arm_done:
                arm_done[arm] += len(entries)
        status = "done" if done >= expected else ("running" if done > 0 else "pending")
        datasets_out.append({
            "name": name, "done": done, "expected": expected,
            "pct": round(100.0 * done / expected, 1), "status": status,
            "classes": n_classes,
        })

        for arm in ARMS:
            if arm == "full":
                continue
            d_auc, d_macro, d_atk = [], [], []
            full_auc_all, arm_auc_all = [], []
            full_macro_all, arm_macro_all = [], []
            full_atk_all, arm_atk_all = [], []
            steps, best_epoch, early = [], [], []
            for (cls, a), entries in rows.get(name, {}).items():
                if a != arm:
                    continue
                full_entries = rows[name].get((cls, "full"), [])
                if not full_entries:
                    continue
                f_auc, n_auc = arm_mean(full_entries, "auc"), arm_mean(entries, "auc")
                f_mac, n_mac = arm_mean(full_entries, "macro_f1"), arm_mean(entries, "macro_f1")
                f_atk, n_atk = arm_mean(full_entries, "attack_f1"), arm_mean(entries, "attack_f1")
                full_auc_all.append(f_auc); arm_auc_all.append(n_auc)
                full_macro_all.append(f_mac); arm_macro_all.append(n_mac)
                full_atk_all.append(f_atk); arm_atk_all.append(n_atk)
                d_auc.append(f_auc - n_auc)
                d_macro.append(f_mac - n_mac)
                d_atk.append(f_atk - n_atk)
                steps.extend(e.get("finetune_gradient_steps") or 0 for e in entries)
                best_epoch.extend(e.get("finetune_best_epoch") or 0 for e in entries)
                early.extend(1 if e.get("finetune_stopped_early") else 0 for e in entries)
                if arm == "no_history_context":
                    class_means[(name, cls)] = f_mac
                    per_class.append({
                        "dataset": name, "cls": cls,
                        "support": entries[0].get("support_attack"),
                        "full_auc": round(f_auc, 4), "nohist_auc": round(n_auc, 4),
                        "d_auc": round(f_auc - n_auc, 4),
                        "full_macro": round(f_mac, 4), "nohist_macro": round(n_mac, 4),
                        "d_macro": round(f_mac - n_mac, 4),
                        "full_atk": round(f_atk, 4), "nohist_atk": round(n_atk, 4),
                        "d_atk": round(f_atk - n_atk, 4),
                    })

            if d_auc:
                lo_a, hi_a = bootstrap_ci(d_auc)
                lo_m, hi_m = bootstrap_ci(d_macro)
                summaries.append({
                    "dataset": name, "arm": arm, "n": len(d_auc),
                    "full_auc": round(mean(full_auc_all), 4),
                    "arm_auc": round(mean(arm_auc_all), 4),
                    "d_auc": round(mean(d_auc), 4),
                    "auc_ci": [round(lo_a, 4), round(hi_a, 4)],
                    "win_auc": sum(1 for x in d_auc if x > 1e-6),
                    "full_macro": round(mean(full_macro_all), 4),
                    "arm_macro": round(mean(arm_macro_all), 4),
                    "d_macro": round(mean(d_macro), 4),
                    "macro_ci": [round(lo_m, 4), round(hi_m, 4)],
                    "full_atk": round(mean(full_atk_all), 4),
                    "arm_atk": round(mean(arm_atk_all), 4),
                    "steps": round(mean(steps)),
                    "best_epoch": round(mean(best_epoch), 1),
                    "early_rate": round(100.0 * mean(early), 1),
                })
                if arm == "no_history_context":
                    pooled_d_auc.extend(d_auc)
                    pooled_d_macro.extend(d_macro)

    # 稳健类：0.7 <= full Macro < 0.99
    robust = []
    by_dataset = defaultdict(list)
    for (name, cls), macro in class_means.items():
        if 0.7 <= macro < 0.99:
            item = next((r for r in per_class if r["dataset"] == name and r["cls"] == cls), None)
            if item:
                by_dataset[name].append(item)
    for name, items in by_dataset.items():
        if not items:
            continue
        da = [x["d_auc"] for x in items]; dm = [x["d_macro"] for x in items]
        lo_a, hi_a = bootstrap_ci(da); lo_m, hi_m = bootstrap_ci(dm)
        robust.append({
            "dataset": name, "n": len(items),
            "d_auc": round(mean(da), 4), "auc_ci": [round(lo_a, 4), round(hi_a, 4)],
            "d_macro": round(mean(dm), 4), "macro_ci": [round(lo_m, 4), round(hi_m, 4)],
            "win_auc": sum(1 for x in da if x > 1e-6),
            "classes": [x["cls"] for x in items],
        })

    elapsed_min = None
    if started:
        elapsed_min = max(0.0, (now - started).total_seconds() / 60.0)

    pooled = None
    if pooled_d_auc:
        lo_a, hi_a = bootstrap_ci(pooled_d_auc)
        lo_m, hi_m = bootstrap_ci(pooled_d_macro)
        pooled = {
            "n": len(pooled_d_auc),
            "d_auc": round(mean(pooled_d_auc), 4), "auc_ci": [round(lo_a, 4), round(hi_a, 4)],
            "d_macro": round(mean(pooled_d_macro), 4), "macro_ci": [round(lo_m, 4), round(hi_m, 4)],
            "win_auc": sum(1 for x in pooled_d_auc if x > 1e-6),
        }

    ratio_rows = load_ratio_metrics()
    ratio_progress = []
    ratio_agg = []
    ratio_by_dataset = []
    for ratio in RATIO_VALUES:
        ratio_ds = ratio_rows.get(ratio, {})
        done = sum(len(v) for classes in ratio_ds.values() for v in classes.values())
        ratio_progress.append({
            "ratio": ratio, "done": done, "expected": RATIO_EXPECTED_PER,
            "pct": round(100.0 * done / RATIO_EXPECTED_PER, 1),
            "status": "done" if done >= RATIO_EXPECTED_PER else ("running" if done else "pending"),
        })
        for arm in ARMS:
            if arm == "full":
                continue
            agg_d_auc, agg_d_macro = [], []
            agg_full_auc, agg_arm_auc = [], []
            agg_full_macro, agg_arm_macro = [], []
            strategies = set()
            for dataset_name, classes in ratio_ds.items():
                d_auc, d_macro = [], []
                f_auc_l, a_auc_l, f_mac_l, a_mac_l = [], [], [], []
                for (cls, a), entries in classes.items():
                    if a != arm:
                        continue
                    full_entries = classes.get((cls, "full"), [])
                    if not full_entries:
                        continue
                    f_auc, a_auc = arm_mean(full_entries, "auc"), arm_mean(entries, "auc")
                    f_mac, a_mac = arm_mean(full_entries, "macro_f1"), arm_mean(entries, "macro_f1")
                    if f_auc is None or a_auc is None:
                        continue
                    f_auc_l.append(f_auc); a_auc_l.append(a_auc)
                    f_mac_l.append(f_mac); a_mac_l.append(a_mac)
                    d_auc.append(f_auc - a_auc); d_macro.append(f_mac - a_mac)
                    for entry in entries + full_entries:
                        strategies.add(entry.get("threshold_strategy"))
                if d_auc:
                    ratio_by_dataset.append({
                        "ratio": ratio, "dataset": dataset_name, "arm": arm, "n": len(d_auc),
                        "full_auc": round(mean(f_auc_l), 4), "arm_auc": round(mean(a_auc_l), 4),
                        "d_auc": round(mean(d_auc), 4),
                        "full_macro": round(mean(f_mac_l), 4), "arm_macro": round(mean(a_mac_l), 4),
                        "d_macro": round(mean(d_macro), 4),
                    })
                    agg_d_auc.extend(d_auc); agg_d_macro.extend(d_macro)
                    agg_full_auc.extend(f_auc_l); agg_arm_auc.extend(a_auc_l)
                    agg_full_macro.extend(f_mac_l); agg_arm_macro.extend(a_mac_l)
            if agg_d_auc:
                ratio_agg.append({
                    "ratio": ratio, "arm": arm, "n": len(agg_d_auc),
                    "full_auc": round(mean(agg_full_auc), 4), "arm_auc": round(mean(agg_arm_auc), 4),
                    "d_auc": round(mean(agg_d_auc), 4),
                    "full_macro": round(mean(agg_full_macro), 4), "arm_macro": round(mean(agg_arm_macro), 4),
                    "d_macro": round(mean(agg_d_macro), 4),
                    "threshold_strategy": "/".join(sorted(str(x) for x in strategies if x)),
                })

    ratio_started_raw, ratio_current, _ = load_ratio_status(
        os.path.join(ROOT, "runs", "r3v_val_ratio_20261026", "status.log")
    )
    ratio_started = None
    if ratio_started_raw:
        try:
            ratio_started = datetime.strptime(
                ratio_started_raw, "%Y-%m-%d %H:%M:%S UTC"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            ratio_started = None
    ratio_done_total = sum(item["done"] for item in ratio_progress)
    ratio_expected_total = RATIO_EXPECTED_PER * len(RATIO_VALUES)
    ratio_finished = ratio_done_total >= ratio_expected_total
    ratio_elapsed_min = (
        max(0.0, (now - ratio_started).total_seconds() / 60.0)
        if ratio_started else None
    )
    ratio_eta_min = None
    if ratio_elapsed_min and ratio_done_total:
        ratio_rate = ratio_done_total / ratio_elapsed_min
        if ratio_rate > 0:
            ratio_eta_min = (ratio_expected_total - ratio_done_total) / ratio_rate

    failures = 0
    for runs_dir in RUNS_DIRS:
        for log in glob.glob(os.path.join(runs_dir, "*.log")):
            if log.endswith("status.log"):
                continue
            try:
                failures += open(log, encoding="utf-8", errors="ignore").read().count("FAIL(")
            except OSError:
                pass

    finished = total_done >= total_expected
    arm_progress = [
        {
            "arm": arm, "done": arm_done.get(arm, 0), "expected": arm_expected,
            "pct": round(100.0 * arm_done.get(arm, 0) / arm_expected, 1),
        }
        for arm in ARMS
    ]

    return {
        "updated": now.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S"),
        "started": started.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S") if started else None,
        "elapsed_min": round(elapsed_min) if elapsed_min is not None else None,
        "finished": finished,
        "done": total_done, "expected": total_expected,
        "pct": round(100.0 * total_done / total_expected, 1) if total_expected else 0.0,
        "failures": failures,
        "datasets": datasets_out,
        "arm_progress": arm_progress,
        "summaries": summaries,
        "robust": robust,
        "pooled": pooled,
        "per_class": sorted(per_class, key=lambda x: (x["dataset"], -x["d_auc"])),
        "ratio_progress": ratio_progress,
        "ratio_agg": ratio_agg,
        "ratio_by_dataset": ratio_by_dataset,
        "ratio_done": ratio_done_total,
        "ratio_expected": ratio_expected_total,
        "ratio_current": ratio_current,
        "ratio_finished": ratio_finished,
        "ratio_elapsed_min": round(ratio_elapsed_min) if ratio_elapsed_min is not None else None,
        "ratio_eta_min": round(ratio_eta_min) if ratio_eta_min is not None else None,
    }


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>R3-V 消融实时看板</title>
<style>
:root{--bg:#0b1220;--panel:#121b2e;--line:#22304a;--txt:#e6edf7;--muted:#8fa3bf;--ok:#31c48d;--bad:#f36b6b;--warn:#f0b429;--accent:#5b9dff;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 20% -10%,#17305a 0%,var(--bg) 55%);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px}
.wrap{max-width:1440px;margin:0 auto;padding:28px 24px 60px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:.3px}
.sub{color:var(--muted);font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px;background:var(--ok)}
.dot.run{background:var(--warn);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.card{background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01));border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.card .k{color:var(--muted);font-size:12px;margin-bottom:6px}
.card .v{font-size:24px;font-weight:650;letter-spacing:.4px}
.card .v small{font-size:12px;color:var(--muted);font-weight:400}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin:18px 0}
.panel h2{font-size:15px;margin:0 0 14px;letter-spacing:.4px;color:#c9d8ef}
.bar{height:10px;background:#1b2740;border-radius:99px;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#31c48d);border-radius:99px}
.ds{display:grid;grid-template-columns:150px 1fr 120px;gap:12px;align-items:center;padding:7px 0;border-bottom:1px dashed #1d2a42}
.ds:last-child{border-bottom:none}
.ds .name{font-weight:600}
.ds .meta{color:var(--muted);font-size:12px;text-align:right}
.tag{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}
.tag.done{color:var(--ok);border-color:#1f5c48;background:#0f2a22}
.tag.running{color:var(--warn);border-color:#5c4a17;background:#2a230f}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid #1b2740}
th{color:var(--muted);font-weight:600;font-size:12px;text-align:right}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:#16223a}
.pos{color:var(--ok)}.neg{color:var(--bad)}.zero{color:var(--muted)}
.note{color:var(--muted);font-size:12px;line-height:1.7}
.legend{display:flex;gap:18px;color:var(--muted);font-size:12px;margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.chart{width:100%;overflow-x:auto}
.empty{color:var(--muted);padding:10px 0}
code{background:#0d1626;border:1px solid #1e2c46;border-radius:6px;padding:1px 6px;font-size:12px}
.footer{margin-top:26px;color:var(--muted);font-size:12px;line-height:1.8}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>R3-V 全量消融 · 实时看板</h1>
    <div class="sub" id="sub">加载中…</div>
  </div>
  <div class="sub" id="updated"></div>
</header>

<div class="grid" id="kpis"></div>

<div class="panel">
  <h2>数据集进度</h2>
  <div id="progress"></div>
  <h2 style="margin-top:18px">各消融臂完成度（每臂 50 类 × 3 seed = 150 格）</h2>
  <div id="armprogress"></div>
</div>

<div class="panel">
  <h2>已完成的 arm 对比（等权：先类内平均 seed，再跨类平均）</h2>
  <div id="summary"></div>
  <div class="note" style="margin-top:10px">Δ = full − arm；正数表示该消融臂更差、full 更好。CI 为类级配对 bootstrap 95%。</div>
</div>

<div class="panel">
  <h2>validation 阈值 · 数据比例 sweep（full vs no_history_context）</h2>
  <div id="ratioprogress"></div>
  <div id="ratiotable"></div>
  <div class="chart" id="ratiochart"></div>
  <div class="chart" id="ratiodelta"></div>
  <div class="note" style="margin-top:10px">
    比例：1% / 2% / 5% / 10% / 20% / 50% / 100%；阈值为验证集选择（validation_search）。
    每档 6 臂 × 50 类 × 3 seed = 900 格。
    上图：full vs no_history 的 AUC 随比例变化；下图：各消融臂的 ΔAUC = full − arm 随比例变化。
  </div>
</div>

<div class="panel">
  <h2>稳健类子集（0.7 ≤ full Macro-F1 &lt; 0.99）</h2>
  <div id="robust"></div>
</div>

<div class="panel">
  <h2>逐类 ΔAUC（full − no_history）</h2>
  <div class="chart" id="chart"></div>
  <div class="legend"><span><i style="background:#31c48d"></i>full 更好</span><span><i style="background:#f36b6b"></i>full 更差</span><span><i style="background:#5b9dff"></i>持平 / |Δ|&lt;0.001</span></div>
</div>

<div class="panel">
  <h2>逐类明细</h2>
  <div id="table"></div>
</div>

<div class="panel">
  <h2>口径提示</h2>
  <div class="note">
    主指标为 <b>AUC</b>（阈值无关）；Macro-F1 当前仍按 <code>fixed_0.5</code> 计算，受分数校准影响，不能单独作为模块增益证据。
    DDOS_19 的 UDP 类就是典型例子：两臂 AUC 相近，但 0.5 切点让 no_history 的 Macro-F1 崩塌。
    最终结论应以 AUC/AUPRC + validation 阈值下的 F1 为准。
  </div>
</div>

<div class="footer" id="footer"></div>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const fmt = (x,n=4)=> (x===null||x===undefined||Number.isNaN(x))?'—':Number(x).toFixed(n);
const signed = (x,n=4)=> (x===null||x===undefined||Number.isNaN(x))?'—':(x>=0?'+':'')+Number(x).toFixed(n);
const cls = x => (x>0.0005?'pos':(x<-0.0005?'neg':'zero'));
const statusText = s => s==='done'?'已完成':(s==='running'?'进行中':'排队');

document.getElementById('sub').innerHTML =
  `<span class="dot ${(D.finished && D.ratio_finished)?'':'run'}"></span>` +
  `${D.finished?'主矩阵完成':'主矩阵进行中'} ${D.done}/${D.expected}` +
  ` · ${D.ratio_finished?'比例 sweep 完成':'比例 sweep 进行中'}` +
  (D.ratio_current?`（${D.ratio_current}）`:'') +
  ` ${D.ratio_done}/${D.ratio_expected}` +
  (D.started?` · 启动 ${D.started}`:'');

document.getElementById('updated').innerHTML =
  `最后更新 ${D.updated}（北京时间）<br>页面每 30 秒自动刷新`;

const rEta = D.ratio_eta_min===null?'—':(D.ratio_eta_min>=60?`${Math.floor(D.ratio_eta_min/60)}h ${D.ratio_eta_min%60}m`:`${D.ratio_eta_min}m`);
const kpis = [
  ['完成进度', `${D.done}<small> / ${D.expected}</small>`, `${D.pct}%`],
  ['运行状态', D.ratio_finished?(D.finished?'已完成':'—'):(D.ratio_current||'—'), D.ratio_finished?'':'比例 sweep'],
  ['已用时间', D.elapsed_min===null?'—':`${D.elapsed_min}m`, '主矩阵自启动'],
  ['比例 sweep 剩余', rEta, '按当前吞吐估算'],
  ['失败任务', D.failures, '扫描日志 FAIL'],
  ['图消融配对类数', D.pooled?D.pooled.n:0, 'full 与 no_history 都完成'],
  ['比例 sweep', `${D.ratio_done||0}<small> / ${D.ratio_expected||0}</small>`, 'validation 阈值 · 7 档'],
];
document.getElementById('kpis').innerHTML = kpis.map(([k,v,s])=>
  `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="k" style="margin:6px 0 0">${s}</div></div>`).join('');

document.getElementById('progress').innerHTML = D.datasets.map(d=>`
  <div class="ds">
    <div class="name">${d.name}<br><span class="tag ${d.status}">${statusText(d.status)}</span></div>
    <div><div class="bar"><i style="width:${d.pct}%"></i></div></div>
    <div class="meta">${d.done}/${d.expected}<br>${d.pct}%</div>
  </div>`).join('');

document.getElementById('armprogress').innerHTML = (D.arm_progress||[]).map(a=>`
  <div class="ds" style="grid-template-columns:220px 1fr 120px">
    <div class="name">${a.arm}</div>
    <div><div class="bar"><i style="width:${a.pct}%"></i></div></div>
    <div class="meta">${a.done}/${a.expected}<br>${a.pct}%</div>
  </div>`).join('');

function summaryTable(rows){
  if(!rows.length) return '<div class="empty">暂无完成的配对结果。</div>';
  const head = `<tr><th>数据集</th><th>消融臂</th><th>类数</th><th>full AUC</th><th>arm AUC</th><th>ΔAUC</th><th>95% CI</th><th>full 更好</th>
    <th>full Macro</th><th>arm Macro</th><th>ΔMacro</th><th>步数</th><th>best ep</th></tr>`;
  const body = rows.map(r=>`<tr>
    <td>${r.dataset}</td><td>${r.arm}</td><td>${r.n}</td>
    <td>${fmt(r.full_auc)}</td><td>${fmt(r.arm_auc)}</td>
    <td class="${cls(r.d_auc)}">${signed(r.d_auc)}</td>
    <td class="zero">[${signed(r.auc_ci[0])}, ${signed(r.auc_ci[1])}]</td>
    <td>${r.win_auc}/${r.n}</td>
    <td>${fmt(r.full_macro)}</td><td>${fmt(r.arm_macro)}</td>
    <td class="${cls(r.d_macro)}">${signed(r.d_macro)}</td>
    <td>${r.steps}</td><td>${r.best_epoch}</td></tr>`).join('');
  return `<table>${head}${body}</table>`;
}
document.getElementById('summary').innerHTML = summaryTable(D.summaries);

// ── validation 阈值 · 比例 sweep ──
const ratioPct = r => (Number(r)*100).toFixed(0) + '%';
document.getElementById('ratioprogress').innerHTML = (D.ratio_progress||[]).map(r=>`
  <div class="ds" style="grid-template-columns:130px 1fr 130px">
    <div class="name">${ratioPct(r.ratio)}<br><span class="tag ${r.status}">${statusText(r.status)}</span></div>
    <div><div class="bar"><i style="width:${r.pct}%"></i></div></div>
    <div class="meta">${r.done}/${r.expected}<br>${r.pct}%</div>
  </div>`).join('');

if((D.ratio_by_dataset||[]).length){
  const head=`<tr><th>比例</th><th>数据集</th><th>消融臂</th><th>类数</th><th>full AUC</th><th>arm AUC</th><th>ΔAUC</th>
    <th>full Macro</th><th>arm Macro</th><th>ΔMacro</th></tr>`;
  const body=D.ratio_by_dataset.map(r=>`<tr>
    <td>${ratioPct(r.ratio)}</td><td>${r.dataset}</td><td>${r.arm}</td><td>${r.n}</td>
    <td>${fmt(r.full_auc)}</td><td>${fmt(r.arm_auc)}</td><td class="${cls(r.d_auc)}">${signed(r.d_auc)}</td>
    <td>${fmt(r.full_macro)}</td><td>${fmt(r.arm_macro)}</td><td class="${cls(r.d_macro)}">${signed(r.d_macro)}</td>
  </tr>`).join('');
  document.getElementById('ratiotable').innerHTML=`<table>${head}${body}</table>`;
} else {
  document.getElementById('ratiotable').innerHTML='<div class="empty">比例 sweep 暂无完成的两臂配对。</div>';
}

function ratioAucChart(){
  const agg=(D.ratio_agg||[]).filter(r=>r.arm==='no_history_context');
  if(!agg.length) return '<div class="empty">暂无比例 sweep 结果。</div>';
  const W=1200,H=340,padL=80,padR=40,padT=30,padB=55;
  const n=agg.length;
  const xAt=i=>padL+(n===1?0:i*(W-padL-padR)/(n-1));
  const vals=[]; agg.forEach(r=>{if(r.full_auc!==null)vals.push(r.full_auc);if(r.arm_auc!==null)vals.push(r.arm_auc)});
  if(!vals.length) return '<div class="empty">暂无 AUC 数据。</div>';
  const vmin=Math.min.apply(null,vals)-0.01, vmax=Math.max.apply(null,vals)+0.01;
  const yAt=v=>H-padB-(v-vmin)/(vmax-vmin)*(H-padT-padB);
  let grid='';
  for(let k=0;k<=4;k++){
    const v=vmin+(vmax-vmin)*k/4, y=yAt(v);
    grid+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="#1e2c46" stroke-width="1"/>
      <text x="${padL-10}" y="${y+4}" fill="#8fa3bf" font-size="11" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  const pts=(key)=>agg.map((r,i)=>`${xAt(i)},${yAt(r[key])}`).join(' ');
  const dots=(key,color)=>agg.map((r,i)=>`<circle cx="${xAt(i)}" cy="${yAt(r[key])}" r="3.5" fill="${color}"/>`).join('');
  const labels=agg.map((r,i)=>`<text x="${xAt(i)}" y="${H-padB+22}" fill="#8fa3bf" font-size="12" text-anchor="middle">${ratioPct(r.ratio)}</text>`).join('');
  const valueLabels=agg.map((r,i)=>`<text x="${xAt(i)}" y="${yAt(r.full_auc)-8}" fill="#5b9dff" font-size="10" text-anchor="middle">${fmt(r.full_auc,3)}</text>`).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    ${grid}${labels}
    <polyline points="${pts('full_auc')}" fill="none" stroke="#5b9dff" stroke-width="2.5"/>
    <polyline points="${pts('arm_auc')}" fill="none" stroke="#f0b429" stroke-width="2.5" stroke-dasharray="6 4"/>
    ${dots('full_auc','#5b9dff')}${dots('arm_auc','#f0b429')}
    ${valueLabels}
    <text x="${padL}" y="18" fill="#5b9dff" font-size="12">● full</text>
    <text x="${padL+80}" y="18" fill="#f0b429" font-size="12">● no_history_context</text>
  </svg>`;
}

function ratioDeltaChart(){
  const all=(D.ratio_agg||[]);
  if(!all.length) return '<div class="empty">暂无 ΔAUC 数据。</div>';
  const ratios=['0.01','0.02','0.05','0.1','0.2','0.5','1.0'];
  const arms=['no_history_context','no_retrieval','no_gated_residual','no_modal_attn','no_pre_train'];
  const colors={'no_history_context':'#f0b429','no_retrieval':'#5b9dff','no_gated_residual':'#31c48d','no_modal_attn':'#b07cff','no_pre_train':'#f36b6b'};
  const W=1200,H=380,padL=80,padR=40,padT=55,padB=55;
  const xAt=i=>padL+(ratios.length===1?0:i*(W-padL-padR)/(ratios.length-1));
  const vals=all.map(r=>r.d_auc);
  const maxAbs=Math.max(0.005,...vals.map(Math.abs));
  const mid=(H-padB+padT)/2;
  const yAt=v=>mid-(v/maxAbs)*(H-padT-padB)/2;
  let grid='';
  for(let k=-2;k<=2;k++){
    const v=maxAbs*k/2, y=yAt(v);
    grid+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="#1e2c46"/>
      <text x="${padL-10}" y="${y+4}" fill="#8fa3bf" font-size="11" text-anchor="end">${v.toFixed(3)}</text>`;
  }
  const labels=ratios.map((r,i)=>`<text x="${xAt(i)}" y="${H-padB+22}" fill="#8fa3bf" font-size="12" text-anchor="middle">${ratioPct(r)}</text>`).join('');
  let lines='', legend='';
  arms.forEach((arm,idx)=>{
    const pts=[];
    ratios.forEach((r,i)=>{
      const hit=all.find(x=>x.arm===arm && x.ratio===r);
      if(hit) pts.push(`${xAt(i)},${yAt(hit.d_auc)}`);
    });
    if(pts.length){
      lines+=`<polyline points="${pts.join(' ')}" fill="none" stroke="${colors[arm]}" stroke-width="2"/>`;
      all.filter(x=>x.arm===arm).forEach(x=>{
        const i=ratios.indexOf(x.ratio);
        if(i>=0) lines+=`<circle cx="${xAt(i)}" cy="${yAt(x.d_auc)}" r="3" fill="${colors[arm]}"/>`;
      });
    }
    legend+=`<text x="${padL+idx*205}" y="20" fill="${colors[arm]}" font-size="11">● ${arm}</text>`;
  });
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${grid}${labels}${lines}
    <line x1="${padL}" y1="${mid}" x2="${W-padR}" y2="${mid}" stroke="#2a3b59"/>
    ${legend}
    <text x="${padL}" y="${H-8}" fill="#8fa3bf" font-size="11">ΔAUC = full − arm；正=full 更好</text>
  </svg>`;
}

document.getElementById('ratiochart').innerHTML = ratioAucChart();
document.getElementById('ratiodelta').innerHTML = ratioDeltaChart();

document.getElementById('robust').innerHTML = D.robust.length ? D.robust.map(r=>`
  <div style="margin-bottom:10px">
    <b>${r.dataset}</b> <span class="tag">${r.n} 类</span>
    <span class="${cls(r.d_auc)}">ΔAUC ${signed(r.d_auc)}</span>
    <span class="zero">[${signed(r.auc_ci[0])}, ${signed(r.auc_ci[1])}]</span>
    · <span class="${cls(r.d_macro)}">ΔMacro ${signed(r.d_macro)}</span>
    <span class="zero">[${signed(r.macro_ci[0])}, ${signed(r.macro_ci[1])}]</span>
    · ${r.win_auc}/${r.n} 类 full 更好
    <div class="note">${r.classes.join(' / ')}</div>
  </div>`).join('') : '<div class="empty">暂无稳健类结果。</div>';

const rows = D.per_class.filter(x=>Math.abs(x.d_auc)>=0).slice(0,60);
if(rows.length){
  const W=1200, rowH=22, padL=210, padR=40, H=rows.length*rowH+30;
  const maxAbs = Math.max(0.02, ...rows.map(x=>Math.abs(x.d_auc)));
  const x0 = padL + (W-padL-padR)/2;
  const scale = (W-padL-padR)/2/maxAbs;
  const bars = rows.map((r,i)=>{
    const y=i*rowH+20, w=Math.abs(r.d_auc)*scale;
    const positive=r.d_auc>0.0005, negative=r.d_auc<-0.0005;
    const color=positive?'#31c48d':(negative?'#f36b6b':'#5b9dff');
    const x=positive?x0:x0-w;
    return `<text x="${padL-10}" y="${y+14}" fill="#a9bbd6" font-size="12" text-anchor="end">${r.dataset}/${r.cls}</text>
      <rect x="${x}" y="${y+3}" width="${w}" height="14" rx="4" fill="${color}" opacity=".85"></rect>
      <text x="${positive?x+w+6:x-6}" y="${y+14}" fill="${color}" font-size="11" text-anchor="${positive?'start':'end'}">${signed(r.d_auc,4)}</text>`;
  }).join('');
  document.getElementById('chart').innerHTML =
    `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <line x1="${x0}" y1="10" x2="${x0}" y2="${H-10}" stroke="#2a3b59" stroke-width="1"></line>
      ${bars}</svg>`;
} else document.getElementById('chart').innerHTML='<div class="empty">暂无逐类结果。</div>';

if(D.per_class.length){
  const head=`<tr><th>数据集</th><th>类别</th><th>攻击样本</th><th>full AUC</th><th>no_hist AUC</th><th>ΔAUC</th>
    <th>full Macro</th><th>no_hist Macro</th><th>ΔMacro</th><th>full Atk-F1</th><th>no_hist Atk-F1</th></tr>`;
  const body=D.per_class.map(r=>`<tr>
    <td>${r.dataset}</td><td>${r.cls}</td><td>${(r.support===null||r.support===undefined)?'—':r.support}</td>
    <td>${fmt(r.full_auc)}</td><td>${fmt(r.nohist_auc)}</td><td class="${cls(r.d_auc)}">${signed(r.d_auc)}</td>
    <td>${fmt(r.full_macro)}</td><td>${fmt(r.nohist_macro)}</td><td class="${cls(r.d_macro)}">${signed(r.d_macro)}</td>
    <td>${fmt(r.full_atk)}</td><td>${fmt(r.nohist_atk)}</td></tr>`).join('');
  document.getElementById('table').innerHTML=`<table>${head}${body}</table>`;
} else document.getElementById('table').innerHTML='<div class="empty">暂无逐类结果。</div>';

const pooled = D.pooled ? `<br>已完成类池化：ΔAUC ${signed(D.pooled.d_auc)} [${signed(D.pooled.auc_ci[0])}, ${signed(D.pooled.auc_ci[1])}]，
  ΔMacro ${signed(D.pooled.d_macro)} [${signed(D.pooled.macro_ci[0])}, ${signed(D.pooled.macro_ci[1])}]，
  ${D.pooled.win_auc}/${D.pooled.n} 类 full 更好。` : '';
document.getElementById('footer').innerHTML =
  `数据目录：<code>save/ablations_r3v_20261026</code> · 日志：<code>runs/r3v_full_20261026/status.log</code>
   · 配置：R3-V（external val, zero-init gate, after_context, 30s/16 neighbors, max_steps=1200）${pooled}`;
</script>
</body>
</html>
"""


def _clean_json(value):
    """把 NaN/Inf 转成 None，保证嵌入的 JSON 可被浏览器 JSON.parse。"""
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    payload = _clean_json(build_payload())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    html = HTML.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(html)
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({payload['done']}/{payload['expected']}, {payload['pct']}%)")


if __name__ == "__main__":
    main()
