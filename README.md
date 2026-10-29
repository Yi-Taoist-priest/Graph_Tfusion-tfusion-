# Graph_Tfusion

图增强 tFusion++ 异常流量检测。主线：

```text
PCAP + 切分产物 → tFusion JSON → 预训练 → 逐类检测 → 指标 / 日志
```

本文只交代主线：数据怎么转、配置在哪、从哪个入口跑。

## 环境

```bash
python -m pip install -r requirements.txt
```

GPU 可选；配置里的 `gpu_enable` 不可用时自动回退 CPU。

## 1. 数据转换

输入是 `mini_datasets` 的 v4 预切分布局：

```text
mini_datasets/<Dataset>/<Class>/{train,test,val}/{train,test,val}.pcap|.csv
```

转换为逐类 JSON：

```text
data/train-test/<Dataset>/<Class>/{train,test,val}.json
```

命令：

```bash
python scripts/pcap/convert_cicids_pcap.py \
  --mini-root /path/to/mini_datasets \
  --output-dir data/train-test \
  --datasets CICIDS2017,CICDDoS2019,UNSW_NB15,IoT-23,Kitsune,CSECICIDS2018 \
  --workers 8
```

无标签 PCAP 直接抽特征：

```bash
python scripts/pcap/extract_pcap_features.py input.pcap -o output.json
```

JSON 顶层为 `benign` / `attack`；每条流包含 `seq_feature`、六维 `nonseq_feature`、
五元组、`start_ts`、`end_ts`。

## 2. 数据集与配置

| 配置目录 | 数据目录 | 类数 |
|---|---|---:|
| `config/train-test/IDS2017` | `data/train-test/CICIDS2017` | 12 |
| `config/train-test/DDOS_19` | `data/train-test/CICDDoS2019` | 17 |
| `config/train-test/UNSW_NB15` | `data/train-test/UNSW_NB15` | 9 |
| `config/train-test/IoT-23` | `data/train-test/IoT-23` | 8 |
| `config/train-test/Kitsune` | `data/train-test/Kitsune` | 3 |
| `config/train-test/CSECICIDS2018` | `data/train-test/CSECICIDS2018` | 1 |

- 预训练配置：`config/pre-train/pretrain.json`（campus）、`config/pre-train/school.json`
- 检测配置：`config/train-test/<Dataset>/<Class>.json`，每个文件只含一个 `data_tag`

数据转换完成后，把 `data_split` 写进逐类配置：

```bash
python scripts/tools/add_data_split_section.py --data-root data/train-test
```

正式训练要求 `data_split.use_pre_split=true`，train/test/val 完全由文件决定。

## 3. 主程序入口

### 3.1 预训练

```bash
python scripts/run_graph_tfusion.py -c config/pre-train/pretrain.json
# 或
python scripts/run_graph_tfusion.py -c config/pre-train/school.json
```

产物：

```text
save/traffic_model/<tag>/<tag>_pp_model.pt
```

`config/pre-train/pretrain.json` 的 `data_path` 是示例路径，按本机数据位置修改；
`school.json` 指向仓库内 `data/pretrain/test`。

### 3.2 单类检测

```bash
python scripts/run_graph_tfusion.py -c config/train-test/IDS2017/PortScan.json
```

流程：加载预训练 backbone → 微调 → 冻结抽特征 → 训练下游检测器 → 测试。

### 3.3 批量检测

```bash
python scripts/tools/run_all.py --config-dir IDS2017
python scripts/tools/run_all.py --config-dir IDS2017 --only PortScan,DoS_Hulk
```

### 3.4 正式矩阵

不可覆盖、逐次落盘 `run_config.json`：

```bash
python scripts/run_formal_matrix.py \
  -c config/train-test/IDS2017/PortScan.json \
  --seeds 777 888 999 --variant main_v1
```

### 3.5 消融矩阵

```bash
python scripts/run_ablation_matrix.py \
  --dataset DDOS_19 --ratio 0.02 --seeds 777 888 999 \
  --arms full no_history_context \
  --workers 8 --gpu 2
```

当前 R3-V 配方：

```bash
WORKERS=10 GPU=2 bash scripts/run_r3v_full.sh            # fixed 0.5, 6 臂, 2%
WORKERS=8  GPU=6 bash scripts/run_r3v_val_ratio.sh        # validation, 7 档比例, 2 臂
WORKERS=6  GPU=6 bash scripts/run_r3v_val_ratio_extra.sh
```

## 4. 产物

```text
save/traffic_model/<tag>/                预训练权重
save/detection_model/<Dataset>/<Class>/  检测模型
log/...                                  训练日志
figure/...                               图
runs/...                                 矩阵状态与逐数据集日志
```

每个 `seed_*` 目录下的 `metrics.json` 记录 `auc`、`auprc`、`macro_f1`、
`attack_f1`、`threshold`、`threshold_strategy` 等。

实时看板：

```bash
python scripts/tools/build_r3v_dashboard.py
# 输出 dashboard/r3v_progress.html
```

## 5. 测试

```bash
pytest -q
```
