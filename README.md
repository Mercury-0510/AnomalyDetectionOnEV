# 电动汽车异常检测系统 (基于 ModernTCN)

基于 ModernTCN 的电动汽车时间序列异常检测系统，用于识别电动汽车运行数据中的故障模式。

## 系统特性

- **数据处理**: 自动处理 CSV 格式的电动汽车数据，支持多通道时间序列
- **模型训练**: 使用 ModernTCN 进行时间序列分类任务
- **异常检测**: 基于样本预测结果进行文件级异常判断
- **结果导出**: 自动生成预测结果、混淆矩阵和异常文件列表

## 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据准备
将 CSV 格式的用于训练的电动汽车数据放入 `./all_datasets/train_datasets` 文件夹  
将 txt 格式的记录训练数据中故障的电动汽车数据文件名目录写为 `./all_datasets/abnormal_list.txt`  
如:
```bash
JBGS25000000620
JBGS25000007612
JBGS25000005692
JBGS25000008795
```
将 CSV 格式的待预测电动汽车数据放入 `./all_datasets/predict_datasets` 文件夹  
系统支持以下通道(若其他数据完整可做拓展)：
- `SUM_VOLTAGE`: 总电压
- `SUM_CURRENT`: 总电流  
- `SOC`: 电池荷电状态
- `U_SD`: 单体电压标准差（自动计算）
- `T_SD`: 温度标准差（自动计算）

### 3. 数据处理及转化
```bash
cd ./ModernTCN-classification
sh scripts/data_process.sh
```
或直接运行`./run_data.py`，内含三种任务可选：  
- `--fft`:使用快速傅里叶变换进行数据扰动，用以扩充故障数据解决分类不平衡问题，从`./data_preprocessor/fft_list.txt`中获取需要扩充的文件目录     
- `--denoise`:进行平滑，处理原数据中出现的异常时间点  
- `--transform`:将原文件csv池化、分段为UEA数据集标准格式`EV_xx.ts`，用于模型训练

### 4. 模型训练
```bash
cd ./ModernTCN-classification
sh scripts/run_classification.sh
```
或直接运行`./run.py`，参数可调

## 输出结果

训练和预测完成后，系统会生成：

- `results/{setting}/predictions.csv`: 样本级预测结果
- `results/{setting}/file_level_predictions_test.csv`: 文件级预测结果
- `results/{setting}/file_level_predictions_test_abnormal_files.txt`: 异常文件列表
- `plots/{setting}/training_curves.png`: 训练曲线
- `plots/{setting}/confusion_matrix.png`: 混淆矩阵

## 配置参数

### 基础训练参数
- `--threshold`: 异常判断阈值
- `--train_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--dropout`: Dropout 率
- `--patience`: 早停耐心值
- `--random_seed`: 随机种子

### 数据配置参数
- `--seq_len`: 输入序列长度
- `--enc_in`: 编码器输入维度
- `--c_out`: 输出维度
- `--num_workers`: 数据加载器工作进程数

### ModernTCN 网络参数
- `--patch_size`: 补丁大小
- `--patch_stride`: 补丁步长
- `--stem_ratio`: 主干比例
- `--downsample_ratio`: 下采样比例
- `--ffn_ratio`: 前馈网络比例
- `--dims`: 各阶段维度
- `--num_blocks`: 各阶段块数
- `--large_size`: 大卷积核尺寸
- `--small_size`: 小卷积核尺寸
- `--use_multi_scale`: 是否使用多尺度融合

### 运行控制参数
- `--is_training`: 训练模式
- `--do_predict`: 是否执行预测
- `--itr`: 实验重复次数
- `--use_gpu`: 是否使用GPU
- `--gpu`: GPU设备ID

## 项目结构

```
ModernTCN-classification/
├── data_transform/             # 数据转换模块
│   └── data_transform.py
├── exp/                        # 实验模块
│   ├── exp_basic.py       
│   └── exp_classification.py  
├── data_provider/              # 数据提供者模块
│   └── ...
├── models/                     # 模型定义
│   └── ModernTCN.py       
├── layers/                     # 网络层定义
│   └── ...
├── utils/                      # 工具函数
│   └── ...
├── scripts/                    # 运行脚本
│   ├── data_process.sh         # 数据处理脚本
│   └── run_classification.sh   # 分类训练脚本
├── all_datasets/EV/            # 数据集目录及样本映射
│   └── ...
├── checkpoints/                # 模型检查点
│   └── {setting}/
│       └── checkpoint.pth
├── predict_results/            # 预测结果
│   └── {setting}/
│       ├── file_level_predictions_predict.csv                      # 文件级预测结果
│       └── file_level_predictions_predict_abnormal_files.txt       # 异常文件清单
├── plots/                      # 可视化结果
│   └── {setting}/
│       ├── training_curves.png                                     # 训练曲线
│       └── confusion_matrix.png                                    # 混淆矩阵
├── run_data.py                 # 数据处理脚本
├── run.py                      # 模型运行脚本
└── requirements.txt            # 依赖包列表
```

## 原始论文

本项目基于 ICLR 2024 论文:
**ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis**
[[OpenReview]](https://openreview.net/forum?id=vpJMJerXHU)
