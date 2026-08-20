# ADPT 发布版完整重跑清单（Python 3.12）

本清单用于在一台没有 ADPT 开发源码的新环境中验收 Research 或 Commercial
二进制发布包。测试结果、日志、配置、GPU 信息和文件哈希应随发布版本归档。

## 1. 准备干净环境

```bash
python3.12 -m venv ~/venvs/adpt-release-test
source ~/venvs/adpt-release-test/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

先安装与 GPU 匹配的 TensorFlow，再安装发布包内依赖：

```bash
pip install /path/to/compatible-tensorflow.whl
pip install -r requirements_gui.txt
pip install -e . --no-deps
```

记录：

```bash
python --version
python -m pip freeze > release-test-pip-freeze.txt
nvidia-smi > release-test-nvidia-smi.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

RTX 5090 额外运行：

```bash
python verify_blackwell.py
```

## 2. 验证二进制封装

从私有源码树运行：

```bash
python release/verify_protected_bundle.py /path/to/unpacked-release
```

验收条件：

- `core/train.py` 和 `core/predict.py` 不存在；
- GUI 入口只加载 `adpt_runtime`；
- SHA-256 清单通过；
- `adpt_engine.engine_info()` 报告 `compiled: true`。

## 3. GUI 与项目管理

```bash
python -m adpt_gui
```

- GUI 能正常打开；
- New Project 能建立自包含项目；
- Open Project 能打开该项目；
- 新项目包含同平台的 `.so` 或 `.pyd`，不包含受保护源码；
- Geek/Science 主题、窗口缩放和面板布局正常。

## 4. 仅录制视频

- 设置 1、2、3、4 个视角分别测试；
- 点击 `Record Video Only`；
- 确认没有加载 TensorFlow/模型、没有生成 H5；
- Stop 后各视角 AVI 可以完整播放，帧数、FPS、分辨率合理；
- 若启用仪器同步，CSV/JSONL 中有对应帧事件。

## 5. 标注

- 从录制视频抽帧；
- 完成至少两个文件夹的标注；
- 切换图片和 Clear Frame 后从第一个身体点开始；
- 保存后确认旧文件夹 annotation 没有被覆盖；
- Apply to Train 后训练配置路径正确。

## 6. 训练冒烟测试

使用小数据集，把 EPOCHS 临时设为 1–2，只验证管线，不评估模型精度：

- 训练后端显示 `adpt_engine.train`；
- GPU Conv2D、数据增强、Dataset、AdamW 正常；
- 训练进度、loss、validation 和日志持续更新；
- 生成 `*.weights.h5`；
- Stop Training 能在当前 batch 后安全结束。

再使用正式配置完成一次完整训练，并记录最终模型、配置与 SHA-256。

## 7. 实时/视频推理

- 加载与训练配置匹配的权重；
- 分别测试显示姿态、隐藏姿态、保存姿态视频；
- 隐藏姿态时中间面板关闭，Control 和 Analysis 保留；
- H5 包含所有活动视角；
- 保存的视频可以播放且速度正确；
- 双 GPU 机器记录每个视角的设备分配和吞吐量。

## 8. 2D 分析

- 加载 H5；
- 检查坐标、置信度、轨迹、速度和逐帧表格；
- 用 ROI 标定物理尺寸；
- 确认图、表、统计和 CSV 全部从 px/px/s 更新为实际单位；
- Export All Views 生成与视角数相同的独立 CSV。

## 9. 标定与 3D

- 选择特定视角组；
- 用默认 0.75 和另一个阈值分别标定；
- 确认至少两个合格视角才生成 3D 点；
- 运行 Ground Level，确认地面为 Z=0；
- Undo Level 能恢复原始坐标；
- 播放 3D 时对应 2D 帧同步；
- 导出 CSV 并确认尺度是相对单位还是已经独立恢复的物理单位。

## 10. 发布归档

归档以下内容：

- 发布 ZIP 与 SHA-256；
- Git commit/tag；
- Research 或 Commercial 最终许可证；
- 第三方 notices；
- Python/pip/GPU/CUDA/cuDNN 信息；
- 完整测试日志；
- 一个可复现的小型测试项目；
- 正式模型及其训练配置哈希。
