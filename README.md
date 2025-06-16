# Pig Detector - OpenCV + PyTorch 项目

该项目用于使用 Python + OpenCV + PyTorch 训练"猪识别模型"，并可导出 ONNX 模型以供 Java 的 OpenCV (4.5.5) 调用。当前仓库仅提供代码框架，未包含已训练的权重。

## 📁 项目目录结构

```
pig-detector-opencv/
├── data/           # 训练与验证数据集
│   ├── train/
│   └── val/
│
├── scripts/        # 训练与推理脚本
│   ├── train.py
│   ├── predict.py
│   └── export_to_onnx.py
│
├── utils/          # 数据集与模型工具
│   ├── model.py
│   ├── dataset.py
│   └── transforms.py
│
├── config.yaml     # 训练参数示例
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 在 `config.yaml` 中配置数据集路径和训练参数。
3. 运行 `python scripts/train.py` 开始训练，训练完成后会在 `models/` 目录下保存权重。
4. 如需在 Java 中使用，可执行 `python scripts/export_to_onnx.py` 导出 ONNX 模型。

运行上述脚本时若仅查看 `--help` 信息，可在未安装深度学习依赖的情况下执行。
真正训练或导出模型则需要提前安装 `torch`、`torchvision` 等依赖，确保环境支持 GPU
或 CPU 推理。

> **注意**：本仓库仅提供代码框架，未包含已训练的模型。若要获得可用的检测效果，请在本地安装好 `torch`、`torchvision` 等依赖后自行训练。
