# Pig Detector - OpenCV + PyTorch 项目

该项目用于使用 Python + OpenCV + PyTorch 训练"猪识别模型"，并可导出 ONNX 模型以供 Java 的 OpenCV (4.5.5) 调用。当前仓库仅提供代码框架，未包含已训练的权重。

## 环境要求

- Python **3.11**（兼容 3.8 及以上版本）

## 📁 项目目录结构

```
pig-detector-opencv/
├── data/           # 训练与验证数据集 (YOLO 格式)
│   ├── train/
│   │   ├── images/  # 图片文件
│   │   └── labels/  # 同名 .txt，格式: class cx cy w h
│   └── val/
│       ├── images/
│       └── labels/
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
1. 执行 `./start_train.sh`（首次运行会创建虚拟环境并安装依赖）。
2. 在 `config.yaml` 中配置数据集路径和训练参数。
3. 训练结束后模型会保存在 `models/` 目录。
4. 如需在 Java 中使用，可执行 `python scripts/export_to_onnx.py` 导出 ONNX 模型。
5. 若希望通过 Docker 构建环境，可运行 `./build_docker.sh` 生成镜像。

如需通过 HTTP 调用模型，可启动 FastAPI 服务：

```bash
python scripts/api.py
```

然后向 `POST /predict` 发送 JSON `{ "image_path": "path/to/img.jpg" }` 即可获得
检测结果列表，默认最多返回 10 条记录。

预测脚本 `scripts/predict.py` 会在检测到单只猪时给出基于框尺寸的粗略长度与体重估计，
该逻辑位于 `utils/estimate.py` 中，可按实际数据调整系数以获得更准确的结果。
示例：

```bash
python scripts/predict.py --image path/to/pig.jpg
```

数据集采用 YOLO v5 标注格式，`utils.dataset.YoloDataset` 会在加载时将相对坐标
转换为像素级的左上角、右下角坐标，以便传入 Faster R-CNN 模型训练。

运行上述脚本时若仅查看 `--help` 信息，可在未安装深度学习依赖的情况下执行。
真正训练或导出模型则需要提前安装 `torch`、`torchvision` 等依赖，确保环境支持 GPU
或 CPU 推理。

> **注意**：本仓库仅提供代码框架，未包含已训练的模型。若要获得可用的检测效果，请在本地安装好 `torch`、`torchvision` 等依赖后自行训练。
