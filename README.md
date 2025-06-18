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
1. 执行 `./start_train.sh`（首次运行会创建虚拟环境并安装依赖，脚本已使用
   清华镜像源加速安装，可按需修改）。
2. 在 `config.yaml` 中配置数据集路径和训练参数。
3. 训练结束后模型会保存在 `models/` 目录。
4. 如需在 Java 中使用，可执行 `python scripts/export_to_onnx.py` 导出 ONNX 模型。
5. 若希望通过 Docker 构建环境，可运行 `./build_docker.sh` 生成镜像。该脚本
   同样默认使用清华镜像安装依赖，构建完成后可通过
   `docker run -p 8000:8000 pig-detector` 启动 API 服务。

如需通过 HTTP 调用模型，可运行 `./start_api.sh` 启动 FastAPI 服务：

```bash
./start_api.sh
```

启动后向 `POST /api/predict` 发送如下 JSON 即可获得检测结果：

```json
{
  "image_path": "path/to/img.jpg",
  "conf": 0.5,      // 置信度阈值，可选
  "top_k": 10       // 最多返回多少条结果，可选
}
```

接口会返回 `type` 字段，若检测到猪则为 `pig`，否则为 `other`，并在仅检测到单只猪时给出长度和体重估计。
通过 `GET /api/version` 可以查看模型版本及训练时间信息。

预测脚本 `scripts/predict.py` 会在检测到单只猪时给出基于框尺寸的粗略长度与体重估计，
该逻辑位于 `utils/estimate.py` 中，可按实际数据调整系数以获得更准确的结果。
示例：

```bash
python scripts/predict.py --image path/to/pig.jpg
```
若未检测到猪，脚本会打印 `Image does not contain pigs.` 以便区分无结果的情况。

数据集采用 YOLO v5 标注格式，`utils.dataset.YoloDataset` 会在加载时将相对坐标
转换为像素级的左上角、右下角坐标，以便传入 Faster R-CNN 模型训练。

运行上述脚本时若仅查看 `--help` 信息，可在未安装深度学习依赖的情况下执行。
真正训练或导出模型则需要提前安装 `torch`、`torchvision` 等依赖，确保环境支持 GPU
或 CPU 推理。

> **注意**：本仓库仅提供代码框架，未包含已训练的模型。若要获得可用的检测效果，请在本地安装好 `torch`、`torchvision` 等依赖后自行训练。
