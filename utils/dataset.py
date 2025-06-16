import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class YoloDataset(Dataset):
    """读取YOLO格式标注的数据集"""

    def __init__(self, root: str, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.imgs = sorted((self.root / "images").glob("*.jpg"))
        self.labels = sorted((self.root / "labels").glob("*.txt"))
        assert len(self.imgs) == len(self.labels), "Image/label count mismatch"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, dict]:
        img_path = self.imgs[idx]
        label_path = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        boxes: List[List[float]] = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # YOLO 格式: class cx cy w h (相对值)
                cx, cy, w, h = map(float, parts[1:])
                # 转为左上角-右下角绝对坐标
                x1 = (cx - w / 2) * img.width
                y1 = (cy - h / 2) * img.height
                x2 = (cx + w / 2) * img.width
                y2 = (cy + h / 2) * img.height
                boxes.append([x1, y1, x2, y2])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target
