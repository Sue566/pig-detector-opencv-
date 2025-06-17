import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

torch = None
Image = None
F = None
estimate_length_weight = None


def load_config(path: str):
    if yaml is None:
        raise RuntimeError(
            "pyyaml is not installed. Please install dependencies from requirements.txt"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(cfg_path: str, weight_path: str):
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')
    from utils.model import create_model
    cfg = load_config(cfg_path)
    model = create_model(cfg['num_classes'] + 1)
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(args):
    global Image, F, estimate_length_weight
    if Image is None or F is None:
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF
        Image = PILImage
        F = TF
    if estimate_length_weight is None:
        from utils.estimate import estimate_length_weight as elw
        estimate_length_weight = elw
    model = load_model(args.config, args.weights)
    img = Image.open(args.image).convert('RGB')
    tensor = F.to_tensor(img)
    outputs = model([tensor])[0]
    boxes = outputs['boxes'].detach().numpy()
    scores = outputs['scores'].detach().numpy()

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box
        length, weight = estimate_length_weight((x1, y1, x2, y2))
        print(
            f"pig detected at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], length={length:.1f}, weight~{weight:.1f}kg"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--weights', default='models/best_model.pth')
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    main(args)
