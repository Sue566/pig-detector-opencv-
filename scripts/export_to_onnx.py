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


def load_config(path: str):
    if yaml is None:
        raise RuntimeError(
            "pyyaml is not installed. Please install dependencies from requirements.txt"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def export(cfg_path: str, weights: str, output: str):
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')
    from utils.model import create_model
    cfg = load_config(cfg_path)
    model = create_model(cfg['num_classes'] + 1)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros(1, 3, 640, 640)
    torch.onnx.export(model, dummy, output, opset_version=11)
    print(f'ONNX model saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--weights', default='models/best_model.pth')
    parser.add_argument('--output', default='models/model.onnx')
    args = parser.parse_args()
    export(args.config, args.weights, args.output)
