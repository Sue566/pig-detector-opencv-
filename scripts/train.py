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

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.ops import misc as misc_nn_ops
except ImportError:  # pragma: no cover - handled at runtime
    torch = None



def load_config(path: str):
    if yaml is None:
        raise RuntimeError(
            "pyyaml is not installed. Please install dependencies from requirements.txt"
        )
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(args):
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Please install dependencies from requirements.txt"
        )
    from utils.dataset import YoloDataset
    from utils.model import create_model
    from utils.transforms import get_train_transforms, get_val_transforms
    cfg = load_config(args.config)

    train_ds = YoloDataset(cfg['train_dir'], transforms=get_train_transforms())
    val_ds = YoloDataset(cfg['val_dir'], transforms=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg['num_classes'] + 1)  # +1 for background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['learning_rate'], momentum=0.9, weight_decay=0.0005)

    for epoch in range(cfg['num_epochs']):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done, loss: {losses.item():.4f}")

    Path(cfg['model_dir']).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(cfg['model_dir']) / 'best_model.pth')
    print('Training finished, model saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pig detector')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    train(args)
