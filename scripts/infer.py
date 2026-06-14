import argparse
import torch
from PIL import Image

from src.config import load_config
from src.model import build_model
from src.pre.transform import get_transform
from src.pre.datasets.cifar10 import CIFAR10Adapter


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to the training config"
    )
    parser.add_argument(
        "--mode", default="full_finetune", help="Training mode used"
    )
    args = parser.parse_args()

    # Load config (this handles num_classes implicitly if we do it right, 
    # but since train.py does it via adapter, we'll set it here)
    cfg = load_config(args.config)
    cfg["dataset"]["num_classes"] = 10  # Hardcoded for CIFAR-10 for now

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and load weights
    model = build_model(cfg, mode=args.mode)
    
    # weights_only=True is recommended for safety in modern PyTorch
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # Load and transform image
    try:
        img = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Use 'val' transform (just resize, center crop, normalize)
    transform = get_transform(img_size=cfg["img_size"], mode="val")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
            output = model(img_tensor)
            # Safe softmax handling like in trainer
            probs = torch.softmax(output.float(), dim=1).squeeze(0).cpu().numpy()

    # Get Top-3 Predictions
    class_names = CIFAR10Adapter.CLASS_NAMES
    top3_idx = probs.argsort()[-3:][::-1]

    print("\n" + "=" * 40)
    print("🎯 PREDICTIONS:")
    print("=" * 40)
    for idx in top3_idx:
        print(f"{class_names[idx].capitalize():<15}: {probs[idx]*100:>6.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
