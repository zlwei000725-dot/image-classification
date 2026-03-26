import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from dataset import build_transforms, CustomImageDataset
from model import build_model


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(ckpt_path: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", None)
        class_to_idx = checkpoint.get("class_to_idx", None)
        return state_dict, cfg, class_to_idx
    raise ValueError("Unsupported checkpoint format")


def resolve_class_to_idx(cfg: dict, class_to_idx: Optional[Dict[str, int]]) -> Dict[str, int]:
    if class_to_idx is not None:
        return class_to_idx
    data_cfg = cfg.get("dataset", {}) if cfg else {}
    temp_dataset = CustomImageDataset(
        data_dir=data_cfg.get("train_dir", "./data/train"),
        transform=None
    )
    return temp_dataset.class_to_idx


def build_val_transform_from_cfg(cfg: dict):
    data_cfg = cfg.get("dataset", {}) if cfg else {}
    model_cfg = cfg.get("model", {}) if cfg else {}
    _, val_tf = build_transforms(
        image_size=data_cfg.get("image_size", 224),
        pretrained=model_cfg.get("pretrained", True),
        aug_mode=data_cfg.get("aug_mode", "basic"),
    )
    return val_tf


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    state_dict, cfg, class_to_idx = load_checkpoint(ckpt_path, device)
    data_cfg = cfg.get("dataset", {}) if cfg else {}
    model_cfg = cfg.get("model", {}) if cfg else {}
    class_to_idx = resolve_class_to_idx(cfg, class_to_idx)
    model = build_model(
        num_classes=len(class_to_idx),
        model_name=model_cfg.get("name", "resnet18"),
        pretrained=False,
        use_se=model_cfg.get("use_se", False),
        se_reduction=model_cfg.get("se_reduction", 16),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg, class_to_idx


def preprocess_image(image_path: str, transform, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0).to(device)


def idx_to_class_list(class_to_idx: Dict[str, int]) -> List[str]:
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def list_images(path: str) -> List[str]:
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    if os.path.isdir(path):
        files = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith(valid_exts):
                files.append(full)
        return files
    if os.path.isfile(path) and path.lower().endswith(valid_exts):
        return [path]
    return []


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
