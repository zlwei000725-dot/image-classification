import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomImageDataset, build_transforms
from model import build_model


def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", None)
        class_to_idx = checkpoint.get("class_to_idx", None)
    else:
        raise ValueError(
            "Unsupported checkpoint format. "
            "Please use the checkpoint saved by the current train.py."
        )

    return state_dict, cfg, class_to_idx


@torch.no_grad()
def evaluate(model, loader, dataset, device, idx_to_class):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    wrong_records = []

    offset = 0  # because val_loader shuffle=False, order matches dataset.image_paths

    pbar = tqdm(loader, desc="[Eval]")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        batch_paths = dataset.image_paths[offset: offset + batch_size]
        offset += batch_size

        for i in range(batch_size):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()

            y_true.append(true_idx)
            y_pred.append(pred_idx)

            if true_idx != pred_idx:
                wrong_records.append({
                    "path": batch_paths[i],
                    "true_label": idx_to_class[true_idx],
                    "pred_label": idx_to_class[pred_idx],
                    "confidence": float(probs[i, pred_idx].item())
                })

    avg_loss = total_loss / total_samples
    return avg_loss, y_true, y_pred, wrong_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # -------------------------
    # 1. device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------
    # 2. load ckpt
    # -------------------------
    state_dict, cfg, class_to_idx = load_checkpoint(args.ckpt, device)

    if cfg is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    if class_to_idx is None:
        # fallback: build from train dir
        temp_dataset = CustomImageDataset(
            data_dir=data_cfg.get("train_dir", "./data/train"),
            transform=None
        )
        class_to_idx = temp_dataset.class_to_idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # -------------------------
    # 3. dataset / dataloader
    # -------------------------
    _, val_transform = build_transforms(
        image_size=data_cfg.get("image_size", 224),
        pretrained=model_cfg.get("pretrained", True)
    )

    val_dataset = CustomImageDataset(
        data_dir=data_cfg.get("val_dir", "./data/val"),
        transform=val_transform,
        class_to_idx=class_to_idx
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Val images: {len(val_dataset)}")
    print(f"Class mapping: {class_to_idx}")

    # -------------------------
    # 4. build model
    # -------------------------
    model = build_model(
        num_classes=len(class_to_idx),
        model_name=model_cfg.get("name", "resnet18"),
        pretrained=False,
        use_se=model_cfg.get("use_se", False),
        se_reduction=model_cfg.get("se_reduction", 16)
    ).to(device)

    model.load_state_dict(state_dict, strict=True)

    # -------------------------
    # 5. evaluate
    # -------------------------
    avg_loss, y_true, y_pred, wrong_records = evaluate(
        model, val_loader, val_dataset, device, idx_to_class
    )

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    overall_acc = np.trace(cm) / np.sum(cm)

    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        acc = class_correct / class_total if class_total > 0 else 0.0
        per_class_acc[class_name] = acc

    # -------------------------
    # 6. save outputs
    # -------------------------
    os.makedirs("./outputs/figures", exist_ok=True)

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    cm_path = os.path.join("./outputs/figures", f"{ckpt_name}_confusion_matrix.png")
    report_path = os.path.join("./outputs/figures", f"{ckpt_name}_eval_report.txt")
    wrong_path = os.path.join("./outputs/figures", f"{ckpt_name}_wrong_predictions.json")

    plot_confusion_matrix(cm, class_names, cm_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Average loss: {avg_loss:.6f}\n")
        f.write(f"Overall accuracy: {overall_acc:.6f}\n")
        f.write("Per-class accuracy:\n")
        for class_name, acc in per_class_acc.items():
            f.write(f"  - {class_name}: {acc:.6f}\n")
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")

    with open(wrong_path, "w", encoding="utf-8") as f:
        json.dump(wrong_records, f, ensure_ascii=False, indent=2)

    # -------------------------
    # 7. print summary
    # -------------------------
    print("=" * 60)
    print(f"Average loss      : {avg_loss:.6f}")
    print(f"Overall accuracy  : {overall_acc:.6f}")
    print("Per-class accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  - {class_name}: {acc:.6f}")

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Eval report saved to     : {report_path}")
    print(f"Wrong samples saved to   : {wrong_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
