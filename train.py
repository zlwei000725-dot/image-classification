import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CustomImageDataset, build_transforms
from model import build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, optimizer_cfg, lr, weight_decay):
    name = optimizer_cfg.get("name", "adam").lower()

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, scheduler_cfg, epochs):
    name = scheduler_cfg.get("name", "none").lower()

    if name == "step":
        step_size = scheduler_cfg.get("step_size", 10)
        gamma = scheduler_cfg.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if name in ["none", "", "null"]:
        return None

    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Val]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # =========================
    # 1. Load config
    # =========================
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})

    # =========================
    # 2. Basic setup
    # =========================
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    save_dir = train_cfg.get("save_dir", "./outputs/checkpoints")
    log_dir = train_cfg.get("log_dir", "./outputs/logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("patience", 0)

    image_size = data_cfg.get("image_size", 224)
    num_workers = data_cfg.get("num_workers", 4)

    pretrained = model_cfg.get("pretrained", True)
    model_name = model_cfg.get("name", "resnet18")
    use_se = model_cfg.get("use_se", False)
    se_reduction = model_cfg.get("se_reduction", 16)

    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Model         : {model_name}")
    print(f"Use SE        : {use_se}")
    print("=" * 60)

    # =========================
    # 3. Dataset & DataLoader
    # =========================
    aug_mode = data_cfg.get("aug_mode", "basic")

    train_transform, val_transform = build_transforms(
        image_size=image_size,
        pretrained=pretrained,
        aug_mode=aug_mode
    )
    print(f"Aug mode      : {aug_mode}")

    train_dataset = CustomImageDataset(
        data_dir=data_cfg.get("train_dir", "./data/train"),
        transform=train_transform
    )

    val_dataset = CustomImageDataset(
        data_dir=data_cfg.get("val_dir", "./data/val"),
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")
    if len(val_dataset) == 0:
        raise RuntimeError("Val dataset is empty.")

    expected_num_classes = data_cfg.get("num_classes", len(train_dataset.classes))
    if expected_num_classes != len(train_dataset.classes):
        raise ValueError(
            f"num_classes mismatch: config={expected_num_classes}, "
            f"dataset={len(train_dataset.classes)}"
        )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train images  : {len(train_dataset)}")
    print(f"Val images    : {len(val_dataset)}")
    print(f"Class mapping : {train_dataset.class_to_idx}")

    # smoke test: check one batch
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Batch image shape : {sample_images.shape}")
    print(f"Batch label shape : {sample_labels.shape}")

    # =========================
    # 4. Build model
    # =========================
    model = build_model(
        num_classes=len(train_dataset.classes),
        model_name=model_name,
        pretrained=pretrained,
        use_se=use_se,
        se_reduction=se_reduction
    ).to(device)

    with torch.no_grad():
        sample_outputs = model(sample_images[:2].to(device))
    print(f"Model output shape: {sample_outputs.shape}")

    # =========================
    # 5. Loss / Optimizer / Scheduler
    # =========================
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_cfg, lr, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_cfg, epochs)

    # =========================
    # 6. Logging
    # =========================
    # run_name = f"{model_name}_{'se' if use_se else 'baseline'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_name = (
        f"{model_name}_"
        f"{'se' if use_se else 'baseline'}_"
        f"{aug_mode}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=run_dir)

    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with open(os.path.join(run_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "epoch", "lr",
            "train_loss", "train_acc",
            "val_loss", "val_acc"
        ])

    # =========================
    # 7. Train loop
    # =========================
    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve = 0
    latest_path = os.path.join(save_dir, f"{run_name}_latest_model.pth")
    best_path = os.path.join(save_dir, f"{run_name}_best_model.pth")

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, epochs
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Acc/train", train_acc, epoch)
        tb_writer.add_scalar("Acc/val", val_acc, epoch)
        tb_writer.add_scalar("LR", current_lr, epoch)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch, current_lr,
                train_loss, train_acc,
                val_loss, val_acc
            ])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "class_to_idx": train_dataset.class_to_idx,
            "config": cfg,
        }

        torch.save(checkpoint, latest_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, best_path)
            print(f"New best model saved. Best Val Acc = {best_val_acc:.4f}")

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss = {best_val_loss:.4f}")
                break

    tb_writer.close()

    print("=" * 60)
    print("Training finished.")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint  : {best_path}")
    print(f"TensorBoard log  : {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
