import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
import yaml

from utils import (
    select_device,
    load_model_from_ckpt,
    build_val_transform_from_cfg,
    preprocess_image,
    idx_to_class_list,
    list_images,
    save_json,
)


def predict_paths(model, transform, device, paths: List[str], class_names: List[str], topk: int = 1):
    records = []
    with torch.no_grad():
        for p in paths:
            x = preprocess_image(p, transform, device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            confs, indices = torch.topk(probs, k=min(topk, probs.numel()))
            confs = confs.tolist()
            indices = indices.tolist()
            labels = [class_names[i] for i in indices]
            record = {
                "path": os.path.normpath(p),
                "topk_labels": labels,
                "topk_confs": confs,
                "pred_label": labels[0],
                "confidence": confs[0],
            }
            records.append(record)
            print(f"{os.path.basename(p)} -> {labels[0]} ({confs[0]:.4f})")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    device = select_device(prefer_cuda=True)

    model, cfg, class_to_idx = load_model_from_ckpt(args.ckpt, device)
    if cfg is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    transform = build_val_transform_from_cfg(cfg)
    class_names = idx_to_class_list(class_to_idx)

    targets = []
    if args.image:
        targets += list_images(args.image)
    if args.dir:
        targets += list_images(args.dir)
    if not targets:
        raise RuntimeError("No valid image inputs. Use --image or --dir.")

    records = predict_paths(model, transform, device, targets, class_names, topk=args.topk)

    if args.out_json:
        save_json(records, args.out_json)
        print(f"Saved predictions to: {args.out_json}")


if __name__ == "__main__":
    main()
