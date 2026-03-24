import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image


def normalize_path(path_str: str) -> str:
    path_str = path_str.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(path_str)


def save_grid(records, save_path, cols=4, max_items=16):
    records = records[:max_items]
    rows = (len(records) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, item in zip(axes, records):
        img_path = normalize_path(item["path"])
        if os.path.isfile(img_path):
            image = Image.open(img_path).convert("RGB")
            ax.imshow(image)
        ax.set_title(
            f'{item["true_label"]} -> {item["pred_label"]}\nconf={item["confidence"]:.4f}',
            fontsize=10
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="./outputs/error_cases")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    records = sorted(records, key=lambda x: x["confidence"], reverse=True)

    os.makedirs(args.out_dir, exist_ok=True)

    cats_as_dogs = [r for r in records if r["true_label"] == "cats" and r["pred_label"] == "dogs"]
    dogs_as_cats = [r for r in records if r["true_label"] == "dogs" and r["pred_label"] == "cats"]

    for sub_name, sub_records in [
        ("cats_as_dogs", cats_as_dogs),
        ("dogs_as_cats", dogs_as_cats),
    ]:
        sub_dir = os.path.join(args.out_dir, sub_name)
        os.makedirs(sub_dir, exist_ok=True)

        for item in sub_records[:args.topk]:
            src = normalize_path(item["path"])
            if os.path.isfile(src):
                dst = os.path.join(sub_dir, os.path.basename(src))
                shutil.copy2(src, dst)

        grid_path = os.path.join(args.out_dir, f"{sub_name}_top{min(args.topk, len(sub_records))}.png")
        save_grid(sub_records, grid_path, cols=4, max_items=min(args.topk, len(sub_records)))

    summary_path = os.path.join(args.out_dir, "error_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total wrong samples: {len(records)}\n")
        f.write(f"cats -> dogs: {len(cats_as_dogs)}\n")
        f.write(f"dogs -> cats: {len(dogs_as_cats)}\n\n")

        f.write("Top 10 wrong samples by confidence:\n")
        for item in records[:10]:
            f.write(
                f'{item["path"]} | {item["true_label"]} -> {item["pred_label"]} '
                f'| conf={item["confidence"]:.6f}\n'
            )

    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
