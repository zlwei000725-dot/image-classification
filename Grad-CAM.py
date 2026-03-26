import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from utils import (
    build_val_transform_from_cfg,
    idx_to_class_list,
    list_images,
    load_model_from_ckpt,
    preprocess_image,
    save_json,
    select_device,
)


MAX_ANALYZE_IMAGES = 8


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target = target_module
        self.handles = []
        self.activations = None
        self.gradients = None
        self.handles.append(self.target.register_forward_hook(self._forward_hook))
        self.handles.append(self.target.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self.handles:
            h.remove()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        if class_idx is None:
            class_idx = pred_idx
        if class_idx < 0 or class_idx >= probs.shape[1]:
            raise ValueError(f"class_idx out of range: {class_idx}")

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return {
            "heatmap": cam.cpu().numpy(),
            "pred_idx": pred_idx,
            "pred_conf": probs[0, pred_idx].item(),
            "target_idx": class_idx,
            "target_conf": probs[0, class_idx].item(),
        }


def load_cfg_if_needed(cfg):
    if cfg is not None:
        return cfg
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target_idx(
    target_mode: str,
    true_label: Optional[str],
    image_path: str,
    class_to_idx: Dict[str, int],
    pred_idx: int,
):
    if target_mode == "pred":
        return pred_idx

    label_name = true_label
    if label_name is None:
        parent_name = os.path.basename(os.path.dirname(image_path))
        if parent_name in class_to_idx:
            label_name = parent_name

    if label_name is None:
        raise RuntimeError("target_mode=true 时必须提供 --true_label，或将图片放在类别目录下。")
    if label_name not in class_to_idx:
        raise ValueError(f"Unknown true_label: {label_name}. Available: {list(class_to_idx.keys())}")
    return class_to_idx[label_name]


def collect_image_paths(image_path: Optional[str], dir_path: Optional[str]) -> List[str]:
    paths = []
    if image_path:
        paths.extend(list_images(image_path))
    if dir_path:
        paths.extend(list_images(dir_path))
    deduped = list(dict.fromkeys(paths))
    if not deduped:
        raise RuntimeError("No valid images found. Use --image or --dir.")
    return deduped[:MAX_ANALYZE_IMAGES]


def infer_target_result(
    model: torch.nn.Module,
    transform,
    class_to_idx: Dict[str, int],
    image_path: str,
    target_mode: str,
    true_label: Optional[str],
    device: torch.device,
):
    class_names = idx_to_class_list(class_to_idx)
    x = preprocess_image(image_path, transform, device)
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    initial = cam.generate(x)
    target_idx = resolve_target_idx(
        target_mode=target_mode,
        true_label=true_label,
        image_path=image_path,
        class_to_idx=class_to_idx,
        pred_idx=initial["pred_idx"],
    )
    result = initial if target_idx == initial["target_idx"] else cam.generate(x, class_idx=target_idx)
    cam.remove()

    result["pred_label"] = class_names[result["pred_idx"]]
    result["target_label"] = class_names[result["target_idx"]]
    result["image_path"] = os.path.normpath(image_path)
    return result


def run_cam(
    ckpt_path: str,
    device: torch.device,
):
    model, cfg, class_to_idx = load_model_from_ckpt(ckpt_path, device)
    cfg = load_cfg_if_needed(cfg)
    transform = build_val_transform_from_cfg(cfg)
    return model, transform, class_to_idx


def draw_panel(ax, image=None, heatmap=None, title="", cmap=None, norm=None, vmin=None, vmax=None):
    if image is not None:
        ax.imshow(image)
    else:
        ax.imshow(heatmap, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def build_row_title(prefix: str, result: Dict[str, object]) -> str:
    return (
        f"{prefix}\n"
        f"pred={result['pred_label']} ({result['pred_conf']:.3f})\n"
        f"target={result['target_label']} ({result['target_conf']:.3f})"
    )


def save_batch_figure(
    image_paths: List[str],
    base_results: Optional[List[Dict[str, object]]],
    se_results: Optional[List[Dict[str, object]]],
    out_path: str,
):
    rows = len(image_paths)
    compare_mode = base_results is not None and se_results is not None
    cols = 4 if compare_mode else 2
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = np.array([axes])

    diff_max_abs = 1.0
    if compare_mode:
        diffs = [se["heatmap"] - base["heatmap"] for base, se in zip(base_results, se_results)]
        diff_max_abs = max(max(float(np.abs(diff).max()) for diff in diffs), 1e-6)
        diff_norm = TwoSlopeNorm(vmin=-diff_max_abs, vcenter=0.0, vmax=diff_max_abs)
    else:
        diff_norm = None

    for row_idx, image_path in enumerate(image_paths):
        pil_img = Image.open(image_path).convert("RGB")
        row_axes = axes[row_idx]
        base_result = base_results[row_idx] if base_results is not None else None
        se_result = se_results[row_idx] if se_results is not None else None

        draw_panel(row_axes[0], image=pil_img, title=os.path.basename(image_path))

        if compare_mode:
            draw_panel(
                row_axes[1],
                heatmap=base_result["heatmap"],
                title=build_row_title("Baseline", base_result),
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )
            draw_panel(
                row_axes[2],
                heatmap=se_result["heatmap"],
                title=build_row_title("SE", se_result),
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )
            diff = se_result["heatmap"] - base_result["heatmap"]
            draw_panel(
                row_axes[3],
                heatmap=diff,
                title="SE - Baseline",
                cmap="coolwarm",
                norm=diff_norm,
            )
        else:
            result = base_result if base_result is not None else se_result
            model_name = "Baseline" if base_result is not None else "SE"
            draw_panel(
                row_axes[1],
                heatmap=result["heatmap"],
                title=build_row_title(model_name, result),
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_base", type=str, default=None)
    parser.add_argument("--ckpt_se", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--target_mode", type=str, default="pred", choices=["pred", "true"])
    parser.add_argument("--true_label", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./outputs/figures")
    args = parser.parse_args()

    if not args.ckpt_base and not args.ckpt_se:
        raise RuntimeError("Provide at least one of --ckpt_base or --ckpt_se")

    device = select_device(prefer_cuda=True)
    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = collect_image_paths(args.image, args.dir)
    print(f"Analyze {len(image_paths)} images (max {MAX_ANALYZE_IMAGES})")

    base_results = None
    se_results = None

    if args.ckpt_base:
        base_model, base_transform, base_class_to_idx = run_cam(args.ckpt_base, device)
        base_results = [
            infer_target_result(
                model=base_model,
                transform=base_transform,
                class_to_idx=base_class_to_idx,
                image_path=image_path,
                target_mode=args.target_mode,
                true_label=args.true_label,
                device=device,
            )
            for image_path in image_paths
        ]
    if args.ckpt_se:
        se_model, se_transform, se_class_to_idx = run_cam(args.ckpt_se, device)
        se_results = [
            infer_target_result(
                model=se_model,
                transform=se_transform,
                class_to_idx=se_class_to_idx,
                image_path=image_path,
                target_mode=args.target_mode,
                true_label=args.true_label,
                device=device,
            )
            for image_path in image_paths
        ]

    summary = []
    for idx, image_path in enumerate(image_paths):
        item = {"image_path": os.path.normpath(image_path)}
        if base_results is not None:
            item["baseline"] = {
                "pred_label": base_results[idx]["pred_label"],
                "pred_conf": base_results[idx]["pred_conf"],
                "target_label": base_results[idx]["target_label"],
                "target_conf": base_results[idx]["target_conf"],
            }
        if se_results is not None:
            item["se"] = {
                "pred_label": se_results[idx]["pred_label"],
                "pred_conf": se_results[idx]["pred_conf"],
                "target_label": se_results[idx]["target_label"],
                "target_conf": se_results[idx]["target_conf"],
            }
        summary.append(item)

    tag = "compare" if args.ckpt_base and args.ckpt_se else ("baseline" if args.ckpt_base else "se")
    suffix = "pred" if args.target_mode == "pred" else f"true_{args.true_label or 'auto'}"
    fig_path = os.path.join(args.out_dir, f"gradcam_batch8_{tag}_{suffix}.png")
    json_path = os.path.join(args.out_dir, f"gradcam_batch8_{tag}_{suffix}.json")

    save_batch_figure(image_paths, base_results, se_results, fig_path)
    save_json(summary, json_path)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
