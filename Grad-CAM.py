import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from utils import (
    select_device,
    load_model_from_ckpt,
    build_val_transform_from_cfg,
    preprocess_image,
    idx_to_class_list,
)


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
        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs[0, class_idx].item()


def overlay_heatmap_on_image(heatmap: np.ndarray, pil_image: Image.Image, alpha: float = 0.4):
    heatmap_img = plt.get_cmap("jet")(heatmap)[..., :3]
    heatmap_img = (heatmap_img * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_img).resize(pil_image.size, Image.BILINEAR)
    blended = Image.blend(pil_image.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=alpha)
    return blended.convert("RGB")


def run_cam(ckpt_path: str, image_path: str, device: torch.device):
    model, cfg, class_to_idx = load_model_from_ckpt(ckpt_path, device)
    if cfg is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    transform = build_val_transform_from_cfg(cfg)
    class_names = idx_to_class_list(class_to_idx)
    x = preprocess_image(image_path, transform, device)
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model, target_layer)
    heatmap, class_idx, conf = cam.generate(x)
    cam.remove()
    pred_label = class_names[class_idx]
    pil_img = Image.open(image_path).convert("RGB")
    overlay = overlay_heatmap_on_image(heatmap, pil_img, alpha=0.45)
    return overlay, heatmap, pred_label, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_base", type=str, default=None)
    parser.add_argument("--ckpt_se", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs/figures")
    args = parser.parse_args()

    if not args.ckpt_base and not args.ckpt_se:
        raise RuntimeError("Provide at least one of --ckpt_base or --ckpt_se")

    device = select_device(prefer_cuda=True)
    os.makedirs(args.out_dir, exist_ok=True)

    base_overlay = None
    base_heat = None
    base_label = None
    base_conf = None
    se_overlay = None
    se_heat = None
    se_label = None
    se_conf = None

    if args.ckpt_base:
        base_overlay, base_heat, base_label, base_conf = run_cam(args.ckpt_base, args.image, device)
    if args.ckpt_se:
        se_overlay, se_heat, se_label, se_conf = run_cam(args.ckpt_se, args.image, device)

    if args.ckpt_base and args.ckpt_se:
        diff = None
        if base_heat is not None and se_heat is not None:
            h = base_heat.shape[0]
            w = base_heat.shape[1]
            bh = base_heat
            sh = se_heat
            diff = (sh - bh)
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        pil_img = Image.open(args.image).convert("RGB")
        fig, axes = plt.subplots(1, 4 if diff is not None else 3, figsize=(16, 4))
        axes = axes.flatten()
        axes[0].imshow(pil_img)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(base_overlay if base_overlay is not None else pil_img)
        t1 = f"Baseline: {base_label} ({base_conf:.3f})" if base_label else "Baseline"
        axes[1].set_title(t1)
        axes[1].axis("off")
        axes[2].imshow(se_overlay if se_overlay is not None else pil_img)
        t2 = f"SE: {se_label} ({se_conf:.3f})" if se_label else "SE"
        axes[2].set_title(t2)
        axes[2].axis("off")
        if diff is not None:
            diff_img = overlay_heatmap_on_image(diff, pil_img, alpha=0.45)
            axes[3].imshow(diff_img)
            axes[3].set_title("SE - Baseline")
            axes[3].axis("off")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"gradcam_compare_{os.path.splitext(os.path.basename(args.image))[0]}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        overlay, _, label, conf = (base_overlay, base_heat, base_label, base_conf) if args.ckpt_base else (se_overlay, se_heat, se_label, se_conf)
        pil_img = Image.open(args.image).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(overlay if overlay is not None else pil_img)
        axes[1].set_title(f"{label} ({conf:.3f})" if label else "Grad-CAM")
        axes[1].axis("off")
        plt.tight_layout()
        tag = "base" if args.ckpt_base else "se"
        out_path = os.path.join(args.out_dir, f"gradcam_{tag}_{os.path.splitext(os.path.basename(args.image))[0]}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
