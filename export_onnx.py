import argparse
import os

import torch
import yaml

from utils import load_model_from_ckpt, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    device = select_device(prefer_cuda=True)

    model, cfg, class_to_idx = load_model_from_ckpt(args.ckpt, device)
    if cfg is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    model.eval()
    data_cfg = cfg.get("dataset", {}) if cfg else {}
    h = w = args.image_size or data_cfg.get("image_size", 224)
    dummy = torch.randn(1, 3, h, w, device=device)

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        out_dir = os.path.dirname(args.ckpt) or "."
        out_path = os.path.join(out_dir, f"{base}.onnx")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX to: {out_path}")


if __name__ == "__main__":
    main()
