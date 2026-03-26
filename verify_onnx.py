import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from utils import (
    build_val_transform_from_cfg,
    idx_to_class_list,
    list_images,
    load_model_from_ckpt,
    preprocess_image,
    save_json,
    select_device,
)


def run_pytorch(model, x: torch.Tensor):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())
    return pred_idx, pred_conf, probs.cpu().numpy()


def run_onnx(session, x: torch.Tensor):
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x.detach().cpu().numpy()})[0]
    logits = torch.from_numpy(logits)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())
    return pred_idx, pred_conf, probs.cpu().numpy()


def collect_paths(image_path: str, dir_path: str):
    paths = []
    if image_path:
        paths.extend(list_images(image_path))
    if dir_path:
        paths.extend(list_images(dir_path))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("No valid image inputs. Use --image or --dir.")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--conf_tol", type=float, default=1e-3)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("verify_onnx.py 需要 onnxruntime，请先安装后再运行。") from exc

    device = select_device(prefer_cuda=True)
    model, cfg, class_to_idx = load_model_from_ckpt(args.ckpt, device)
    if cfg is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    transform = build_val_transform_from_cfg(cfg)
    class_names = idx_to_class_list(class_to_idx)
    image_paths = collect_paths(args.image, args.dir)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    records = []
    same_top1_count = 0
    conf_close_count = 0

    for image_path in image_paths:
        x = preprocess_image(image_path, transform, device)
        pt_idx, pt_conf, pt_probs = run_pytorch(model, x)
        onnx_idx, onnx_conf, onnx_probs = run_onnx(session, x)

        same_top1 = pt_idx == onnx_idx
        conf_diff = abs(pt_conf - onnx_conf)
        conf_close = conf_diff <= args.conf_tol

        same_top1_count += int(same_top1)
        conf_close_count += int(conf_close)

        record = {
            "image_path": os.path.normpath(image_path),
            "pytorch_top1": class_names[pt_idx],
            "onnx_top1": class_names[onnx_idx],
            "same_top1": same_top1,
            "pytorch_confidence": pt_conf,
            "onnx_confidence": onnx_conf,
            "confidence_diff": conf_diff,
            "confidence_close": conf_close,
            "max_prob_diff": float(np.max(np.abs(pt_probs - onnx_probs))),
        }
        records.append(record)

        print(
            f"{os.path.basename(image_path)} | "
            f"PT={record['pytorch_top1']} ({pt_conf:.6f}) | "
            f"ONNX={record['onnx_top1']} ({onnx_conf:.6f}) | "
            f"same_top1={same_top1} | conf_diff={conf_diff:.6f}"
        )

    summary = {
        "num_images": len(records),
        "same_top1_count": same_top1_count,
        "same_top1_rate": same_top1_count / len(records),
        "confidence_close_count": conf_close_count,
        "confidence_close_rate": conf_close_count / len(records),
        "confidence_tolerance": args.conf_tol,
        "avg_confidence_diff": float(np.mean([r["confidence_diff"] for r in records])),
        "max_confidence_diff": float(np.max([r["confidence_diff"] for r in records])),
        "avg_max_prob_diff": float(np.mean([r["max_prob_diff"] for r in records])),
        "max_max_prob_diff": float(np.max([r["max_prob_diff"] for r in records])),
    }

    print("=" * 60)
    print(f"Images              : {summary['num_images']}")
    print(f"Top1 same rate      : {summary['same_top1_rate']:.4f}")
    print(f"Confidence close    : {summary['confidence_close_rate']:.4f}")
    print(f"Avg confidence diff : {summary['avg_confidence_diff']:.6f}")
    print(f"Max confidence diff : {summary['max_confidence_diff']:.6f}")
    print(f"Avg prob diff       : {summary['avg_max_prob_diff']:.6f}")
    print(f"Max prob diff       : {summary['max_max_prob_diff']:.6f}")
    print("=" * 60)

    if args.out_json:
        save_json({"summary": summary, "records": records}, args.out_json)
        print(f"Saved verification to: {args.out_json}")


if __name__ == "__main__":
    main()
