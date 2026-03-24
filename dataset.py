import os
from typing import Dict, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if class_to_idx is None:
            self.classes = sorted(
                [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            )
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            label = self.class_to_idx[cls_name]
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transforms(
    image_size: int = 224,
    pretrained: bool = True,
    aug_mode: str = "basic"
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    baseline 先只保留最基础增强：
    - train: Resize + HorizontalFlip + Normalize
    - val:   Resize + Normalize

    后面 Day4 再做 Rotation / ColorJitter 对比实验。
    """
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    train_tf = [transforms.Resize((image_size,image_size))]

    if aug_mode == "basic":
        train_tf += [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    elif aug_mode == "rot":
        train_tf += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    elif aug_mode == "color":
        train_tf +=[
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
        ]
    else:
        raise ValueError(f"Unsupported aug_mode: {aug_mode}")
    train_tf+= [transforms.ToTensor(),
                transforms.Normalize(mean,std),
                ]
    val_tf = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    return transforms.Compose(train_tf), val_tf


if __name__ == "__main__":
    train_transform, _ = build_transforms(image_size=224, pretrained=True)

    dataset = CustomImageDataset(
        data_dir="./data/train",
        transform=train_transform
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class mapping: {dataset.class_to_idx}")

    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image tensor shape: {img.shape}")
        print(f"Label: {label}")
