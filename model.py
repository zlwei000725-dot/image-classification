import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_dim = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    """
    用现成 BasicBlock 包一层 SE。
    这样如果 pretrained=True，原 resnet18 的卷积和 BN 权重仍能保留，
    新增的 SE 层随机初始化即可。
    """
    expansion = 1

    def __init__(self, block: BasicBlock, reduction: int = 16):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride

        self.se = SEBlock(self.conv2.out_channels, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def replace_basic_blocks_with_se(module: nn.Module, reduction: int = 16):
    """
    递归替换 resnet18 中所有 BasicBlock 为 SEBasicBlock
    """
    for name, child in list(module.named_children()):
        if isinstance(child, BasicBlock):
            setattr(module, name, SEBasicBlock(child, reduction=reduction))
        else:
            replace_basic_blocks_with_se(child, reduction=reduction)


def build_model(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
    use_se: bool = False,
    se_reduction: int = 16
):
    if model_name.lower() != "resnet18":
        raise ValueError("Current baseline only supports 'resnet18'.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if use_se:
        replace_basic_blocks_with_se(model, reduction=se_reduction)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_model(
        num_classes=2,
        model_name="resnet18",
        pretrained=False,
        use_se=True
    )
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # torch.Size([2, 2])
