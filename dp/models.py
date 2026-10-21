"""ResNet-18 in its CIFAR form (3x3 stem, no max-pool) with nn.Dropout(p) before the classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(block_dropout) if block_dropout > 0 else nn.Identity()  # E2: channel-wise, between the two convs
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.0, where="head"):
        """where='head': nn.Dropout(p) before the classifier (main study).
        where='block': nn.Dropout2d(p) inside every residual block (extension E2); head dropout is then 0."""
        super().__init__()
        assert where in ("head", "block")
        self.block_p = dropout_p if where == "block" else 0.0
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.dropout = nn.Dropout(dropout_p if where == "head" else 0.0)  # p=0 is the dense control
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s, self.block_p))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(self.dropout(out))


def prunable_params(model):
    """(name, weight) for every Conv2d / Linear weight. BatchNorm and biases are never pruned."""
    return [(n + ".weight", m.weight) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def count_params(model):
    return sum(p.numel() for p in model.parameters())
