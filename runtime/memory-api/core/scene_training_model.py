from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class SceneRepresentationConfig:
    scene_count: int
    style_count: int
    source_family_count: int
    use_count: int
    class_count: int
    image_size: int = 256
    width: int = 48
    embed_dim: int = 160
    dropout: float = 0.1


if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.block(x)


    class SceneRepresentationNet(nn.Module):
        def __init__(self, config: SceneRepresentationConfig):
            super().__init__()
            self.config = config
            width = config.width
            self.stem = nn.Sequential(
                nn.Conv2d(1, width, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
            )
            self.encoder = nn.Sequential(
                ConvBlock(width, width),
                ConvBlock(width, width * 2, stride=2),
                ConvBlock(width * 2, width * 2),
                ConvBlock(width * 2, width * 4, stride=2),
                ConvBlock(width * 4, width * 4),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.backbone_head = nn.Sequential(
                nn.Linear(width * 4, config.embed_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.scene_head = nn.Linear(config.embed_dim, config.scene_count)
            self.style_head = nn.Linear(config.embed_dim, config.style_count)
            self.source_family_head = nn.Linear(config.embed_dim, config.source_family_count)
            self.use_head = nn.Linear(config.embed_dim, config.use_count)
            self.class_presence_head = nn.Linear(config.embed_dim, config.class_count)
            self.component_count_head = nn.Linear(config.embed_dim, 1)
            self.mapped_component_count_head = nn.Linear(config.embed_dim, 1)

        def encode(self, image):
            features = self.stem(image)
            features = self.encoder(features)
            pooled = self.pool(features).flatten(1)
            return self.backbone_head(pooled)

        def forward(self, image):
            embedding = self.encode(image)
            return {
                "embedding": embedding,
                "scene_logits": self.scene_head(embedding),
                "style_logits": self.style_head(embedding),
                "source_family_logits": self.source_family_head(embedding),
                "use_logits": self.use_head(embedding),
                "class_presence_logits": self.class_presence_head(embedding),
                "component_count": self.component_count_head(embedding).squeeze(-1),
                "mapped_component_count": self.mapped_component_count_head(embedding).squeeze(-1),
            }
else:  # pragma: no cover
    class SceneRepresentationNet:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use SceneRepresentationNet")
