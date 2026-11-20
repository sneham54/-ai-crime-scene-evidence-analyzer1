import torch
from torch import nn

from .attention import TemporalMultiHeadSelfAttention
from .config import ModelConfig
from .convlstm import BidirectionalConvLSTM
from .densenet3d import DenseNet3D


class DenseNetAttentionConvLSTM(nn.Module):
    """DenseNet-3D -> temporal MHSA -> bidirectional ConvLSTM -> 4 logits."""
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.backbone = DenseNet3D(growth_rate=cfg.growth_rate, block_layers=cfg.block_layers,
                                   initial_features=cfg.initial_features)
        self.attention = TemporalMultiHeadSelfAttention(self.backbone.out_channels, cfg.attention_heads)
        self.convlstm = BidirectionalConvLSTM(self.backbone.out_channels, cfg.convlstm_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.convlstm_hidden * 2, cfg.num_classes)

    def forward(self, video):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("video input must have shape (batch, time, 3, height, width)")
        features = self.backbone(video.permute(0, 2, 1, 3, 4).contiguous())
        batch, channels, steps, height, width = features.shape
        sequence = features.mean(dim=(-2, -1)).transpose(1, 2)
        attended, _ = self.attention(sequence)
        spatial_sequence = features.permute(0, 2, 1, 3, 4).contiguous()
        spatial_sequence = spatial_sequence + attended[:, :, :, None, None]
        temporal = self.convlstm(spatial_sequence)
        pooled = temporal.mean(dim=(-2, -1)).mean(dim=1)
        return self.classifier(self.dropout(pooled))


def load_checkpoint(path, device="cpu", strict=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("classes") != ["Assault", "Shooting", "Robbery", "Normal"]:
        raise ValueError("Checkpoint must declare the project four-class order")
    raw_config = checkpoint.get("model_config", {})
    raw_config["block_layers"] = tuple(raw_config.get("block_layers", (2, 2, 2)))
    config = ModelConfig(**raw_config)
    model = DenseNetAttentionConvLSTM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device).eval()
    return model, checkpoint


def load_compatible_pretrained_weights(model, path):
    """Load only shape-compatible DenseNet3D backbone tensors from a local checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Pretrained checkpoint must contain a state dictionary")
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Pretrained checkpoint has no state dictionary")
    target = model.backbone.state_dict()
    compatible = {}
    for key, value in state.items():
        normalized = key.removeprefix("module.").removeprefix("backbone.")
        if normalized in target and hasattr(value, "shape") and tuple(value.shape) == tuple(target[normalized].shape):
            compatible[normalized] = value
    if not compatible:
        raise ValueError("No shape-compatible DenseNet3D backbone weights were found")
    model.backbone.load_state_dict(compatible, strict=False)
    return {"loaded_tensor_count": len(compatible), "available_tensor_count": len(target),
            "source": str(Path(path).resolve())}
