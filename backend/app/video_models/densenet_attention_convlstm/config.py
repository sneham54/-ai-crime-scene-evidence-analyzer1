from dataclasses import asdict, dataclass


CLASSES = ("Assault", "Shooting", "Robbery", "Normal")


@dataclass(frozen=True)
class PreprocessConfig:
    clip_length: int = 16
    image_size: int = 112
    clip_stride: int = 16
    max_duration_sec: float | None = 600.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if self.clip_length < 2 or self.image_size < 16 or self.clip_stride < 1:
            raise ValueError("clip_length >= 2, image_size >= 16 and clip_stride >= 1 are required")
        if len(self.mean) != 3 or len(self.std) != 3 or any(v <= 0 for v in self.std):
            raise ValueError("mean and std must contain three channels and std must be positive")


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 4
    growth_rate: int = 12
    block_layers: tuple[int, ...] = (2, 2, 2)
    initial_features: int = 24
    attention_heads: int = 4
    convlstm_hidden: int = 64
    dropout: float = 0.3

    def __post_init__(self):
        if self.num_classes != 4:
            raise ValueError("This project model has exactly four output classes")
        if self.attention_heads < 1 or self.convlstm_hidden < 1 or not self.block_layers:
            raise ValueError("Model dimensions must be positive")
        final_channels = self.initial_features
        for index, layers in enumerate(self.block_layers):
            final_channels += self.growth_rate * layers
            if index < len(self.block_layers) - 1:
                final_channels = max(8, final_channels // 2)
        if final_channels % self.attention_heads:
            raise ValueError("Final feature channels must be divisible by attention_heads")

    def to_dict(self):
        return asdict(self)
