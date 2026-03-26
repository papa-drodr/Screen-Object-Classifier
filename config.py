from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "resnet50"  # "resnet18" or "resnet50"


@dataclass
class CaptureConfig:
    width: int = 1080
    height: int = 720
    start_top: int = 150
    start_left: int = 150
    move_speed: int = 2


@dataclass
class DisplayConfig:
    minimap_width: int = 240
    smoothing_window: int = 10
    top_k: int = 3


@dataclass
class Config:
    model: ModelConfig = None
    capture: CaptureConfig = None
    display: DisplayConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.capture is None:
            self.capture = CaptureConfig()
        if self.display is None:
            self.display = DisplayConfig()


config = Config()
