import torchvision.transforms as transforms
from config import config
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


class ScreenClassifier:
    def __init__(self):
        model_name = config.model.name
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}"
            )

        model_fn, weights = MODELS[model_name]
        print(f"Getting {model_name} model...")

        self.model = model_fn(weights=weights)
        self.model.eval()
        self.class_names = weights.meta["categories"]
        self.preprocess = self._build_preprocess()
        print(f"{model_name} model loaded")

    def _build_preprocess(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
