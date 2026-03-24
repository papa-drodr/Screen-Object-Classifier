import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18


class ScreenClassifier:
    def __init__(self):
        print("ResNet18 모델을 불러오는 중입니다...")
        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(weights=weights)
        self.model.eval()
        self.class_names = weights.meta["categories"]
        self.preprocess = self._build_preprocess()
        print("load model")

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
