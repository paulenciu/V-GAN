import torch
import numpy as np
import torchvision
from anomalib.data import MVTec, TaskType
from anomalib.data.utils import ValSplitMode
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from anomalib.models import Padim, Patchcore, Cflow, Dfm, Stfpm
from pytorch_lightning import Trainer
import os

from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.utils.preprocessing import normalize_images


class CustomDataset(Dataset):
    def __init__(self, dataset_type=DatasetType.MVTEC_AD, category="bottle",
                 image_size=(256, 256), train=True, transform=None):
        self.transform = transform
        if train:
            x, _ = load_data(dataset_type, category, image_size, train=train, custom_transform=transform)
            self.y = torch.zeros(len(x))
        else:
            x, self.y = load_data(dataset_type, category, image_size, train=train, custom_transform=transform)

        x = extract_and_flatten_images_dataset_3d(x).to("cpu")
        x = torch.from_numpy(normalize_images(x.numpy()))
        self.x = x.view(-1, 3, *image_size).to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "image": self.x[idx],  # Should be [C, H, W]
            "label": self.y[idx]  # Return scalar label
        }


# Configuration with proper normalization
input_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256), transform=transform)
test_dataset = CustomDataset(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256), train=False, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, test_loader):
    image_scores = []
    pixel_scores = []
    labels = []
    masks = []

    for batch in test_loader:
        with torch.no_grad():
            outputs = model(batch["image"])

        image_scores.extend(outputs["image_scores"].cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())

        if "anomaly_maps" in outputs:
            pixel_scores.extend(outputs["anomaly_maps"].cpu().numpy())
            masks.extend(batch["mask"].cpu().numpy())

    # Image-level AUROC
    auroc = roc_auc_score(labels, image_scores)
    print(f"Image AUROC: {auroc:.3f}")

    # Pixel-level AUPRO (if available)
    if len(pixel_scores) > 0:
        from anomalib.utils.metrics import AUPRO
        aupro = AUPRO()(torch.tensor(np.stack(pixel_scores)),
                        torch.tensor(np.stack(masks)))
        print(f"Pixel AUPRO: {aupro:.3f}")

    return auroc

def run_padim():
    model = Padim(
        input_size=input_size,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"]
    )
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu")
    trainer.fit(model, train_loader)
    return model

def run_patchcore():
    model = Patchcore(
        input_size=input_size,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = Trainer(max_epochs=1, accelerator="auto")
    trainer.fit(model, train_loader)
    return model

def run_cflow():
    model = Cflow(
        input_size=input_size,
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
    )
    trainer = Trainer(max_epochs=100, accelerator="auto")
    trainer.fit(model, train_loader)
    return model

def run_dfm():
    model = Dfm(
        input_size=input_size,
        backbone="resnet18",
        layer="layer3",
        pre_trained=True,
    )
    trainer = Trainer(max_epochs=1, accelerator="auto")
    trainer.fit(model, train_loader)
    return model

def run_stfpm():
    model = Stfpm(
        input_size=input_size,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"]
    )
    trainer = Trainer(max_epochs=100, accelerator="auto")
    trainer.fit(model, train_loader)
    return model

models = {
    "PADIM": run_padim(),
    "PatchCore": run_patchcore(),
    #"CFLOW": run_cflow(),
    "DFM": run_dfm(),
    #"STFPM": run_stfpm()
}

if __name__ == "__main__":
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        evaluate_model(model, test_loader)