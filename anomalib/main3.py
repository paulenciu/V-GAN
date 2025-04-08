# anomalib/main3.py
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from anomalib.data import MVTec
from anomalib.models import Padim, Patchcore, Dfm
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score
import numpy as np


class BenchmarkDataset(Dataset):
    def __init__(self, dataset_name, normal_class, image_size=(256, 256), train=True):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_name == "mvtec":
            dataset = MVTec(
                root="./datasets/mvtec",
                category=normal_class,
                transform=self.transform,
                train=train
            )
            self.data = dataset

        elif dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(
                root="./datasets",
                train=train,
                download=True,
                transform=self.transform
            )
            # Filter normal class
            idx = np.array(dataset.targets) == normal_class
            self.data = [(dataset.data[i], 0 if dataset.targets[i] == normal_class else 1)
                         for i in range(len(dataset)) if train == (dataset.targets[i] == normal_class)]

        elif dataset_name == "fashionmnist":
            dataset = torchvision.datasets.FashionMNIST(
                root="./datasets",
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.Grayscale(3),  # Convert to 3 channels
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
            # Filter normal class
            idx = np.array(dataset.targets) == normal_class
            self.data = [(dataset.data[i], 0 if dataset.targets[i] == normal_class else 1)
                         for i in range(len(dataset)) if train == (dataset.targets[i] == normal_class)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        image = self.transform(image)
        return {"image": image, "label": label}


def evaluate_model(model, test_loader):
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["image"])
            scores.extend(outputs["image_scores"].cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())

    auroc = roc_auc_score(labels, scores)
    return auroc


def run_benchmark():
    datasets = [
        ("mvtec", "bottle"),
        ("cifar10", 3),  # 3 is cat
        ("fashionmnist", 1)  # 1 is trouser
    ]

    models = {
        "PaDiM": lambda: Padim(
            input_size=(256, 256),
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"]
        ),
        "PatchCore": lambda: Patchcore(
            input_size=(256, 256),
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"]
        ),
        "DFM": lambda: Dfm(
            input_size=(256, 256),
            backbone="resnet18",
            layer="layer3"
        )
    }

    results = {}
    for dataset_name, normal_class in datasets:
        print(f"\nProcessing {dataset_name} dataset...")

        # Create datasets
        train_dataset = BenchmarkDataset(dataset_name, normal_class, train=True)
        test_dataset = BenchmarkDataset(dataset_name, normal_class, train=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        dataset_results = {}
        for model_name, model_fn in models.items():
            print(f"Training {model_name}...")
            model = model_fn()
            trainer = Trainer(max_epochs=1, accelerator="auto")
            trainer.fit(model, train_loader)

            auroc = evaluate_model(model, test_loader)
            dataset_results[model_name] = auroc
            print(f"{model_name} AUROC: {auroc:.3f}")

        results[dataset_name] = dataset_results

    return results


if __name__ == "__main__":
    results = run_benchmark()
    print("\nFinal Results:")
    for dataset, scores in results.items():
        print(f"\n{dataset}:")
        for model, auroc in scores.items():
            print(f"{model}: {auroc:.3f}")