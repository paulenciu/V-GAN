import pandas as pd
import torch
from src.data.dataset.occ.OCCFMNIST import OCCFMNIST
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from torchvision import transforms, datasets
from anomalib.models import Patchcore, Padim, Dfm, Cflow, Stfpm
from anomalib.engine import Engine
from anomalib.data import Folder, AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import TestSplitMode, ValSplitMode
import warnings
import logging

def safe_normalize(x):
    norm = x.norm()
    return x / norm if norm > 0 else x

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Uncommented ToTensor
    # transforms.Lambda(safe_normalize),
])

class AnomalibWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.collate_fn = torch.utils.data.default_collate

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return {
            'image': data,
            'label': target,
        }

    def __len__(self):
        return len(self.dataset)

class TorchvisionDataModule(AnomalibDataModule):
    def __init__(self, dataset_name, x_train, y_train, x_test, y_test):
        super().__init__(train_batch_size=64, eval_batch_size=64, num_workers=4)
        self._name = dataset_name
        self.train_dataset = AnomalibDataset(augmentations=custom_transform)
        self.train_dataset.samples = pd.DataFrame(x_train.list())
        self.train_dataset.category = 0
        

    @property
    def name(self):
        return self._name

    def _setup(self):
        pass

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_data = self.train_dataset
        if stage in ("test", None):
            self.test_data = self.test_dataset

def get_datamodule(dataset_name):
    if dataset_name == "mvtec":
        folder = Folder(
            name="mvtec_ad",
            root="../datasets/mvtec_ad/bottle",
            normal_dir="custom/train/good",
            abnormal_dir="custom/test/defect",
            normal_test_dir="custom/test/good",
        )
        folder.transform_config = {"train": custom_transform, "test": custom_transform}
        return folder
    elif dataset_name == "fashionmnist":
        x_train, y_train = load_data(dataset_type=DatasetType.OCCFMNIST, category="Trouser", train=True)
        x_test, y_test = load_data(dataset_type=DatasetType.OCCFMNIST, category="Trouser", train=False)
        return TorchvisionDataModule(dataset_name, x_train, y_train, x_test, y_test)

    elif dataset_name == "cifar10":
        train_ds = datasets.CIFAR10("../datasets/cifar10", train=True, download=True, transform=custom_transform)
        test_ds = datasets.CIFAR10("../datasets/cifar10", train=False, download=True, transform=custom_transform)
        wrapped_train_ds = AnomalibWrapper(train_ds)
        wrapped_test_ds = AnomalibWrapper(test_ds)
        return TorchvisionDataModule(dataset_name, wrapped_train_ds, wrapped_test_ds)
    else:
        raise ValueError("Unsupported dataset")

def run(model_class, dataset_name):
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)

    datamodule = get_datamodule(dataset_name)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")
    model = model_class()

    engine = Engine(
        max_epochs=10,
        logger=False,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    engine.fit(datamodule=datamodule, model=model)
    engine.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    dataset_list = ["fashionmnist", "cifar10"]
    model_list = [Patchcore, Padim, Dfm, Stfpm]

    for dataset_name in dataset_list:
        for model_class in model_list:
            print(f"\nRunning {model_class.__name__} on {dataset_name}")
            run(model_class, dataset_name)