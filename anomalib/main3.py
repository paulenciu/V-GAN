import os

import pandas as pd
import seaborn as sns
import torch
from pathlib import Path
from PIL import Image

import torchvision
from anomalib.data.errors import MisMatchError
from anomalib.models import Padim, Dfm
from matplotlib import pyplot as plt
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.utils.BigUBuilder import calculate_average_u
from src.utils.preprocessing import normalize_images
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.vmmd.VMMDWrapper import VMMDWrapper
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel

from torchvision.transforms.v2 import Transform
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.data.utils import LabelName, Split, validate_path

from typing import List, Dict
from anomalib.models import Padim, Dfm, Patchcore, Cflow, Stfpm
import time

from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore


from pathlib import Path

from anomalib.data import AnomalibDataModule
from anomalib.data.utils import Split, ValSplitMode, TestSplitMode
from torchvision.transforms.v2 import Transform

def create_attention_mask(dataset_type, category, exp_date="21-03"):
    root_dir = Path("../experiments/remote/") / exp_date
    vmmd = VMMDDiagonal1Channel(filename="placeholder_name", autoencoder=IdentityEncoder(), generator=None)
    vmmd_wrapper = VMMDWrapper(vmmd)
    model_param_path = None
    for dir_name in os.listdir(root_dir):
        if dir_name.startswith(dataset_type.name) and dir_name.__contains__(category.split("/")[0]):
            model_path =  root_dir / dir_name / "models"
            fname =  f"generator_{len(os.listdir(model_path)) - 1}.pt"
            model_param_path = model_path / fname
    if model_param_path is None:
        raise FileNotFoundError(f"No model found in {root_dir}")

    vmmd_wrapper.load_model(str(model_param_path))
    u = vmmd.sample_count_subspaces(count=500)
    u_avg = calculate_average_u(u)
    return u_avg, vmmd

class CifarAnomalibDataModule(AnomalibDataModule):

    def __init__(
            self,
            root: Path | str = "../datasets/cifar10",
            category: str = "cat",
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            num_workers: int = 8,
            train_augmentations: Transform | None = None,
            val_augmentations: Transform | None = None,
            test_augmentations: Transform | None = None,
            augmentations: Transform | None = None,
            test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
            test_split_ratio: float = 0.2,
            val_split_mode: ValSplitMode | str = ValSplitMode.NONE,
            val_split_ratio: float = 0.0,
            seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = CifarDataset(
            split=Split.TRAIN,
            category=self.category,
        )
        self.test_data = CifarDataset(
            split=Split.TEST,
            category=self.category,
        )
        self.val_data = None

    def val_dataloader(self):
        return []

import torch

from anomalib.data import AnomalibDataset
from anomalib.data.dataclasses import DatasetItem, ImageBatch, ImageItem
from anomalib.data.utils import Split, LabelName
from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.utils.ImageFlattenerUtility import unflatten_images_3d
from src.utils.preprocessing import normalize_images
from torchvision.transforms.v2 import Transform


class CifarDataset(AnomalibDataset):

    def __init__(
        self,
        category: str = "cat",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.category = category
        self.split = split
        self.samples, self.images, self.labels = self.make_samples(
            category=self.category,
            split=self.split
        )

    def make_samples(self, split: str | Split = None, category: str | None = None) -> list[str]:
        if split==Split.TRAIN:
            images, labels = load_data(dataset_type=DatasetType.OCCCIFAR10, category=category, image_size=(32, 32))

            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.OCCCIFAR10, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)


            samples_list = [("", "train", "good", "image.png") for _ in range(len(images))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        elif split==Split.TEST:
            images, labels =  load_data(dataset_type=DatasetType.OCCCIFAR10, category=category, image_size=(32, 32), train=False)

            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.OCCCIFAR10, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)

            samples_list = [("", "test", "good", "image.png") if y == 0 else ("", "test", "defect", "image.png") for y in range(len(labels))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        # separate masks from samples
        mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )
        samples = samples[samples.split != "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )

        # assign mask paths to anomalous test images
        samples["mask_path"] = ""
        # infer the task type
        samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

        if split:
            samples = samples[samples.split == split].reset_index(drop=True)

        return samples,images, labels

    def __getitem__(self, index: int) -> DatasetItem:
        image = self.images[index]
        label = self.labels[index]

        item = {"image_path": "image_path", "gt_label": index}
        item["image"] = image

        return ImageItem(
            image=item["image"],
            gt_mask=item.get("gt_mask"),
            gt_label=int(label),
            image_path="image_path",
            mask_path="mask_path",
        )

class FashionDataset(AnomalibDataset):

    def __init__(
        self,
        category: str = "Trouser",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.category = category
        self.split = split
        self.samples, self.images, self.labels = self.make_samples(
            category=self.category,
            split=self.split
        )

    def make_samples(self, split: str | Split = None, category: str | None = None) -> list[str]:
        if split==Split.TRAIN:
            images, labels = load_data(dataset_type=DatasetType.OCCFMNIST, category=category, image_size=(28, 28))

            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.OCCFMNIST, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)


            samples_list = [("", "train", "good", "image.png") for _ in range(len(images))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        elif split==Split.TEST:
            images, labels =  load_data(dataset_type=DatasetType.OCCFMNIST, category=category, image_size=(28, 28), train=False)

            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.OCCFMNIST, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)

            samples_list = [("", "test", "good", "image.png") if y == 0 else ("", "test", "defect", "image.png") for y in range(len(labels))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        # separate masks from samples
        mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )
        samples = samples[samples.split != "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )

        # assign mask paths to anomalous test images
        samples["mask_path"] = ""

        # infer the task type
        samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

        if split:
            samples = samples[samples.split == split].reset_index(drop=True)

        return samples,images, labels

    def __getitem__(self, index: int) -> DatasetItem:
        image = self.images[index]
        label = self.labels[index]

        item = {"image_path": "image_path", "gt_label": index}
        item["image"] = image

        return ImageItem(
            image=item["image"],
            gt_mask=item.get("gt_mask"),
            gt_label=int(label),
            image_path="image_path",
            mask_path="mask_path",
        )

class FashionMnistAnomalibDataModule(AnomalibDataModule):

    def __init__(
            self,
            root: Path | str = "../datasets/fashion_mnist",
            category: str = "Trouser",
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            num_workers: int = 8,
            train_augmentations: Transform | None = None,
            val_augmentations: Transform | None = None,
            test_augmentations: Transform | None = None,
            augmentations: Transform | None = None,
            test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
            test_split_ratio: float = 0.2,
            val_split_mode: ValSplitMode | str = ValSplitMode.NONE,
            val_split_ratio: float = 0.0,
            seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = FashionDataset(
            split=Split.TRAIN,
            category=self.category,
        )
        self.test_data = FashionDataset(
            split=Split.TEST,
            category=self.category,
        )
        self.val_data = None

    def val_dataloader(self):
        return []

class MVTecDataset(AnomalibDataset):

    def __init__(
        self,
        category: str = "bottle",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.category = category
        self.split = split
        self.samples, self.images, self.labels = self.make_samples(
            category=self.category,
            split=self.split
        )

    def make_samples(self, split: str | Split = None, category: str | None = None) -> list[str]:
        if split==Split.TRAIN:
            images, labels = load_data(dataset_type=DatasetType.MVTEC_AD, category=category, image_size=(256, 256))
            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.MVTEC_AD, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)

            samples_list = [("", "train", "good", "image.png") for _ in range(len(images))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        elif split==Split.TEST:
            images, labels =  load_data(dataset_type=DatasetType.MVTEC_AD, category=category, image_size=(256, 256), train=False)
            n_channels, height, width = images[0].shape[0], images[0].shape[1], images[0].shape[2]

            attention_mask, vmmd = create_attention_mask(dataset_type=DatasetType.MVTEC_AD, category=category)
            attention_mask = attention_mask.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
            images = vmmd.apply_subspaces_operator(images, attention_mask)

            flattened_images = images.view(images.size(0), -1).cpu()
            x_flattened_preprocessed = torch.from_numpy(normalize_images(flattened_images.numpy())).float()
            images = unflatten_images_3d(x_flattened_preprocessed, n_channels, height, width)

            samples_list = [("", "test", "good", "image.png") if y == 0 else ("", "test", "defect", "image.png") for y in range(len(labels))]
            samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
            samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
            samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
            samples.label_index = samples.label_index.astype(int)

        # separate masks from samples
        mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )
        samples = samples[samples.split != "ground_truth"].sort_values(
            by="image_path",
            ignore_index=True,
        )

        # assign mask paths to anomalous test images
        samples["mask_path"] = ""

        # infer the task type
        samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

        if split:
            samples = samples[samples.split == split].reset_index(drop=True)

        return samples,images, labels

    def __getitem__(self, index: int) -> DatasetItem:
        image = self.images[index]
        label = self.labels[index]

        item = {"image_path": "image_path", "gt_label": index}
        item["image"] = image

        return ImageItem(
            image=item["image"],
            gt_mask=item.get("gt_mask"),
            gt_label=int(label),
            image_path="image_path",
            mask_path="mask_path",
        )

class MVTecADAnomalibDataModule(AnomalibDataModule):

    def __init__(
            self,
            root: Path | str = "../datasets/mvtec_ad",
            category: str = "bottle",
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            num_workers: int = 8,
            train_augmentations: Transform | None = None,
            val_augmentations: Transform | None = None,
            test_augmentations: Transform | None = None,
            augmentations: Transform | None = None,
            test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
            test_split_ratio: float = 0.2,
            val_split_mode: ValSplitMode | str = ValSplitMode.NONE,
            val_split_ratio: float = 0.0,
            seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = MVTecDataset(
            split=Split.TRAIN,
            category=self.category,
        )
        self.test_data = MVTecDataset(
            split=Split.TEST,
            category=self.category,
        )
        self.val_data = None

    def val_dataloader(self):
        return []

fashionmnist_categories = [
   "T-shirt/top",
   "Trouser",
   "Pullover",
   "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

mvtec_categories = [
   "bottle",
   "cable",
   "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

cifar10_classes = [
   "airplane",
   "automobile",
   "bird",
   "cat",
   "deer",
   "dog",
   "frog",
   "horse",
   "ship",
    "truck"
]



# Modify the setup_results function
def setup_results(models: List[str], datasets: List[DatasetType]) -> pd.DataFrame:
    """Initialize a structured DataFrame for storing benchmark results."""

    # Create MultiIndex for all combinations
    categories = []
    for dataset in datasets:
        if dataset == DatasetType.OCCFMNIST:
            categories.extend([(dataset.value, cat) for cat in fashionmnist_categories])
        elif dataset == DatasetType.MVTEC_AD:
            categories.extend([(dataset.value, cat) for cat in mvtec_categories])
        elif dataset == DatasetType.OCCCIFAR10:
            categories.extend([(dataset.value, cat) for cat in cifar10_classes])

    index = pd.MultiIndex.from_tuples(
        [(dataset, category, model)
         for dataset, category in categories
         for model in models],
        names=["dataset", "category", "model"]
    )

    # Define metrics columns
    metrics = [
        "train_time",
        "test_time",
        "auroc",
        "f1_score",
    ]

    return pd.DataFrame(
        index=index,
        columns=metrics
    ).reset_index()


# Modified run_benchmarks function
def run_benchmarks(models: List[str], datasets: List[DatasetType], root_dir: Path) -> pd.DataFrame:
    """Run benchmarking across all models and datasets."""

    results_df = setup_results(models, datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, row in results_df.iterrows():
        dataset_name = row["dataset"]
        category = row["category"]
        model_name = row["model"]

        print(f"\n{'=' * 40}")
        print(f"Training {model_name} on {dataset_name} ({category})")
        print(f"{'=' * 40}")

        # Initialize model
        model = {
            "Padim": Padim(backbone="resnet18"),
            "Dfm": Dfm(backbone="resnet18"),
            "Patchcore": Patchcore(backbone="resnet18"),
            "Cflow": Cflow(backbone="resnet18"),
            "Stfpm": Stfpm(backbone="resnet18")
        }[model_name].to(device)

        if dataset_name == DatasetType.OCCFMNIST.value:
            datamodule = FashionMnistAnomalibDataModule(
                category=category,
                train_batch_size=32,
                eval_batch_size=32
            )
        elif dataset_name == DatasetType.OCCCIFAR10.value:
            datamodule = CifarAnomalibDataModule(
                category=category,
                train_batch_size=32,
                eval_batch_size=32
            )
        elif dataset_name == DatasetType.MVTEC_AD.value:
            datamodule = MVTecADAnomalibDataModule(
                category=category,
                train_batch_size=32,
                eval_batch_size=32,
                val_split_ratio=0.0
            )

        # Training
        start_time = time.time()
        engine = Engine(accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        train_time = time.time() - start_time

        # Testing
        start_test = time.time()
        metrics = engine.test(model=model, datamodule=datamodule)[0]
        test_time = time.time() - start_test

        # Store results
        results_df.loc[idx, "train_time"] = train_time
        results_df.loc[idx, "test_time"] = test_time
        results_df.loc[idx, "auroc"] = metrics["image_AUROC"]
        results_df.loc[idx, "f1_score"] = metrics["image_F1Score"]

        # Cleanup
        del model, datamodule, engine
        torch.cuda.empty_cache()

    return results_df


# Modified main block
if __name__ == "__main__":
    # Configuration
    root_dir = Path("../datasets")
    #""Cflow"
    models = ["Padim", "Dfm", "Stfpm"]
    datasets = [
        #DatasetType.MVTEC_AD,
        DatasetType.OCCFMNIST,
        #DatasetType.OCCCIFAR10,
    ]

    results = run_benchmarks(models, datasets, root_dir)

    results.to_csv("anomaly_benchmarks_attention_fmnist.csv", index=False)

    datasets = [
        #DatasetType.MVTEC_AD,
        #DatasetType.OCCFMNIST,
        DatasetType.OCCCIFAR10,
    ]

    results = run_benchmarks(models, datasets, root_dir)

    results.to_csv("anomaly_benchmarks_attention_cifar10.csv", index=False)
