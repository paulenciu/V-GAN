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
        # samples.loc[
        #     (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        #     "mask_path",
        # ] = mask_samples.image_path.to_numpy()

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
