from pathlib import Path

from anomalib.cifar.CifarDataset import CifarDataset
from anomalib.data import AnomalibDataModule
from anomalib.data.utils import Split, ValSplitMode, TestSplitMode
from torchvision.transforms.v2 import Transform


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
