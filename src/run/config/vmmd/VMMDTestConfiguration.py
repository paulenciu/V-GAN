from src.run.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.data.dataset_type import DatasetType


class VMMDTestConfiguration(VMMDBaseConfiguration):

    def __init__(self, dataset_type=DatasetType.OCCCIFAR10, dateset_category = "cat"):
        super().__init__(
            filename="test",
            n_subspace_sample=2,
            dataset_type=dataset_type,
            dateset_category=dateset_category,
            epochs=1
        )