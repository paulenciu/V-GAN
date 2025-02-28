from src.run.config.BaseConfiguration import BaseConfiguration
from src.data.dataset_type import DatasetType


class TestConfiguration(BaseConfiguration):

    def __init__(self, dataset_type=DatasetType.OCCCIFAR10, dateset_category = "cat"):
        super().__init__(
            filename="test",
            n_subspace_sample=2,
            dataset_type=dataset_type,
            dateset_category=dateset_category,
            epochs=1
        )




