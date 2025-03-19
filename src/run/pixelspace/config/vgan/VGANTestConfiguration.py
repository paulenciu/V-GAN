from src.data.dataset_type import DatasetType
from src.models.autoencoder.ResNet18AutoEncoderFineTuning import ResNet18AutoEncoderFineTuning
from src.run.config.vgan.VGANBaseConfiguration import VGANBaseConfiguration


class VGANTestConfiguration(VGANBaseConfiguration):

    def __init__(self, dataset_type=DatasetType.OCCCIFAR10, dateset_category = "cat"):
        super().__init__(
            filename="test",
            n_subspace_sample=2,
            dataset_type=dataset_type,
            dateset_category=dateset_category,
            epochs=1,
            detector=ResNet18AutoEncoderFineTuning()
        )