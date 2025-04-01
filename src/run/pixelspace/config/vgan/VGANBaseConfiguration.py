from pyod.models.lunar import LUNAR
from src.models.autoencoder.ResNet18AutoEncoderFineTuning import ResNet18AutoEncoderFineTuning
from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax
from src.utils.preprocessing import normalize_features
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty


class VGANBaseConfiguration:

    def __init__(self,
                 lr_g=0.0001,
                 lr_d=0.0001,
                 latent_size=128,
                 epochs=2000,
                 batch_size=64,
                 store_stats=True,
                 penalty=MMDLossNoPenalty(),
                 momentum=0.8,
                 weight_decay=0.1,
                 standardize_data=False,
                 seed=333,
                 n_channels=3,
                 iternum_g = 10,
                 iternum_d = 5,
                 preprocessing_fn=normalize_features,
                 image_size_od=(32, 32),
                 image_size_generator=(32, 32),
                 path_to_directory="../experiments/remote",
                 detector=ResNet18AutoEncoderFineTuning(),
                 generator=None,
                 n_subspace_sample=None,
                 filename=None,
                 dataset_type=None,
                 dateset_category=None,
                 add_to_title: str = None,
                 ens_base_estimator=LUNAR()):
        self.generator = generator or GeneratorOneChannelV4Softmax(latent_size=latent_size,
                                                                   image_shape=(n_channels, *image_size_generator))
        self.detector = detector
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.epochs = epochs
        self.batch_size = batch_size
        self.store_stats = store_stats
        self.penalty = penalty
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.standardize_data = standardize_data
        self.seed = seed
        self.n_channels = n_channels
        self.preprocessing_fn = preprocessing_fn
        self.image_size_train = 224 #Autoencoder input size
        self.image_size_od = image_size_od
        self.path_to_directory = path_to_directory
        self.n_subspace_sample = n_subspace_sample or 100
        self.dataset_type = dataset_type
        self.dateset_category = dateset_category
        self.iternum_g = iternum_g
        self.iternum_d = iternum_d
        self.ens_base_estimator = ens_base_estimator
        self.filename = filename or self.create_filename(add_to_title)

    def create_filename(self, add_to_title):
        return (
            "vgan_"
            f"{self.dataset_type.name}"
            f"_{self.preprocessing_fn.__name__}"
            f"_train{self.image_size_train}"
            f"_od{self.image_size_od[0]}"
            f"_lr_d={self.lr_d}"
            f"_lr_g={self.lr_g}"
            f"iternum_d={self.iternum_d}"
            f"iternum_g={self.iternum_g}"
            f"_bs={self.batch_size}"
            f"_ep={self.epochs}"
            f"_{add_to_title if add_to_title is not None else ''}"
        )


