import math

from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from src.models.autoencoder.pretrained_autoencoder.resnet.ResNet18AutoEncoder import ResNet18AutoEncoder

from src.models.encoder.IdentityEncoder import IdentityEncoder
from src.models.generator.diagonal_matrix.embedding.GeneratorRes50 import GeneratorRes18
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax
from src.utils.preprocessing import normalize_images_col_softmax, normalize_features, normalize_images, no_preprocessing
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty


class EmbeddingVMMDBaseConfiguration:

    def __init__(self, lr = 0.001,
                latent_size = 128,
                epochs = 2000,
                batch_size = 1024,
                store_stats = True,
                penalty = MMDLossNoPenalty(),
                momentum = 0.8,
                weight_decay = 0.1,
                standardize_data = False,
                seed = 333,
                n_channels = 3,
                preprocessing_fn = no_preprocessing,
                image_size_train=(224, 224),
                path_to_directory = "../experiments/remote",
                autoencoder = ResNet18AutoEncoder(),
                generator = None,
                n_subspace_sample = None,
                filename=None,
                dataset_type=None,
                dateset_category=None,
                add_to_title: str=None,
                ens_base_estimator=LUNAR(),
                set_decoder_eval=True):

        self.generator = generator or GeneratorRes18(latent_size=latent_size, image_shape=math.prod(autoencoder.get_encoder_input_shape()))
        self.autoencoder = autoencoder
        self.lr = lr
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
        self.path_to_directory = path_to_directory
        self.n_subspace_sample = n_subspace_sample or 100
        self.dataset_type = dataset_type
        self.dateset_category = dateset_category
        self.image_size_train = image_size_train
        self.filename = filename or self.create_filename(add_to_title)
        self.ens_base_estimator = ens_base_estimator
        self.set_decoder_eval = set_decoder_eval

    def create_filename(self, add_to_title: str = ""):
        return ("embedding_"
                f"{self.dataset_type.name}"
                f"[{self.dateset_category}]"
                f"_{self.preprocessing_fn.__name__ if self.preprocessing_fn else ''}"
                f"_{ 'standardised' if self.standardize_data else ''}"
                f"_train{self.image_size_train[0]}"
                f"_lr={self.lr}"
                f"_bs={self.batch_size}"
                f"_ep={self.epochs}"
                f"_{add_to_title}")