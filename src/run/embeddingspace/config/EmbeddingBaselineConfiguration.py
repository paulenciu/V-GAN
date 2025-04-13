from pyod.models.lunar import LUNAR

from src.models.autoencoder.pretrained_autoencoder.resnet.imagenet.ResNet18AutoEncoder import ResNet18AutoEncoder

from src.models.generator.diagonal_matrix.embedding.GeneratorRes50Conv import GeneratorRes50Conv
from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelSNGN import GeneratorOneChannelSNGN

from src.models.generator.diagonal_matrix.one_channel.GeneratorOneChannelV4Softmax import GeneratorOneChannelV4Softmax
from src.utils.preprocessing import normalize_images_col_softmax, normalize_features, normalize_images, no_preprocessing
from src.vmmd.MMDLossConstrained import RBF
from src.utils.preprocessing import no_preprocessing
from src.vmmd.penalty.MMDLossPenalty import MMDLossNoPenalty


class EmbeddingBaselineConfiguration:

    def __init__(self,
                store_stats = True,
                standardize_data = False,
                seed = 333,
                n_channels = 3,
                preprocessing_fn = no_preprocessing,
                image_size_od=(224, 224),
                path_to_directory = "../experiments/",
                encoder = ResNet18AutoEncoder().get_encoder_and_freeze(),
                n_subspace_sample = None,
                dataset_type=None,
                dateset_category=None,
                ens_base_estimator=LUNAR(),
                encoder_name=""
                ):

        self.encoder = encoder
        self.store_stats = store_stats
        self.standardize_data = standardize_data
        self.seed = seed
        self.n_channels = n_channels
        self.preprocessing_fn = preprocessing_fn
        self.path_to_directory = path_to_directory
        self.n_subspace_sample = n_subspace_sample or 100
        self.dataset_type = dataset_type
        self.dateset_category = dateset_category
        self.image_size_od = image_size_od
        self.ens_base_estimator = ens_base_estimator
        self.encoder_name = encoder_name
