from pathlib import Path

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR

from src.data.dataset_type import DatasetType
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.run.embeddingspace.EmbeddingOutlierDetectionExperiments import EmbeddingOutlierDetectionExperiments
from src.run.pixelspace.OutlierDetectionExperiment import OutlierDetectionExperiment
from src.run.pixelspace.config.vmmd.VMMDBaseConfiguration import VMMDBaseConfiguration
from src.run.pixelspace.config.vmmd.VMMDTestConfiguration import VMMDTestConfiguration
from src.utils.preprocessing import normalize_images, normalize_features
from src.vmmd.VMMDEmbedding import VMMDEmbedding
from src.vmmd.model.VMMDDiagonal1Channel import VMMDDiagonal1Channel


fashionmnist_categories = [
   # "T-shirt/top",
   # "Trouser",
   # "Pullover",
   # "Dress",
   #  "Coat",
   #  "Sandal",
   #  "Shirt",
   #  "Sneaker",
   #  "Bag",
    "Ankle boot"
]

mvtec_categories = [
#    "bottle",
#    "cable",
#    "capsule",
#     "carpet",
#     "grid",
#     "hazelnut",
#     "leather",
#     "metal_nut",
#     "pill",
#     "screw",
#     "tile",
#     "toothbrush",
#     "transistor",
#     "wood",
#     "zipper"
]

cifar10_classes = [
#    "airplane",
#    "automobile",
#    "bird",
#    "cat",
#    "deer",
#    "dog",
#    "frog",
#    "horse",
#    "ship",
    "truck"
]


def launch_vmmd_embedding_config(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDEmbedding(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            autoencoder=config.autoencoder, generator=config.generator, kernel=config.kernel
        )

        experiement = EmbeddingOutlierDetectionExperiments(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit()
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def rerun_od_experiments():
    root_dir = Path("../experiments/remote/12-03/")
    for mvtec_category in mvtec_categories:
        prefix = str(DatasetType.MVTEC_AD.name) + "[" + str(mvtec_category) + "]"

        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):

                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.MVTEC_AD,
                    dateset_category=mvtec_category,
                    image_size_od=(256,256),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator =  str(dir / "models" / "generator_1.pt")

                pretrained_vmmd_experiment(config, path_to_generator)

    for fashionmnist_category in fashionmnist_categories:
        prefix = str(DatasetType.OCCFMNIST.name) + "[" + str(fashionmnist_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.OCCFMNIST,
                    dateset_category=fashionmnist_category,
                    image_size_od=(28,28),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")

                pretrained_vmmd_experiment(config, path_to_generator)

    for cifar_category in cifar10_classes:
        prefix = str(DatasetType.OCCCIFAR10.name) + "[" + str(cifar_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.OCCCIFAR10,
                    dateset_category=cifar_category,
                    image_size_od=(32,32),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")
                pretrained_vmmd_experiment(config, path_to_generator)

def pretrained_vmmd_embedding_experiment(configs, path_to_pretrained_model):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDEmbedding(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            autoencoder=config.autoencoder, generator=config.generator
        )

        experiement = EmbeddingOutlierDetectionExperiments(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit_pretrained_model(path_to_pretrained_model)
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)
def launch_all_od_experiments():
    configs = [VMMDTestConfiguration()]
    launch_vmmd_experiment(configs)
    for mvtec_category in mvtec_categories:
            config = [
                VMMDBaseConfiguration(
                dataset_type=DatasetType.MVTEC_AD,
                dateset_category=mvtec_category,
                image_size_generator=(64, 64),
                image_size_train=(256,256),
                image_size_od=(256,256),
                preprocessing_fn=normalize_features,
                standardize_data=False,
                n_subspace_sample=2,
            )]

            launch_vmmd_experiment(config)

    for fashionmnist_category in fashionmnist_categories:
        config = [VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCFMNIST,
            dateset_category=fashionmnist_category,
            image_size_od=(28,28),
            image_size_generator=(28,28),
            image_size_train=(28,28),
            preprocessing_fn=normalize_features,
            standardize_data=False,
            n_subspace_sample=2
        )]
        launch_vmmd_experiment(config)

    for cifar_category in cifar10_classes:
        config = [VMMDBaseConfiguration(
            dataset_type=DatasetType.OCCCIFAR10,
            dateset_category=cifar_category,
            image_size_od=(32,32),
            preprocessing_fn=normalize_features,
            standardize_data=False,
            n_subspace_sample=2
        )]

        launch_vmmd_experiment(config)


def pretrained_vmmd_experiment(configs, path_to_pretrained_model):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            encoder=config.encoder, generator=config.generator
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[config.ens_base_estimator],
                vmmd=vmmd, max_n_jobs=-1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_train=config.image_size_train,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit_pretrained_model(path_to_pretrained_model)
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def run_all_vmmd_od_benchmark():
    root_dir = Path("../experiments/remote/21-03/")
    for mvtec_category in mvtec_categories:
        prefix = str(DatasetType.MVTEC_AD.name) + "[" + str(mvtec_category) + "]"

        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):

                config = [
                    VMMDBaseConfiguration(
                    dataset_type=DatasetType.MVTEC_AD,
                    dateset_category=mvtec_category,
                    image_size_od=(256,256),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=50
                )]

                path_to_generator =  str(dir / "models" / "generator_1.pt")

                run_vmmd_od_benchmark(config, path_to_generator)

    for fashionmnist_category in fashionmnist_categories:
        prefix = str(DatasetType.OCCFMNIST.name) + "[" + str(fashionmnist_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.OCCFMNIST,
                    dateset_category=fashionmnist_category,
                    image_size_od=(28,28),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")

                run_vmmd_od_benchmark(config, path_to_generator)

    for cifar_category in cifar10_classes:
        prefix = str(DatasetType.OCCCIFAR10.name) + "[" + str(cifar_category) + "]"
        for dir in root_dir.iterdir():
            if dir.stem.startswith(prefix):
                config = [VMMDBaseConfiguration(
                    dataset_type=DatasetType.OCCCIFAR10,
                    dateset_category=cifar_category,
                    image_size_od=(32,32),
                    preprocessing_fn=normalize_images,
                    standardize_data=False,
                    n_subspace_sample=100
                )]

                path_to_generator = str(dir / "models" / "generator_1.pt")
                run_vmmd_od_benchmark(config, path_to_generator)



def run_vmmd_od_benchmark(configs, path_to_pretrained_model):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            autoencoder=config.autoencoder, generator=config.generator
        )

        experiement = OutlierDetectionExperiment(
            vmmd=vmmd,
            od_model=CombinedOutlierDetector(
                base_estimators=[LUNAR(), LOF(), KNN()],
                vmmd=vmmd, max_n_jobs=-1,
                preprocessing_fn=config.preprocessing_fn
            ),
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_train=config.image_size_train,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            n_subspaces_sample=config.n_subspace_sample
        )

        experiement.fit_pretrained_model(path_to_pretrained_model)
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)

def launch_vmmd_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VMMDDiagonal1Channel(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr=config.lr, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            autoencoder=config.autoencoder, generator=config.generator
        )

        if config.ens_base_estimator is not None:
            experiement = OutlierDetectionExperiment(
                vmmd=vmmd,
                od_model=CombinedOutlierDetector(
                    base_estimators=[config.ens_base_estimator],
                    vmmd=vmmd, max_n_jobs=1,
                    preprocessing_fn=config.preprocessing_fn
                ),
                dataset_type=config.dataset_type,
                category=config.dateset_category,
                image_size_train=config.image_size_train,
                image_size_od=config.image_size_od,
                standardize_data=config.standardize_data,
                preprocessing_fn=config.preprocessing_fn,
                n_subspaces_sample=config.n_subspace_sample
            )

            experiement.fit()
            #experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)
        else:
            for ens_model in [LUNAR(), LOF(), KNN()]:
                experiement = OutlierDetectionExperiment(
                    vmmd=vmmd,
                    od_model=CombinedOutlierDetector(
                        base_estimators=[ens_model],
                        vmmd=vmmd, max_n_jobs=-1,
                        preprocessing_fn=config.preprocessing_fn
                    ),
                    dataset_type=config.dataset_type,
                    category=config.dateset_category,
                    image_size_train=config.image_size_train,
                    image_size_od=config.image_size_od,
                    standardize_data=config.standardize_data,
                    preprocessing_fn=config.preprocessing_fn,
                    n_subspaces_sample=config.n_subspace_sample
                )

                experiement.fit()
                experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)
