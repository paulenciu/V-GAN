from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sympy.strategies.branch import canon

from src.data.dataset_type import DatasetType
from src.run.embeddingspace.EmbeddingSpaceOutlierDetectionExperiment import EmbeddingSpaceOutlierDetectionExperiment
from src.run.pixelspace.OutlierDetectionBaselineExperiment import OutlierDetectionBaselineExperiment
from src.utils.preprocessing import normalize_images

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

def launch_baseline_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        baseline_experiments = OutlierDetectionBaselineExperiment(
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            od_model=config.ens_base_estimator,
        )

        baseline_experiments.fit()
        baseline_experiments.evaluate()

def launch_all_baseline_experiments_fs(od_model):

    configs = [
        (DatasetType.MVTEC_AD, mvtec_categories, (256,256)),
        (DatasetType.OCCCIFAR10, cifar10_classes, (32,32)),
        (DatasetType.OCCFMNIST, fashionmnist_categories, (28,28))
    ]

    for dataset_type, categories, image_size_od in configs:
        exps = [
            OutlierDetectionBaselineExperiment(
                dataset_type=dataset_type,
                category=category,
                image_size_od=image_size_od,
                standardize_data=False,
                preprocessing_fn=normalize_images,
                od_model=od_model,
            )
            for category in categories
        ]

        for exp in exps:
            exp.fit()
            exp.evaluate()
