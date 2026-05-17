from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from src.run.pixelspace.PODAttentionBaselineExperiment import PODAttentionBaselineExperiment
from sympy.strategies.branch import canon

from src.data.dataset_type import DatasetType
from src.run.embeddingspace.EODEncodedBaselineExperiment import \
    EODEncodedBaselineExperiment
from src.run.embeddingspace.EODEncodedExperiment import EODEncodedExperiment
from src.run.pixelspace.PODBaselineExperiment import PODBaselineExperiment
from src.utils.preprocessing import normalize_images, no_preprocessing

fashionmnist_categories = [
    # "T-shirt/top",
    #  "Trouser",
    # "Pullover",
    # "Dress",
    # "Coat",
    # "Sandal",
    # "Shirt",
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
    # "airplane",
    # "automobile",
    # "bird",
    # "cat",
    # "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def launch_baseline_on_embedding_space(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        for method in [
            LUNAR(),
            LOF() ,
            KNN(),
            FeatureBagging(base_estimator=LOF(), n_estimators=10),
            FeatureBagging(base_estimator=LUNAR(),  n_estimators=10),
            FeatureBagging(base_estimator=KNN(),  n_estimators=10)
                       ]:
            print("RUNNING ", method.__class__.__name__.upper())
            baseline_experiments = EODEncodedBaselineExperiment(
                dataset_type=config.dataset_type,
                category=config.dateset_category,
                image_size_od=config.image_size_od,
                standardize_data=config.standardize_data,
                preprocessing_fn=config.preprocessing_fn,
                od_models=[
                    method
                ],
                encoder=config.encoder,
                encoder_name=config.encoder_name,
            )

            baseline_experiments.fit()
            baseline_experiments.evaluate()



def launch_baseline_on_embedding(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        baseline_experiments = EODEncodedBaselineExperiment(
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            od_models=[
                LUNAR(),
                LOF(),
                KNN(),
                FeatureBagging(base_estimator=LUNAR()),
                FeatureBagging(base_estimator=LOF()),
                FeatureBagging(base_estimator=KNN()),
            ],
            encoder=config.encoder,
            encoder_name=config.encoder_name,
        )

        baseline_experiments.fit()
        baseline_experiments.evaluate()

def launch_baseline_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        baseline_experiments = PODBaselineExperiment(
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
            PODBaselineExperiment(
                dataset_type=dataset_type,
                category=category,
                image_size_od=image_size_od,
                standardize_data=False,
                od_model=od_model,
                preprocessing_fn=no_preprocessing,
                root_dir="../experiments/od_baselines/pixelspace/unnormalized",
            )
            for category in categories
        ]

        for exp in exps:
            exp.fit()
            exp.evaluate()

def launch_attention_baseline(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))
        experiment = PODAttentionBaselineExperiment(
            dataset_type=config.dataset_type,
            category=config.dateset_category,
            image_size_od=config.image_size_od,
            standardize_data=config.standardize_data,
            preprocessing_fn=config.preprocessing_fn,
            od_models=[
                LUNAR(),
                LOF(),
                KNN(),
            ],
            exp_date="21-03"
        )
        experiment.fit()
        experiment.evaluate()
