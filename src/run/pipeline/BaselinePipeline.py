from src.run.pixelspace.OutlierDetectionBaselineExperiment import OutlierDetectionBaselineExperiment
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
