from src.run.pixelspace.OutlierDetectionBaselineExperiment import OutlierDetectionBaselineExperiment


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