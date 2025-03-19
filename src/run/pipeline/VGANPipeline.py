from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.run.pixelspace.OutlierDetectionExperiment import OutlierDetectionExperiment
from src.vgan.VGAN import VGAN


def launch_vgan_experiment(configs):
    for i, config in enumerate(configs):
        print("RUNNING EXPERIMENT", i, " FROM", len(configs))

        vmmd = VGAN(
            epochs=config.epochs, seed=config.seed, path_to_directory=config.path_to_directory,
            lr_G=config.lr_g, lr_D=config.lr_d, penalty=config.penalty, filename=config.filename,
            batch_size=config.batch_size, momentum=config.momentum, weight_decay=config.weight_decay,
            detector=config.detector, generator=config.generator, iternum_g=config.iternum_g, iternum_d=config.iternum_d,
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

        experiement.fit()
        experiement.evaluate_interval(ensemble_weight_start=0, ensemble_weight_end=1, step=1.0 / 10.0)