import numpy as np
from matplotlib import pyplot as plt

from src.data.IDataset import IDataset
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.vmmd.outlier_detection.VMMDOD import VMMDOD


class OutlierDetectionLogger:

    def __init__(self, od_model: CombinedOutlierDetector, vmmd_od: VMMDOD):
        self.od_model = od_model
        self.vmmd_od = vmmd_od

    def log(self, x, y):
        #ensemble_score = self.od_model.ensemble_detector.decision_score_agg(x)
        #distance_score = self.od_model.distance_detector.decision_score(x)

        ensemble_scores = self.od_model.get_ensemble_scores(x)
        distance_score = self.od_model.get_distance_score(x)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Outlier Detection Scores', fontsize=14, y=0.95)

        max_example = min(len(x), 100)
        basis = np.arange(max_example)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Data Index', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)

        ax.plot(basis, y[:max_example], color='#2c7bb6', linewidth=1.5, label='Ground Truth')
        plt.fill_between(basis, y[:max_example], color='#2c7bb6', alpha=0.3)
        ax.plot(basis, distance_score[:max_example], color='#d7191c', linewidth=1.5, label='Distance Scores', linestyle='dashed')
        colors = ['#008000', '#0000FF', '#FF0000', '#800080', '#FFA500']

        for idx, ensemble_score in enumerate(ensemble_scores):
            ax.plot(basis, ensemble_score[:max_example], color=colors[idx % len(colors)], linewidth=1.5,
                    label=f'Ensemble Scores ({self.od_model.ensemble_detector.base_estimators[idx].__class__.__name__})',
                    linestyle='dashed')

        ax.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()
        self.vmmd_od.store_od_plots(fig)