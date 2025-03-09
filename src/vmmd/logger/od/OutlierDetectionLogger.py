import numpy as np
from matplotlib import pyplot as plt

from src.data.IDataset import IDataset
from src.od.CombinedOutlierDetector import CombinedOutlierDetector
from src.vmmd.outlier_detection.VMMDOD import VMMDOD


class OutlierDetectionLogger:

    def __init__(self, od_model: CombinedOutlierDetector, vmmd_od: VMMDOD):
        self.od_model = od_model
        self.vmmd_od = vmmd_od

    def log(self, x_standard, x_unstandard, y):
        ensemble_score = self.od_model.ensemble_detector.decision_score(x_standard)
        distance_score = self.od_model.distance_detector.decision_score(x_unstandard)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Outlier Detection Scores', fontsize=14, y=0.95)

        max_example = min(len(x_standard), 100)
        basis = np.arange(max_example)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Data Index', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)

        ax.plot(basis, y[:max_example], color='#2c7bb6', linewidth=1.5, label='Ground Truth')
        plt.fill_between(basis, y[:max_example], color='#2c7bb6', alpha=0.3)
        ax.plot(basis, distance_score[:max_example], color='#d7191c', linewidth=1.5, label='Distance Scores', linestyle='dashed')
        ax.plot(basis, ensemble_score[:max_example], color='#008000', linewidth=1.5, label=f'Ensemble Scores ({self.od_model.ensemble_detector.base_estimators[0].__class__.__name__})', linestyle='dashed')

        ax.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        plt.show()
        self.vmmd_od.store_od_plots(fig)