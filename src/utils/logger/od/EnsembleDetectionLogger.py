import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

class EnsembleDetectionLogger:

    def __init__(self, ensemble_detector, vmmd_od):
        self.ensemble_detector = ensemble_detector
        self.vmmd_od = vmmd_od

    def log(self, x, y):
        ensemble_score = self.ensemble_detector.decision_score(x).T
        roc_auc_scores = np.zeros(len(ensemble_score))
        for i in range(len(ensemble_score)):
            roc_auc_scores[i] = auc(y, ensemble_score[i])

        detector_weights = self.vmmd_od.vmmd.proba

        fig, ax = plt.subplots(figsize=(25, 6))

        norm = Normalize(vmin=min(detector_weights), vmax=max(detector_weights))
        colormap = plt.cm.magma

        # Create bars with colors corresponding to weights
        bars = ax.bar(np.arange(len(ensemble_score)), roc_auc_scores,
                      color=colormap(norm(detector_weights)), edgecolor='black')

        ax.set_title('ROC AUC Scores for Ensemble Detector', fontsize=16)
        ax.set_xlabel('Detector Index', fontsize=14)
        ax.set_ylabel('ROC AUC Score', fontsize=14)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_xticks(np.arange(len(ensemble_score)))
        ax.set_xticklabels(np.arange(len(ensemble_score)))

        for bar, prob, auc_score in zip(bars, detector_weights, roc_auc_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{auc_score:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Add a colorbar to show the mapping of weights to colors
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Detector Weight')

#        plt.style.use('seaborn-v0_8')
        plt.style.use('seaborn')
        plt.tight_layout()
        plt.show()
        self.vmmd_od.store_ensemble_score(fig)