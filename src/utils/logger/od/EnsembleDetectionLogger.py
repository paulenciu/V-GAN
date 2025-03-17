import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as auc

class EnsembleDetectionLogger:

    def __init__(self, ensemble_detector, vmmd_od):
        self.ensemble_detector = ensemble_detector
        self.vmmd_od = vmmd_od

    def log(self, x, y):
        ensemble_score = self.ensemble_detector.decision_score(x).T
        roc_auc_scores = np.zeros(len(ensemble_score))
        for i in range(len(ensemble_score)):
            roc_auc_scores[i] = auc(y, ensemble_score[i])

        plt.figure(figsize=(10, 6))
        bars = plt.bar(np.arange(len(ensemble_score)), roc_auc_scores, color='skyblue', edgecolor='black')

        # Adding titles and labels
        plt.title('ROC AUC Scores for Ensemble Detector', fontsize=16)
        plt.xlabel('Detector Index', fontsize=14)
        plt.ylabel('ROC AUC Score', fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(np.arange(len(ensemble_score)), rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10)

        #plt.style.use('seaborn-v0_8')
        plt.style.use('seaborn')
        plt.show()