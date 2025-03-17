import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.vmmd.VMMD import VMMD


class VMMDOD:

    def __init__(self, vmmd: VMMD):
        self.vmmd = vmmd

    def store_ensemble_score(self, fig):
        path_to_directory = Path(self.vmmd.path_to_directory)
        #current_date = datetime.now().strftime("%d-%m")
        path_to_directory = Path(self.vmmd.path_to_directory) / "ens_score"
        path_to_directory.mkdir(parents=True, exist_ok=True)
        run_number = int(len(os.listdir(path_to_directory)))
        file_path = path_to_directory / f"ens_scores_{run_number}.pdf"
        fig.savefig(file_path, format="pdf", bbox_inches="tight")


    def store_od_stats(self, stats: dict, run_number: int):
        print("Stats: ", stats)
        path_to_directory = Path(self.vmmd.path_to_directory)

        file_path = path_to_directory / f"od_stats_{run_number}.csv"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        stats_df = pd.DataFrame([stats])

        if not file_path.exists():
            stats_df.to_csv(file_path, index=False)
        else:
            stats_df.to_csv(file_path, mode='a', header=False, index=False)

    def store_od_plots(self, fig):
        path_to_directory = Path(self.vmmd.path_to_directory)
        #current_date = datetime.now().strftime("%d-%m")
        path_to_directory = Path(self.vmmd.path_to_directory) / "od_scores"
        path_to_directory.mkdir(parents=True, exist_ok=True)
        run_number = int(len(os.listdir(path_to_directory)))
        file_path = path_to_directory / f"od_scores_{run_number}.pdf"
        fig.savefig(file_path, format="pdf", bbox_inches="tight")

    def store_od_benchmarks(self, fig):
        path_to_directory = Path(self.vmmd.path_to_directory)
        path_to_directory = path_to_directory / "od_benchmarks"
        path_to_directory.mkdir(parents=True, exist_ok=True)
        run_number = int(len(os.listdir(path_to_directory)))
        file_path = path_to_directory / f"od_benchmarks_{run_number}.pdf"
        fig.savefig(file_path, format="pdf", bbox_inches="tight")