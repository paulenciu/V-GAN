from datetime import datetime
from pathlib import Path

import pandas as pd

from src.vmmd.VMMD import VMMD

class VMMDOD:

    def __init__(self, vmmd: VMMD):
        self.vmmd = vmmd

    def store_od_stats(self, stats: dict, run_number: int):
        stats = pd.DataFrame([stats])
        path_to_directory = Path(self.vmmd.path_to_directory)
        current_date = datetime.now().strftime("%d-%m")

        path_to_directory = path_to_directory / current_date / self.vmmd.filename / f"od_stats_{run_number}.csv"
        stats.to_csv(path_to_directory, index=False)


