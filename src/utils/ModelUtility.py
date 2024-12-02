import pandas as pd
import os
import operator
from pathlib import Path

import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt

def plot_loss(losses, dir_path):
    plt.style.use('ggplot')
    x = np.linspace(1, len(losses), len(losses))
    fig, ax = plt.subplots()
    ax.plot(x, losses, color="cornflowerblue",
            label="Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend(loc="upper right")
    plt.savefig(dir_path / "train_history" / "loss_history.pdf", format="pdf", dpi=1200)


def model_snapshot(losses, model, filename, path_to_directory=None, run_number=0):
    path_to_directory = Path(path_to_directory)
    if operator.not_(path_to_directory.exists()):
        os.mkdir(path_to_directory)
    if operator.not_((path_to_directory/"train_history").exists()):
        os.mkdir(path_to_directory / "train_history")

    pd.DataFrame(losses).to_csv(
        path_to_directory/'train_history'/f'scaled_autoencoder_loss_{run_number}.csv', header=False, index=False)

    torch.save(model.state_dict(), path_to_directory / "train_history" / f'{filename}_runs_{run_number}.pth')

    plot_loss(losses, path_to_directory)
