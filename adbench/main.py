import numpy as np
import torch
from adbench.baseline.Customized.run import Customized

from src.data.dataset_loader import load_data
from src.data.dataset_type import DatasetType
from src.utils.ImageFlattenerUtility import extract_and_flatten_images_dataset_3d, unflatten_images_3d
from src.utils.preprocessing import normalize_images
from adbench.run import RunPipeline
from adbench.myutils import Utils


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    #utils = Utils()
    #utils.download_datasets()
    #dataset = np.load("../.venv/lib/python3.9/site-packages/adbench/datasets/CV_by_ResNet18/MVTec-AD_bottle.npz",
                      #allow_pickle=True)

    run = RunPipeline(suffix="ADBench", parallel="unsupervise")

    x_train = load_data(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256))
    y_train = torch.zeros(len(x_train))
    x_test, y_test  = load_data(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256), train=False)
    y_test = torch.Tensor(y_test)

    x_train = extract_and_flatten_images_dataset_3d(x_train).to("cpu")
    x_train = torch.from_numpy(normalize_images(x_train.numpy())).to(torch.float32)
    x_test = extract_and_flatten_images_dataset_3d(x_test).to("cpu")
    x_test = torch.from_numpy(normalize_images(x_test.numpy())).to(torch.float32)

    x = torch.cat((x_train, x_test), dim=0).numpy()
    y = torch.cat((y_train, y_test), dim=0).numpy()

    dataset = {}
    dataset["X"] = x
    dataset["y"] = y
    pipeline=RunPipeline(suffix="ADBench", parallel="unsupervise")
    results = pipeline.run(dataset)
    print(results)
