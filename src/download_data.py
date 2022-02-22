# author: Artan Zandian
# date: 2021-02-18

"""Downloads a Kaggle dataset.

Usage: download_data.py --dataset=<dataset> --file_path=<file_path>

Options:
--dataset=<dataset>          The dataset of the data file to download.
--file_path=<file_path>      File path (including file name with extension) to store the file
"""

import os
import kaggle
from docopt import docopt

opt = docopt(__doc__)


def main(dataset, file_path):
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset,
            path=file_path,
            unzip=True,
        )
    except:
        os.makedirs(os.path.dirname(file_path))
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset,
            path=file_path,
            unzip=True,
        )


if __name__ == "__main__":
    main(opt["--dataset"], opt["--file_path"])
