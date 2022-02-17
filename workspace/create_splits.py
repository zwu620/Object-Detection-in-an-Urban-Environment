import argparse
import glob
import os
import random

import numpy as np
import math
import shutil

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    # 80% for training, 15% for validation, 5% for testing
    raw_data_path = os.path.join(data_dir, "training_and_validation")
    raw = glob.glob(raw_data_path + "/*.tfrecord")
    tf_files = [files for files in raw]
    np.random.shuffle(tf_files)
    total = len(tf_files)
    train = math.floor(0.8 * total)
    val = math.floor(0.15 * total)
    
    trainf = os.path.join(data_dir, "train")
    valf = os.path.join(data_dir, "val")
    testf = os.path.join(data_dir, "test")
    
    for index, file in enumerate (tf_files):
        files = os.path.join(data_dir, file)
        if index < train:
            shutil.move (files, trainf)
        elif index < train + val:
            shutil.move (files, valf)
        else:
            shutil.move (files, testf)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)