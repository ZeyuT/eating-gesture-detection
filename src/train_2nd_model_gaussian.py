import sys
import os
import numpy as np
import time
import datetime
import argparse

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary

from utils import class_weights,RateMeter,AverageMeter
from constants import LABEL_TABLE,LABEL_NUM, RESULT_ROOT_LOC, DATA_LOC
from tqdm.auto import tqdm
import multiprocessing as mp

from train_2nd_model_control import run

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--train", 
                        type = int, 
                        help = "1: train and test, 0: test only",
                        default = 1)
    parser.add_argument('-l', "--first_network", 
                        type = str, 
                        help = "The local network name. Default: x3d-s",
                        default = 'x3d-s')
    parser.add_argument('-g', "--network", 
                        type = str, 
                        help = "The global detector name. Default: double_lstm",
                        default = "double_lstm")

    args = vars(parser.parse_args())
    
    
    label_type = 'all'
    batch_size = 32
    epochs = 50
    weight_type = 4
    
    for run_idx in range(5):
        run(train = args['train'],
            label_type = label_type,
            first_network = args['first_network'],
            network = args['network'],
            root_data_loc = DATA_LOC,
            root_result_loc = os.path.join(RESULT_ROOT_LOC, "gaussian", f"{args['first_network']}", f"{run_idx+1}"),
            batch_size = 32,
            epochs = 50,
            weight_type = 4,
            run_idx = run_idx+1,
            add_noise=True)