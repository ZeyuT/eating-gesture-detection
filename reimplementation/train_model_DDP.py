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

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchinfo import summary

from models import cnn_lstm, slowfast, x3d
from utils import FrameSequenceDataset,AverageMeter,RateMeter,test_model
from model_loader import get_model
from constants import DATA_LOC,RESULT_LOC,IMAGE_SIZES,CHANNEL,LABEL_NUM
from tqdm.auto import tqdm

def main():   
    '''
    train = 0: only tests model (i.e. inference) on test set
          = 1: train models on train set, and test models on test set
    '''
    parser = argparse.ArgumentParser()
    # Need this argument to make DDP work
    parser.add_argument('--local_rank', type=int, help="local gpu id")
    parser.add_argument('--train_mode', type=int, help="1 for training, 0 for testing")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--network', help="network name")
   
    args = parser.parse_args()
    
    train = args.train_mode
    batch_size = args.batch_size
    network = args.network

    epochs = 50
    seq_len = 16
    stride = 8
    
    log_loc = os.path.join(RESULT_LOC,f"log_{network}_{epochs}_{seq_len}_{stride}")
    model_loc = os.path.join(RESULT_LOC,f"model_{network}_{epochs}_{seq_len}_{stride}")
    test_loc = os.path.join(RESULT_LOC,f"result_{network}_{epochs}_{seq_len}_{stride}") 
    
    try: 
        os.mkdir(RESULT_LOC)
    except:
        pass     
    try:
        os.mkdir(log_loc)
    except:
        pass
    try:
        os.mkdir(model_loc)
    except:
        pass
    try:
        os.mkdir(test_loc)
    except:
        pass
    try:
        os.mkdir(os.path.join(test_loc,"frame_probs"))
    except:
        pass
    try:
        os.mkdir(os.path.join(test_loc,"frame_preds"))
    except:
        pass      
    
    model, model_type, inference_type, fps = get_model(network).cuda()
    FRAME_LOC = os.path.join(DATA_LOC, f"VideoData_rawFrames_{fps}hz")    
    learning_rate = 0.0001
    decay_rate = 0.9
    if train:      
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        global_rank = dist.get_rank()   
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
   
        print("Preparing dataset...")
        sys.stdout.flush()
        train_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("p")]
        val_video_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("p")]
        
        preprocess = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.ColorJitter(brightness=0.4),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        train_set = FrameSequenceDataset(
                    root_path=os.path.join(FRAME_LOC,"train_set/"),
                    video_list=train_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=preprocess,
                    test_mode=False
                    )
        val_set = FrameSequenceDataset(
                    root_path=os.path.join(FRAME_LOC,"val_set/"),
                    video_list=val_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=None,
                    test_mode=True
                    )     
        
        sampler = DistributedSampler(trainset)                                   
        train_loader = data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                num_workers=16,
                pin_memory=True,
                sampler=sampler
            )  
        val_loader = data.DataLoader(
                dataset=val_set,
                batch_size=batch_size*8,
                shuffle=False,
                num_workers=16,
                pin_memory=True
            )                
             
        log = open(os.path.join(log_loc, "train_log.txt"), 'w')    
        print ("=======================Experimental Settings=======================")
        log.write("=======================Experimental Settings=======================\n")
        log.flush()
        sys.stdout.flush() 
        available_gpu = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"started at {start_time}\n"\
                  f"available GPU: {available_gpu}\n"\
                  f"model: {network}\n" \
                  f"batch size: {batch_size}  epochs: {epochs}\n"\
                  f"sequence length: {seq_len}  stride: {stride}"
        print(message)
        log.write(message+"\n")
        sys.stdout.flush()
        log.flush()
                    
        weights,class_counts = class_weights(train_set.label_list) 
        val_weights,val_class_counts = class_weights(val_set.label_list) 
        message = f"training set: {len(train_video_list)} videos -> {len(train_set)} patterns\n" \
                  f"training set class counts: {class_counts}\n" \
                  f"training set class ratio: {weights}\n"\
                  f"validation set: {len(val_video_list)} videos -> {len(val_set)} patterns\n" \
                  f"validation set class counts: {val_class_counts}\n"  \
                  f"validation set class ratio: {val_weights}"
        print(message)
        log.write(message+"\n")
        print ("===================================================================\n")
        log.write("===================================================================\n")
        sys.stdout.flush()
        log.flush()
        sys.stdout.flush()   
        
        train(model=model,
              train_loader=train_loader,
              val_loader=val_loader,
              sampler=sampler,
              log=log,
              model_loc=model_loc,
              epochs=epochs,
              learning_rate=learning_rate,
              decay_rate=decay_rate
             )        
        log.close()                

        print("model training finished")
        
    print("load the best model for testing")
    sys.stdout.flush()
    try:
        checkpoint = torch.load(os.path.join(model_loc, f"checkpoint_best.tar"))
    except:
        print("no 'checkpoint_best.tar' found")
        exit(0)
    model.load_state_dict(checkpoint['model_state_dict'])
           
    print("testing model...")
    start_time = time.time()
    sys.stdout.flush()
    test_video_list = [f for f in os.listdir(FRAME_LOC+"test_set") if f.startswith("p")] 
    test_model(model = model, 
                test_video_list = test_video_list, 
                root_path = os.path.join(FRAME_LOC,"test_set/"),
                test_save_loc = test_loc, 
                seq_len = seq_len, 
                model_type = model_type,
                inference_type = inference_type,
                test_batch_size = 64,
                test_stride=1)
    
    elapsed_time = time.time() - start_time
    print(f"Test finished, elapsed time: {elapsed_time:>6f} s")


            
if __name__ == "__main__":  
    main()