import sys
import os
import numpy as np
import time
import datetime

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary

from models import RES_LSTM
from utils import class_weights,FrameSequenceDataset,AverageMeter,RateMeter,test_model
from constants import FRAME_LOC,RESULT_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM
from tqdm import tqdm

def main():   
    '''
    train = 0: test model only
          = 1: train and test model on raw video data
          = 2: debug mode on simulated data
          = 3: train and test model on raw simulated data
          = 4: test model on raw simulated data
          = 5: continue training on raw video data and test model
    model_type  = 1: seq2seq frame-wise prediction
                = 2: seq2one frame prediction
    '''
    train = int(sys.argv[1])
    if train == 2:
        #for debugging
        seq_len = 16
        stride = 8
        batch_size = 16
        epochs = 3
        network = "RES_LSTM"
        weight_type = 3
    else:
        batch_size = int(sys.argv[2])
        epochs = int(sys.argv[3])
        network = sys.argv[4]
        seq_len = int(sys.argv[5])
        stride = int(sys.argv[6])
        weight_type = int(sys.argv[7])
    if network == "RES_LSTM":
        model_type = 1
    elif network == "CNN3D_Model":
        model_type = 2

    #v{x}: version x for class weight calculation
    if train == 2:
        print("training models in test mode\n")
        log_loc = f"log_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}"
        model_loc = f"model_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}"
        test_loc = f"test_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}"
    elif train == 5:
        print("continue training models\n")
        log_loc = os.path.join(RESULT_LOC,f"log_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
        model_loc = os.path.join(RESULT_LOC,f"model_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
        test_loc = os.path.join(RESULT_LOC,f"test_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
    else:
        log_loc = os.path.join(RESULT_LOC,f"log_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}")
        model_loc = os.path.join(RESULT_LOC,f"model_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}")
        test_loc = os.path.join(RESULT_LOC,f"test_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}")
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
        
    print("Preparing dataset...")
    sys.stdout.flush()
    train_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("p")]
    val_video_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("p")]
    test_video_list = [f for f in os.listdir(FRAME_LOC+"test_set") if f.startswith("p")] 
    
    preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ColorJitter(brightness=0.4),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])  
    if train == 2:
        train_video_list = [train_video_list[1]]
        val_video_list = train_video_list
        test_video_list = train_video_list
        train_set = FrameSequenceDataset(
                root_path=FRAME_LOC+"train_set/",
                video_list=train_video_list,
                seq_len=seq_len,
                stride=stride,
                model_type=model_type,
                transform=preprocess,
                test_mode=False
                )
                        
        val_set = FrameSequenceDataset(
                root_path=FRAME_LOC+"train_set/",
                video_list=train_video_list,
                seq_len=seq_len,
                stride=stride,
                model_type=model_type,
                transform=None,
                test_mode=True
                )
    else:            
        train_set = FrameSequenceDataset(
                    root_path=FRAME_LOC+"train_set/",
                    video_list=train_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=preprocess,
                    test_mode=False
                    )
                            
        val_set = FrameSequenceDataset(
                    root_path=FRAME_LOC+"val_set/",
                    video_list=val_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=None,
                    test_mode=True
                    )     
                               
    train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )  
    val_loader = data.DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )                
 
    model = RES_LSTM(seq_len=seq_len).cuda()
    model = torch.nn.DataParallel(model).cuda()
    #summary(model, input_size=(batch_size,seq_len, CHANNEL, HEIGHT, WIDTH))     
    if train != 0:    
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
                  f"sequence length: {seq_len}  stride: {stride}\n"\
                  f"weight_type: {weight_type}"
        print(message)
        log.write(message+"\n")
        sys.stdout.flush()
        log.flush()
        
        weights,class_counts = class_weights(train_set.label_list,weight_type) 
        _,val_class_counts = class_weights(val_set.label_list,weight_type) 
        message = f"training set: {len(train_video_list)} videos -> {len(train_set)} patterns\n" \
                  f"training set class counts: {class_counts}\n" \
                  f"training set class weights: {weights}\n"\
                  f"validation set: {len(val_video_list)} videos -> {len(val_set)} patterns\n" \
                  f"validation set class counts: {val_class_counts}" 
        print(message)
        log.write(message+"\n")
        print ("===================================================================\n")
        log.write("===================================================================\n")
        sys.stdout.flush()
        log.flush()
        sys.stdout.flush()    
        
        print("training model...")
        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().cuda())
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        best_val_uar = 0
        for epoch in range(epochs):
            message =  f"Epoch {epoch}   lr: {scheduler.get_last_lr()[0]:>6f}"
            print(message)
            sys.stdout.flush() 
            log.write(message+ '\n') 
            train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer)
            print("validating...")   
            sys.stdout.flush() 
            val_loss, val_acc, val_uar = val_loop(val_loader, model, loss_fn)        
            message = f"train acc: {(100*train_acc):>0.1f}%   train loss: {train_loss:>8f}\n" \
                      f"val acc: {(100*val_acc):>0.1f}%   val loss: {val_loss:>8f}   val uar: {(100*val_uar):>0.1f}%"
            print(message)  
            log.write(message+ '\n') 
            sys.stdout.flush()  
            log.flush()   
            torch.save({'model_state_dict': model.state_dict(), 
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                        },os.path.join(model_loc, f"checkpoint_{epoch}.tar"))  
            if best_val_uar < val_uar:
                best_val_uar = val_uar
                message = f"current model is the best; checkpoint saved"
                print(message) 
                log.write(message+ '\n')   
                sys.stdout.flush()  
                log.flush()          
                torch.save({'model_state_dict': model.state_dict(), 
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict()
                            },os.path.join(model_loc, f"checkpoint_best.tar"))   
            scheduler.step()
        log.close() 
        print("model training finished")
    print("load the best model for testing")
    sys.stdout.flush()
    try:
        checkpoint = torch.load(os.path.join(model_loc, f"checkpoint_best.tar"))
    except:
        checkpoint = torch.load(os.path.join(model_loc, f"checkpoint_29.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
           
    #'''
    print("testing model...")
    if train == 2:
        root_path = os.path.join(FRAME_LOC,"train_set/")
    else:
        root_path = os.path.join(FRAME_LOC,"test_set/")
    start_time = time.time()
    sys.stdout.flush()
    test_model(model = model, 
              test_video_list = test_video_list, 
              root_path = root_path,
              test_save_loc = test_loc, 
              seq_len = seq_len, 
              model_type = model_type,
              test_batch_size = 100,
              test_stride=1)
    elapsed_time = time.time() - start_time
    print(f"Test finished, elapsed time: {elapsed_time:>6f} s")
    #'''

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_acc = RateMeter() 
    with tqdm(dataloader,unit= "batch") as tepoch:
        for input, target in tepoch:
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            # Compute prediction and loss
            output = model(input).permute(0, 2, 1)
            loss = loss_fn(output, target)
            pred = output.argmax(1)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            #update results
            correct = (pred == target).sum().item()     
            cur_acc = correct/(target.size()[0]*target.size()[1])
            train_loss.update(cur_loss)
            train_acc.update(correct,target.size()[0]*target.size()[1]) 
            tepoch.set_postfix(loss=f"{(cur_loss):>0.6f}", accuracy=f"{(100*cur_acc):>0.1f}%")
    return train_loss.avg, train_acc.rate
            
def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(LABEL_NUM)]  
    with torch.no_grad():
        for (input, target) in dataloader:
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            output = model(input).permute(0, 2, 1)
            loss = loss_fn(output, target)
            pred = output.argmax(1)
            val_loss.update(loss.item())
            correct = (pred == target).sum().item()
            #update results
            for label in range(LABEL_NUM):
                cur_tp = torch.logical_and(pred==target, target==label).sum().item()
                cur_p = (target==label).sum().item()
                val_tpr[label].update(cur_tp,cur_p) 
            val_acc.update(correct,target.size()[0]*target.size()[1])  
    val_uar = np.sum([val_tpr[label].rate for label in range(LABEL_NUM)]) / LABEL_NUM   
    return val_loss.avg, val_acc.rate, val_uar

    
if __name__ == "__main__":  
    main()