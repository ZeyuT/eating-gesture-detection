import sys
import os
import numpy as np
import time
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchinfo import summary

from models import RES_LSTM
from utils import class_weights,FrameSequenceDataset,AverageMeter
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM
from tqdm import tqdm

def train_loop(dataloader, model, loss_fn,epoch):
    train_loss = AverageMeter()
    train_acc = AverageMeter() 
    with tqdm(dataloader,unit= "batch") as tepoch:
        for input, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            # Compute prediction and loss
            pred = model(input)
            loss = loss_fn(pred.permute(0, 2, 1), target)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update results
            cur_loss = loss.item()
            correct = (pred.argmax(-1) == target).sum().item()     
            cur_acc = correct/(target.size()[0]*target.size()[1])
            train_loss.update(cur_loss)
            train_acc.update(cur_acc) 
            tepoch.set_postfix(loss=f"{(cur_loss):>0.6f}", accuracy=f"{(100. * cur_acc):>0.1f}%")
    return train_loss.avg, train_acc.avg
            
def val_loop(dataloader, model, loss_fn):
    val_loss = AverageMeter()
    val_acc = AverageMeter()    
    with torch.no_grad():
        for (input, target) in dataloader:
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            pred = model(input)
            loss = loss_fn(pred.permute(0, 2, 1), target)
            #update results
            val_loss.update(loss.item())
            correct = (pred.argmax(-1) == target).type(torch.float).sum().item()
            val_acc.update(correct/(target.size()[0]*target.size()[1]))  
    return val_loss.avg, val_acc.avg
    
if __name__ == "__main__":  
    
    print("start")
    sys.stdout.flush() 
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
        video_num = 1
        batch_size = 16
        epochs = 5
        network = "RES_LSTM"
        weight_type = 0
    else:
        batch_size = int(sys.argv[2])
        epochs = int(sys.argv[3])
        network = sys.argv[4]
        seq_len = int(sys.argv[5])
        stride = int(sys.argv[6])
        video_num = int(sys.argv[7])
        weight_type = int(sys.argv[8])
    if network == "RES_LSTM":
        model_type = 1
    elif network == "CNN3D_Model":
        model_type = 2
        
    print("model: {}".format(network))
    print("batch size: {}  epochs: {}".format(batch_size, epochs))
    print("sequence length: {}  stride: {}\n".format(seq_len, stride))
    print("weight_type: {}\n".format(weight_type))
    sys.stdout.flush()

    #v{x}: version x for class weight calculation
    #bcb: batch class balance. needs to find a way to convert y_true to numpy. leave it on todo list.
    if train == 5:
        print("continue training models\n")
        log_loc = "./log_{}_{}_{}_{}_{}_v{}".format(network,30+epochs,seq_len,stride,video_num,weight_type)
        model_loc = "./model_{}_{}_{}_{}_{}_v{}".format(network,30+epochs,seq_len,stride,video_num,weight_type)
        test_loc = "./test_{}_{}_{}_{}_{}_v{}_60videos".format(network,30+epochs,seq_len,stride,video_num,weight_type)
    else:
        log_loc = "./log_{}_{}_{}_{}_{}_v{}_torch".format(network,epochs,seq_len,stride,video_num,weight_type)
        model_loc = "./model_{}_{}_{}_{}_{}_v{}_torch".format(network,epochs,seq_len,stride,video_num,weight_type)
        test_loc = "./test_{}_{}_{}_{}_{}_v{}_torch".format(network,epochs,seq_len,stride,video_num,weight_type)    
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
        
    print("Preparing training sample list...")
    start_time = time.time()
    sys.stdout.flush()
    
    if train == 2:
        train_video_list = ['p026_c1']
        test_video_list = ['p026_c2']
    else:
        video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith("p")]
        video_list.sort(reverse=False)
        video_list = video_list[0:video_num]
        train_video_list = video_list[0:int(video_num*0.70)]
        val_video_list = video_list[int(video_num*0.70):int(video_num*0.85)]
        test_video_list = video_list[int(video_num*0.85):]
        
    preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ColorJitter(brightness=0.4),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    train_set = FrameSequenceDataset(
            root_path=FRAME_LOC,
            video_list=train_video_list,
            seq_len=seq_len,
            stride=stride,
            model_type=model_type,
            transform=preprocess,
            test_mode=False
            )
    train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )  
        
    val_set = FrameSequenceDataset(
            root_path=FRAME_LOC,
            video_list=val_video_list,
            seq_len=seq_len,
            stride=stride,
            model_type=model_type,
            transform=None,
            test_mode=False
            )
    val_loader = data.DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )   

    test_set = FrameSequenceDataset(
            root_path=FRAME_LOC,
            video_list=test_video_list,
            seq_len=seq_len,
            stride=stride,
            model_type=model_type,
            transform=None,
            test_mode=False
            )
    test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )   
                 
    weights = class_weights(train_set.label_list,weight_type) 
    print("{} videos in training set".format(len(train_video_list)))
    print("{} patterns in training set".format(len(train_set)))
    print("class weights in training set: {}".format(weights)) 
    print("{} videos in validation set".format(len(val_video_list)))
    print("{} patterns in validation set".format(len(val_set)))
        
    print("training model...")
    sys.stdout.flush()
    model = RES_LSTM(seq_len=seq_len).cuda()
    summary(model, input_size=(batch_size,seq_len, CHANNEL, HEIGHT, WIDTH))     
    model = torch.nn.DataParallel(model).cuda()
   
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().cuda())
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    log = open(os.path.join(log_loc, "train_log.txt"), 'w')

    best_val_loss = 1000
    for epoch in range(epochs):
        train_loss, train_acc = train_loop(train_loader, model, loss_fn,epoch)
        print("validating...")   
        sys.stdout.flush() 
        val_loss, val_acc = val_loop(val_loader, model, loss_fn)
        
        output =  f"Epoch {epoch}:"\
                  f"train acc: {(100*train_acc):>0.1f}%   train loss: {train_loss:>8f}   "\
                  f"val acc: {(100*val_acc):>0.1f}%   val loss: {val_loss:>8f}\n"
        print(output)  
        sys.stdout.flush()  
        log.write(output) 
 
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                        },os.path.join(model_loc, "checkpoint.tar"))
            message = f"current model is the best; checkpoint saved\n"
            print(message) 
            sys.stdout.flush()  
            log.write(message)  
    log.close() 
    print("model training finished")
    """
    print("Testing model...")
    start_time = time.time()
    sys.stdout.flush()
                         
    test_model(model, test_video_list, test_loc, seq_len, model_type)
    
    elapsed_time = time.time() - start_time
    print("Test finished, elapsed time: {0:.6f} s".format(elapsed_time))
    """
