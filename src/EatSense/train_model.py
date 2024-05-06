import sys
import os
import numpy as np
import time
import datetime
import gc

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary

from models import RES_LSTM,RES_BILSTM
from utils import class_weights,FrameSequenceDataset,AverageMeter,RateMeter,test_model
from constants import DATA_LOC,RESULT_ROOT_LOC,CHANNEL,LABEL_NUM
from tqdm import tqdm
sys.path.append("../")
from reimplementation.model_loader import get_model
def main():   
    '''
    train = 0: tests model (i.e. inference) on test set
          = 1: main mode, train on train set, validate on val set, test on test set, and inference all
          = 2: train models in debug mode (training,valiating, and testing are all on train set)
          = 3: continue training on raw video data and test model
          = 4: inference on train set
          = 5: inference on val set
          = 6: re-test seq2seq model in seq2one manner on test set
    model_type  = 1: seq2seq frame-wise prediction
                = 2: seq2one frame prediction
    '''
    train = int(sys.argv[1])
    if train == 2:
        #for debugging
        seq_len = 16
        stride = 32
        batch_size = 8
        epochs = 3
        network = "RES_BILSTM"
        weight_type = 3
    else:
        batch_size = int(sys.argv[2])
        epochs = int(sys.argv[3])
        network = sys.argv[4]
        seq_len = int(sys.argv[5])
        stride = int(sys.argv[6])
        model_idx = int(sys.argv[7])
        weight_type = 3
        val_stride = 2
    if network in ["RES_BILSTM","RES_LSTM","x3d-s"]:
        model_type = 1
    elif network == "CNN3D_Model":
        model_type = 2

    learning_rate = 0.0001
    decay_rate = 0.9
    
    if train == 6:
        model_type = 2
        
    RESULT_LOC = os.path.join(RESULT_ROOT_LOC, f"results_10runs_{network}")
    if network == "RES_BILSTM":                
        model = RES_BILSTM(seq_len=seq_len)
        FRAME_LOC = os.path.join(DATA_LOC, "VideoData_eatSense_8hz/")
    elif network == "RES_LSTM":                
        model = RES_LSTM(seq_len=seq_len)
        FRAME_LOC = os.path.join(DATA_LOC, "VideoData_eatSense_8hz/")
    elif network == "x3d-s":
        model, _, _, fps, seq_len = get_model(network)
        FRAME_LOC = os.path.join(DATA_LOC, "VideoData_eatSense_5hz/")
    model = torch.nn.DataParallel(model).cuda()
    
    #v{x}: version x for class weight calculation
    if train == 2:
        print("training models in debug mode\n")
        log_loc = f"log_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}_debug"
        model_loc = f"model_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}_debug"
        test_loc = f"result_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}_debug"
    elif train == 3:
        print("continue training models\n")
        log_loc = os.path.join(RESULT_LOC,f"log_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
        model_loc = os.path.join(RESULT_LOC,f"model_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
        test_loc = os.path.join(RESULT_LOC,f"result_{network}_{30+epochs}_{seq_len}_{stride}_v{weight_type}")
    else:
        log_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"log_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}")
        model_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"model_{network}_{epochs}_{seq_len}_{stride}_v{weight_type}")
        if train == 4:
            print("test models on train set \n")
            test_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_train")
        elif train == 5:
            print("test models on val set \n")
            test_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_val")
        elif train == 6:
            print("test models on test set using seq2one manner \n")
            test_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_test_seq2one") 
        else:
            test_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_test") 
    
    try: 
        os.mkdir(RESULT_LOC)
    except:
        pass  
    try: 
        os.mkdir(os.path.join(RESULT_LOC,f"{model_idx}"))
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
     
    if train != 0 and train < 4:      
        print("Preparing dataset...")
        sys.stdout.flush()
        train_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("202")]
        val_video_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("202")]
        
        preprocess = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.ColorJitter(brightness=0.4),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        train_set = FrameSequenceDataset(
                    root_path=FRAME_LOC+"train_set/",
                    video_list=train_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=preprocess,
                    test_mode=False
                    )
        if train == 2:
            train_video_list = [train_video_list[1]]
            val_video_list = train_video_list
            test_video_list = train_video_list
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
            val_set = FrameSequenceDataset(
                        root_path=FRAME_LOC+"val_set/",
                        video_list=val_video_list,
                        seq_len=seq_len,
                        stride=val_stride,
                        model_type=model_type,
                        transform=None,
                        test_mode=True
                        )     
                                   
        train_loader = data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=False
            )  # set pin_memory to false to reduce memory usage
        val_loader = data.DataLoader(
                dataset=val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=False
            )                
        log = open(os.path.join(log_loc, "train_log.txt"), 'a')    
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
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        best_val_uar = 0
        for epoch in range(epochs):
            start_time = time.time()
            message = f"Epoch {epoch}   lr: {scheduler.get_last_lr()[0]:>6f}"
            print(message)
            sys.stdout.flush() 
            log.write(message+ '\n') 
            train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, 3000)
            epoch_time = time.time() - start_time
            start_time = time.time()
            print("validating...") 
            sys.stdout.flush() 
            val_loss, val_acc, val_uar = val_loop(val_loader, model, loss_fn)        
            message = f"epoch duration: {epoch_time:>0.1f}s   val duration: {time.time()-start_time:>0.1f}s\n" \
                    f"train acc: {(100*train_acc):>0.1f}%   train loss: {train_loss:>8f}\n" \
                    f"val acc: {(100*val_acc):>0.1f}%   val loss: {val_loss:>8f}   val uar: {(100*val_uar):>0.1f}%"
            print(message)  
            log.write(message+ '\n') 
            sys.stdout.flush()  
            log.flush()   
            if (epoch+1) %10 == 0:
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
            gc.collect() 

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
           
    #'''
    print("testing model...")
    start_time = time.time()
    sys.stdout.flush()
    if train == 1:
        # Test and encode training set
        test_model(model = model, 
                  test_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("202")], 
                  root_path = os.path.join(FRAME_LOC,"train_set/"),
                  test_save_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_train"), 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
        # Test and encode test set
        test_model(model = model, 
                  test_video_list = [f for f in os.listdir(FRAME_LOC+"test_set") if f.startswith("202")], 
                  root_path = os.path.join(FRAME_LOC,"test_set/"),
                  test_save_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_test"), 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
        # Test and encode val set
        test_model(model = model, 
                  test_video_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("202")], 
                  root_path = os.path.join(FRAME_LOC,"val_set/"),
                  test_save_loc = os.path.join(RESULT_LOC,f"{model_idx}",f"result_val"), 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
    elif train == 2:
        train_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("202")]
        test_model(model = model, 
                  test_video_list = train_video_list, 
                  root_path = os.path.join(FRAME_LOC,"train_set/"),
                  test_save_loc = test_loc, 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
    elif train < 4 or train == 6:
        test_video_list = [f for f in os.listdir(FRAME_LOC+"test_set") if f.startswith("202")] 
        test_model(model = model, 
                  test_video_list = test_video_list, 
                  root_path = os.path.join(FRAME_LOC,"test_set/"),
                  test_save_loc = test_loc, 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
    elif train == 4:
        train_video_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("202")]
        test_model(model = model, 
                  test_video_list = train_video_list, 
                  root_path = os.path.join(FRAME_LOC,"train_set/"),
                  test_save_loc = test_loc, 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
    elif train == 5:
        val_video_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("202")]
        test_model(model = model, 
                  test_video_list = val_video_list, 
                  root_path = os.path.join(FRAME_LOC,"val_set/"),
                  test_save_loc = test_loc, 
                  seq_len = seq_len, 
                  model_type = model_type,
                  test_batch_size = 32,
                  test_stride=1)
    elapsed_time = time.time() - start_time
    print(f"Test finished, elapsed time: {elapsed_time:>6f} s")
    #'''

def train_loop(dataloader, model, loss_fn, optimizer, maximum_step = 10000):
    model.train()
    train_loss = AverageMeter()
    train_acc = RateMeter() 
    # with tqdm(dataloader,unit= "batch",total=maximum_step, leave=False) as tepoch:
        # for step, (input_, target) in enumerate(dataloader):
    for step, (input_, target) in enumerate(dataloader):
        input_ = input_.type(torch.cuda.FloatTensor)
        input_ = Variable(input_).cuda()
        target = Variable(target).cuda()
        # Compute prediction and loss
        output,_ = model(input_)
        output = output.permute(0, 2, 1)
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
        # tepoch.set_postfix(loss=f"{(cur_loss):>0.6f}", accuracy=f"{(100*cur_acc):>0.1f}%")
        # Maximum train step per epoch is maximum_step
        if step >= maximum_step:
            break
    gc.collect() 
    return train_loss.avg, train_acc.rate
            
def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(LABEL_NUM)]  
    with torch.no_grad():
        for (input_, target) in dataloader:
            input_ = input_.type(torch.cuda.FloatTensor)
            input_ = Variable(input_).cuda()
            target = Variable(target).cuda()
            output,_ = model(input_)
            output = output.permute(0, 2, 1)
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
    gc.collect() 
    return val_loss.avg, val_acc.rate, val_uar

    
if __name__ == "__main__":  
    main()