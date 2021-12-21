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
from tqdm import tqdm

sys.path.append("../")
from reimplementation.models import cnn_lstm, slowfast, x3d
from reimplementation.utils import FrameSequenceDataset,AverageMeter,RateMeter,test_model
from reimplementation.model_loader import get_model
from reimplementation.constants import DATA_LOC,RESULT_LOC,IMAGE_SIZES,LABEL_NUM

def main():   
    '''
    train = 0: only tests model (i.e. inference) on test set
          = 1: only train models on train set
          = 2: resume training
    '''
    train = int(sys.argv[1])
    network = sys.argv[2]
    start_epoch = int(sys.argv[3])
    end_epoch = int(sys.argv[4])

    batch_size = 8
    val_batch_size = 32
    
    # learning_rate = 0.005
    # decay_rate = 0.8
    
    #learning_rate = 0.001
    #decay_rate = 0.9
    learning_rate = 0.0001
    decay_rate = 0.9  
    
    model, model_type, inference_type, fps, seq_len = get_model(network)
    stride = int(fps*1) # sample stride is 1 sec for training
    val_stride = int(fps*2) # sample stride is 2 sec for validating
    model = torch.nn.DataParallel(model).cuda()
    FRAME_LOC = os.path.join(DATA_LOC, f'VideoData_independent_{fps}hz')    
    log_loc = os.path.join(RESULT_LOC,f'log_{network}_{seq_len}_{stride}')
    model_loc = os.path.join(RESULT_LOC,f'model_{network}_{seq_len}_{stride}')
    test_loc = os.path.join(RESULT_LOC,f'result_{network}_{seq_len}_{stride}') 
    
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
        os.mkdir(os.path.join(test_loc,'frame_probs'))
    except:
        pass
    try:
        os.mkdir(os.path.join(test_loc,'frame_preds'))
    except:
        pass      

    if train:      
        print('Preparing dataset...')
        sys.stdout.flush()
        train_video_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'train_set')) if f.startswith('p')]
        val_video_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'val_set')) if f.startswith('p')]
        
        preprocess = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.ColorJitter(brightness=0.4),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        train_set = FrameSequenceDataset(
                    root_path=os.path.join(FRAME_LOC,'train_set/'),
                    video_list=train_video_list,
                    seq_len=seq_len,
                    stride=stride,
                    model_type=model_type,
                    transform=preprocess,
                    test_mode=False
                    )
        val_set = FrameSequenceDataset(
                    root_path=os.path.join(FRAME_LOC,'val_set/'),
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
                num_workers=16,
                pin_memory=True
            )  
        val_loader = data.DataLoader(
                dataset=val_set,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True
            )                
             
        log = open(os.path.join(log_loc, f'train_log_{start_epoch}to{end_epoch}.txt'), 'w')    
        print ('=======================Experimental Settings=======================')
        log.write('=======================Experimental Settings=======================\n')
        log.flush()
        sys.stdout.flush() 
        available_gpu = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
        
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f'started at {start_time}\n'\
                  f'available GPU: {available_gpu}\n'\
                  f'model: {network} batch size: {batch_size}\n' \
                  f'start epoch: {start_epoch} end epoch: {end_epoch}\n'\
                  f'sequence length: {seq_len}  stride: {stride}'
        print(message)
        log.write(message+'\n')
        sys.stdout.flush()
        log.flush()
                    
        weights,class_counts = class_weights(train_set.label_list) 
        val_weights,val_class_counts = class_weights(val_set.label_list) 
        message = f'training set: {len(train_video_list)} videos -> {len(train_set)} patterns\n' \
                  f'training set class counts: {class_counts}\n' \
                  f'training set class ratio: {weights}\n'\
                  f'validation set: {len(val_video_list)} videos -> {len(val_set)} patterns\n' \
                  f'validation set class counts: {val_class_counts}\n'  \
                  f'validation set class ratio: {val_weights}'
        print(message)
        log.write(message+'\n')
        print ('===================================================================\n')
        log.write('===================================================================\n')
        sys.stdout.flush()
        log.flush()
        sys.stdout.flush()    
            
        print('training model...')
        # Initialize the loss function, optimizer
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().cuda())
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        
        # If this is to resume training, load states from the last epoch
        if start_epoch != 0:
            checkpoint = torch.load(os.path.join(model_loc, f'checkpoint_{start_epoch-1}.tar'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_checkpoint = torch.load(os.path.join(model_loc, 'checkpoint_best.tar'))
            best_val_uar = best_checkpoint['val_uar']
        else:
            best_val_uar = float('-inf')
            
        
        for epoch in range(start_epoch,end_epoch):
            message = f'Epoch {epoch}   lr: {scheduler.get_last_lr()[0]:>6f}'
            print(message)
            sys.stdout.flush() 
            log.write(message+ '\n') 
            train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer)
            print('validating...')   
            sys.stdout.flush() 
            start_time = time.time()
            val_loss, val_acc, val_uar = val_loop(val_loader, model, loss_fn)  
            elapsed_time = time.time() - start_time
            print(f'validation finished, elapsed time: {elapsed_time:>6f} s')
        
            message = f'train acc: {(100*train_acc):>0.1f}%   train loss: {train_loss:>8f}\n' \
                      f'val acc: {(100*val_acc):>0.1f}%   val loss: {val_loss:>8f}   val uar: {(100*val_uar):>0.1f}%'
            print(message)  
            log.write(message+ '\n') 
            sys.stdout.flush()  
            log.flush()   
            if (epoch+1) %5 == 0:
                torch.save({'model_state_dict': model.state_dict(), 
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            },os.path.join(model_loc, f'checkpoint_{epoch}.tar'))  
            if best_val_uar < val_uar:
                best_val_uar = val_uar
                message = f'current model is the best; checkpoint saved'
                print(message) 
                log.write(message+ '\n')   
                sys.stdout.flush()  
                log.flush()          
                torch.save({'model_state_dict': model.state_dict(), 
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'val_uar': best_val_uar,
                            },os.path.join(model_loc, f'checkpoint_best.tar'))   
            scheduler.step()
        log.close() 
            
        print('model training finished')
    
    elif train == 0:
        print('load the best model for testing')
        sys.stdout.flush()
        try:
            checkpoint = torch.load(os.path.join(model_loc, f'checkpoint_best.tar'))
        except:
            print("no 'checkpoint_best.tar' found")
            exit(0)
        model.load_state_dict(checkpoint['model_state_dict'])

        print('testing model...')
        start_time = time.time()
        sys.stdout.flush()
        test_video_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'test_set')) if f.startswith('p')] 
        test_model(model = model, 
                    test_video_list = test_video_list, 
                    root_path = os.path.join(FRAME_LOC,'test_set/'),
                    test_save_loc = test_loc, 
                    seq_len = seq_len, 
                    model_type = model_type,
                    inference_type = inference_type,
                    test_batch_size = val_batch_size,
                    test_stride=1)

        elapsed_time = time.time() - start_time
        print(f'Test finished, elapsed time: {elapsed_time:>6f} s')

def class_weights(label_list):
    class_counts = []
    for label in range(LABEL_NUM):
        class_counts.append(np.sum(label_list==label))
    class_counts = np.array(class_counts)
    total = np.sum(class_counts)
    weights = []
    for i in range(LABEL_NUM):
        if class_counts[i] == 0:
            weights.append(0)
        else:
            weights.append(1/class_counts[i]**0.5)
            #weights.append(total/(class_counts[i]*LABEL_NUM))
    weights = np.array(weights) 
    weights = weights/np.sum(weights)
    return weights, class_counts

def batch_class_weights(label_list):
    class_counts = []
    for label in range(LABEL_NUM):
        class_counts.append(torch.sum(label_list==label))
    class_counts = torch.FloatTensor(class_counts)
    weights = []
    #total = torch.sum(class_counts)
    for i in range(LABEL_NUM):
        if class_counts[i] == 0:
            weights.append(0)
        else:
            weights.append(1/class_counts[i]**0.5)
            #weights.append(total/(class_counts[i]*LABEL_NUM))
    weights = torch.FloatTensor(weights) 
    weights = weights/torch.sum(weights)
    return weights

def train_loop(dataloader, model, loss_fn, optimizer, maximum_step = 10000):
    model.train()
    train_loss = AverageMeter()
    train_acc = RateMeter() 
    with tqdm(dataloader,unit= 'batch', total=maximum_step) as tepoch:
        for step, (input, target) in enumerate(tepoch):
            # Make predictions
            input = input.type(torch.cuda.FloatTensor)
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            # Compute prediction and loss
            output = model(input)
            if output.dim() > 2:
                # output is frame-wise prediction
                # then change the dimension form [-1,T,C] to [-1,C,T], to fit in loss function
                output = output.permute(0, 2, 1)           
            # Backpropagation           
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()            
            #update results
            pred = output.argmax(1)
            correct = (pred == target).sum().item()     
            if target.dim() == 1:
                total_num = target.size()[0]
            else:
                total_num = target.size()[0]*target.size()[1]
            cur_acc = correct/total_num
            train_loss.update(cur_loss)
            train_acc.update(correct,total_num) 
            tepoch.set_postfix(loss=f'{(cur_loss):>0.6f}', accuracy=f'{(100*cur_acc):>0.1f}%')            
            # Maximum train step per epoch is maximum_step
            if step >= maximum_step:
                break
        del input,target,output,loss
        torch.cuda.empty_cache()
    return train_loss.avg, train_acc.rate
            
def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(LABEL_NUM)]  
    with torch.no_grad():
        for step, (input, target) in enumerate(dataloader):
            input = input.type(torch.cuda.FloatTensor)
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            output = model(input)
            if output.dim() > 2:
                # output is frame-wise prediction
                # then change the dimension form [-1,T,C] to [-1,C,T], to fit in loss function
                output = output.permute(0, 2, 1)              
            #update results
            loss = loss_fn(output, target)
            val_loss.update(loss.item())
            pred = output.argmax(1)
            correct = (pred == target).sum().item()
            for label in range(LABEL_NUM):
                cur_tp = torch.logical_and(pred==target, target==label).sum().item()
                cur_p = (target==label).sum().item()
                val_tpr[label].update(cur_tp,cur_p) 
            if target.dim() == 1:
                total_num = target.size()[0]
            else:
                total_num = target.size()[0]*target.size()[1]
            val_acc.update(correct,total_num)  
        del input,target,output,loss
        torch.cuda.empty_cache()
    val_uar = np.sum([val_tpr[label].rate for label in range(LABEL_NUM)]) / LABEL_NUM   
    return val_loss.avg, val_acc.rate, val_uar

    
if __name__ == '__main__':  
    main()