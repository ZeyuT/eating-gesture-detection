import sys
import os
import numpy as np
import time

import torch.utils.data as data
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import AverageMeter,RateMeter,test_model
from constants import FRAME_LOC,RESULT_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM


def train(model,
          train_loader,
          val_loader,
          sampler,
          log,
          model_loc,
          epochs=50,
          learning_rate=0.0001,
          decay_rate=0.9
         )
    print("training model...")
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().cuda())
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)
    best_val_uar = float('-inf')
    
    for epoch in range(epochs):
        sampler.set_epoch(e)
        
        message = f"Epoch {epoch}   lr: {scheduler.get_last_lr()[0]:>6f}"
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
        
        if global_rank == 0 and (epoch+1) %10 == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save({'model_state_dict': model.state_dict(), 
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                        },os.path.join(model_loc, f"checkpoint_{epoch+1}.tar"))  
        
        if best_val_uar < val_uar:
            best_val_uar = val_uar
            if global_rank == 0:
                torch.save({'model_state_dict': model.state_dict(), 
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict()
                            },os.path.join(model_loc, f"checkpoint_best.tar"))   
            message = f"current model is the best; checkpoint saved"
            print(message) 
            log.write(message+ '\n')   
            sys.stdout.flush()  
            log.flush()       

        scheduler.step()

    
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
    word_size = dist.get_world_size()
    global_rank = dist.get_rank()   
    with tqdm(dataloader,unit= "batch", total=maximum_step) as tepoch:
        for step, (input, target) in enumerate(tepoch):
            # Make predictions
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            # Compute prediction and loss
            output = model(input)
            if output.dim() > 2:
                # output is frame-wise prediction
                # then change the dimension form [-1,T,C] to [-1,C,T], to fit in loss function
                output = output.permute(0, 2, 1)    
            pred = output.argmax(1)
            # Backpropagation           
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            dist.barrier(device_ids=[torch.cuda.current_device()])

            # Update training results            
            pred,target = all_gather_results(pred,target)
            dist.all_reduce(cur_loss, async_op=False)
            cur_loss = loss.item()
            cur_loss /= world_size
           
            correct_num = (pred == target).sum().item()     
            if target.dim() == 1:
                total_num = target.size()[0]
            else:
                total_num = target.size()[0]*target.size()[1]         
            
            cur_acc = correct_num/total_num
            train_loss.update(cur_loss)
            train_acc.update(correct,total_num) 
            tepoch.set_postfix(loss=f"{(cur_loss):>0.6f}", accuracy=f"{(100*cur_acc):>0.1f}%")            
                
            # Maximum train step per epoch is maximum_step
            if step >= maximum_step:
                break
                
    return train_loss.avg, train_acc.rate
            
def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(LABEL_NUM)]  
    global_rank = dist.get_rank()  
    word_size = dist.get_world_size()

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
            loss = loss_fn(output, target)
            pred = output.argmax(1)
            
            dist.barrier(device_ids=[torch.cuda.current_device()])
           
            #update results
            pred,target = all_gather_results(pred,target)
            dist.all_reduce(cur_loss, async_op=False)
            cur_loss = loss.item()
            cur_loss /= world_size
            
            correct_num = (pred == target).sum().item()
            for label in range(LABEL_NUM):
                tp_num = torch.logical_and(pred==target, target==label).sum().item()
                p_num = (target==label).sum().item()               
            if target.dim() == 1:
                total_num = target.size()[0]
            else:
                total_num = target.size()[0]*target.size()[1]       
                   
            cur_acc = correct_num/total_num
            val_loss.update(cur_loss)
                    val_acc.update(correct_num,total_num)  
                    val_tpr[label].update(cur_tp,cur_p) 
                  
    val_uar = np.sum([val_tpr[label].rate for label in range(LABEL_NUM)]) / LABEL_NUM   
    return val_loss.avg, val_acc.rate, val_uar

def reduce_results(tensors):
    """
    All reduce the provided tensors from all processes across machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    return tensors

def all_gather_results(tensors):
    """
    All reduce the provided tensors from all processes across machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor

def reduce_val_results(cur_loss, correct, total_num, cur_tp, cur_p, rank, world_size):
    with torch.no_grad():
        dist.reduce(cur_loss, dst=0)
        dist.reduce(correct, dst=0)
        dist.reduce(total_num, dst=0)
        if rank == 0:
            cur_loss /= world_size