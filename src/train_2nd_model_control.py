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

class probabilityDataset(data.Dataset):
    def __init__(self,
                 data_loc,
                 gt_loc,
                 mode="train",
                 label_type="all",
                 raw_seq_len=16,
                 raw_stride=1,
                 downsample_rate=4,
                 raw_sample_len=20000,
                 run_idx=None,
                 add_noise=False):
        'Initialization'
        self.finish_initialize = False
        sample_list = []
        label_list = []
        read_samples_args = []
        self.label_type = label_type
        self.raw_seq_len = raw_seq_len
        self.raw_sample_len = raw_sample_len
        self.downsample_rate = downsample_rate
        self.raw_stride = raw_stride
        self.run_idx = run_idx
        self.add_noise = add_noise
        if label_type in ["bite","drink"]:
            self.feature_num = 2
        elif label_type == "all":
            self.feature_num = 3
        self.mean_list = []
        self.std_list = []
        
        # use selected local run for training
        feature_loc = os.path.join(data_loc,f"{self.run_idx}/result_{mode}/frame_features/")
        self.sample_list, self.label_list, self.mean_list, self.std_list = self.read_raw_samples(feature_loc,gt_loc)
        
        np.savetxt(os.path.join(data_loc,f"train_mean_{label_type}.txt"),np.array(self.mean_list))
        np.savetxt(os.path.join(data_loc,f"train_std_{label_type}.txt"),np.array(self.std_list))

        self.finish_initialize = True
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        cur_sample = self.sample_list[idx]
        cur_label = self.label_list[idx]
        if self.add_noise:
            cur_sample[cur_label!=-1] += np.random.normal(0, 0.1, cur_sample[cur_label!=-1].shape) 
        return cur_sample, cur_label
    
    def read_raw_samples(self, sample_loc, gt_loc):
        video_list = ["_".join(f.split(".")[0].split("_")[1:3]) for f in os.listdir(sample_loc)]
        #video_list = ["p072_c1"]
        samples = []
        labels = []
        count = 0
        for video_idx in video_list:
            cur_samples,cur_labels = self.read_features(sample_loc, gt_loc, video_idx)
            samples.append(cur_samples)
            labels.append(cur_labels)        
        samples = np.concatenate(samples) 
        labels = np.concatenate(labels)
        mean = []
        std = []
        
        # standarization per model, and per feature
        for feature_idx in range(self.feature_num):
            cur_mean = samples[labels!=-1,feature_idx].mean(axis=0)
            cur_std = samples[labels!=-1,feature_idx].std(axis=0)
            samples[labels!=-1,feature_idx] = (samples[labels!=-1,feature_idx]-cur_mean)/cur_std
            mean.append(cur_mean)
            std.append(cur_std)
          
        # downsample
        samples = samples[:,::self.downsample_rate,:]
        labels = labels[:,::self.downsample_rate]
        mean = np.array(mean)
        std = np.array(std)

        return samples, labels, mean, std
  
    def read_features(self, feature_loc, gt_loc, video_idx):
        f_gt = open(os.path.join(gt_loc,video_idx,"gt_frame_3labels.txt"),"r")
        gt_content = f_gt.readlines()
        f_gt.close()         
        
        f_feature = open(os.path.join(feature_loc,f"features_{video_idx}.txt"),"r")
        if self.label_type == "all":
            # default sample value is 0, and default label is -1( will be ingored when computing loss)
            samples = np.zeros((1,self.raw_sample_len,3)) 
            labels = np.ones((1,self.raw_sample_len)) * (-1)            
            for row_idx, line in enumerate(f_feature.readlines()):
                seq_features = line.split("\n")[0].split("\t")[1:]
                position_idx = len(seq_features)//3//2  # use index of the frame centered at current feature sequence
                frame_idx = row_idx * self.raw_stride + position_idx  # the frame index in the video's time span
                samples[0,frame_idx,0:3] = seq_features[position_idx*3:position_idx*3+3]
            frame_gt = np.array([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content])

            f_feature.close()
            labels[0,self.raw_seq_len//2:len(frame_gt)-self.raw_seq_len//2+1] = frame_gt[self.raw_seq_len//2:len(frame_gt)-self.raw_seq_len//2+1]
        else:
            sys.exit("Only support 'all' label types")
        return samples.astype("float"),labels.astype("int")

class single_LSTM(nn.Module):
    def __init__(self,input_size=3,seq_len=10000,label_num=3):
        super(single_LSTM, self).__init__()
        # Was using 64 units * 2 layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=64,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(affine=False,
                                          num_features=int(seq_len))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(64*2, label_num),
                                    nn.ReLU())         
        self.act = nn.Softmax(dim=-1)
            
        '''initialization'''                          
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)   

    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        output = self.act(x)
        return output

class double_LSTM(nn.Module):
    def __init__(self,input_size=3,seq_len=10000,label_num=3):
        super(double_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size,
                            hidden_size=256,
                            num_layers=2,
                            bidirectional=False,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=2,
                            bidirectional=False,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(affine=False,
                                          num_features=int(seq_len))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(128, 32),
                                nn.Linear(32, label_num),
                                 nn.ReLU())         
        self.act = nn.Softmax(dim=-1)
            
        '''initialization'''                          
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)   

    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        output = self.act(x)
        return output
        
def test_2nd_stage(model, 
                    data_loc, 
                    gt_loc, 
                    test_save_loc,
                    mean,
                    std,
                    label_type="all",
                    downsample_rate=4, 
                    raw_sample_len=20000,
                    raw_seq_len=16,
                    test_stride=1):      
    try:
        os.mkdir(os.path.join(test_save_loc))
    except:
        pass    
    try:
        os.mkdir(os.path.join(test_save_loc,"frame_probs"))
    except:
        pass
    try:
        os.mkdir(os.path.join(test_save_loc,"frame_preds"))
    except:
        pass     
    
    test_video_list = ["_".join(f.split(".")[0].split("_")[1:3]) for f in os.listdir(data_loc)]
    for video_idx in test_video_list:
        raw_samples,raw_labels,raw_frame_gt,raw_frame_names = read_features(feature_loc=data_loc, 
                                                                              gt_loc=gt_loc, 
                                                                              video_idx=video_idx, 
                                                                              label_type=label_type,
                                                                              raw_stride=1, 
                                                                              raw_seq_len=raw_seq_len, 
                                                                              raw_sample_len=raw_sample_len)
        # Besides of downsampling samples, 
        # also resample in the raw sequence with test_stride*downsample_rate fps. 
        # That is equivalent with applying test_stride stride on the downsampled raw data.  
        samples = raw_samples[::test_stride*downsample_rate,::downsample_rate,:]
        labels = raw_labels[::test_stride*downsample_rate,::downsample_rate]
        frame_gt = raw_frame_gt[::test_stride*downsample_rate]
        frame_names = raw_frame_names[::test_stride*downsample_rate]
        # Standardization
        samples[labels!=-1] = (samples[labels!=-1]-mean)/std
            
        input = torch.from_numpy(samples).type(torch.cuda.FloatTensor)
        input = Variable(input).cuda()
        output = model(input)
        probs = output.detach().cpu().numpy()
        preds = probs.argmax(-1)
        
        if label_type in ["bite","drink"]:
            heat_map = np.zeros((len(samples[0]),2))
        else:
            heat_map = np.zeros((len(samples[0]),LABEL_NUM))
        for sample_idx in range(len(preds)):
            cur_label = labels[sample_idx]
            # Here the length of a test video and the the starting idx of current frame sequence are known, 
            # so it is reasonable to trim predictions accordingly before evaluating results
            # which is indicated by array [cur_label!=-1]
            cur_prob = probs[sample_idx][cur_label!=-1]
            cur_pred = preds[sample_idx][cur_label!=-1]
            cur_names = frame_names[cur_label[:len(frame_names)]!=-1]
            for frame_idx in range(len(cur_pred)):
                heat_map[sample_idx*test_stride + frame_idx][cur_pred[frame_idx]] += 1
                
            f_probs = open(os.path.join(test_save_loc,"frame_probs",f"probs_{video_idx}_{sample_idx}.txt"),'w')
            for name, prob_group in zip(cur_names,cur_prob):
                # frame's indexes start from 1.
                frame_idx = int(name[6:-4])
                f_probs.write("{}".format(frame_idx))
                for prob_value in prob_group:
                    f_probs.write("\t{0:.6f}".format(prob_value))
                f_probs.write("\n")
            f_probs.close()   
            f_preds = open(os.path.join(test_save_loc,"frame_preds",f"sub_preds_{video_idx}_{sample_idx}.txt"),'w')
            f_preds.write("\n".join(["{}\t{}\t{}".format(i,int(j),int(k)) for i,j,k in \
                                        (zip(frame_names[cur_label[:len(frame_names)]!=-1],cur_label[cur_label!=-1],cur_pred))]))
            f_preds.close()
            
        final_preds = np.argmax(heat_map, axis=-1)
        '''
        # upsample final predictions to original 8hz
        upsampled_preds = np.zeros(len(raw_frame_gt))
        for pred_idx in range(len(upsampled_preds)):
            upsampled_preds[pred_idx] = final_preds[round(pred_idx/downsample_rate)]
        '''
        f_results = open(os.path.join(test_save_loc,"frame_preds",f"preds_{video_idx}.txt"),'w')
        f_results.write("\n".join(["{}\t{}\t{}".format(i,int(j),int(k)) for i,j,k in (zip(frame_names,frame_gt,final_preds))]))
        f_results.close() 
        
def read_features(feature_loc, 
              gt_loc, 
              video_idx, 
              label_type="all",
              raw_stride=1, 
              raw_seq_len=16, 
              raw_sample_len=20000):
    f_gt = open(os.path.join(gt_loc,video_idx,"gt_frame_3labels.txt"),"r")
    gt_content = f_gt.readlines()
    frame_names = np.array([line.split("\n")[0].split("\t")[0] for line in gt_content])
    f_gt.close()   
    
    f_feature = open(os.path.join(feature_loc,f"features_{video_idx}.txt"),"r")
    # default sample value is 0, and default label is -1( will be ingored when computing loss)
    if label_type == "all":
        # Samples dimension: [position_in_window, position_in_new_sample, channel]
        samples = np.zeros((raw_seq_len,raw_sample_len,3)) 
        labels = np.ones((raw_seq_len,raw_sample_len)) * (-1)            
        for row_idx, line in enumerate(f_feature.readlines()):
            seq_features = line.split("\n")[0].split("\t")[1:]
            for col_idx in range(0,len(seq_features),3):
                position_idx = col_idx // 3  # the frame index in the current sequence
                frame_idx = row_idx * raw_stride + position_idx  # the frame index in the current frame list
                samples[position_idx,frame_idx,0:3] = seq_features[col_idx:col_idx+3]
        frame_gt = np.array(([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content]))
        
    if label_type in ["bite","drink"]:
        samples = np.zeros((raw_seq_len,raw_sample_len,2)) 
        labels = np.ones((raw_seq_len,raw_sample_len)) * (-1)       
        target_position = 0 if label_type == "bite" else 1
        for row_idx, line in enumerate(f_feature.readlines()):
            seq_features = line.split("\n")[0].split("\t")[1:]
            for col_idx in range(0,len(seq_features),3):
                position_idx = col_idx // 3  # the frame index in the current sequence
                frame_idx = row_idx * raw_stride + position_idx  # the frame index in the current frame list
                samples[position_idx,frame_idx,0] = seq_features[col_idx+target_position]
                samples[position_idx,frame_idx,1] = seq_features[col_idx+2]
                
        frame_gt = np.array(([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content]))
        frame_gt = (frame_gt==LABEL_TABLE[label_type]).astype("int")
    f_feature.close()
        
    for idx in range(0,raw_seq_len):
        labels[idx,idx:idx+len(frame_gt)-raw_seq_len+1] = frame_gt[idx:idx+len(frame_gt)-raw_seq_len+1]
    
    return samples.astype("float"),labels.astype("int"),frame_gt,frame_names

def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(LABEL_NUM)]  
    with torch.no_grad():
        for (input, target) in dataloader:
            input = input.type(torch.cuda.FloatTensor)
            input = Variable(input).cuda()
            target = Variable(target).cuda()            
            # Compute prediction and loss
            output = model(input).permute(0, 2, 1)
            loss = loss_fn(output, target)
            pred = output.argmax(1)
            val_loss.update(loss.item())
            correct = (pred == target).sum().item()
            #update results
            effective_pred = pred[target!=-1]
            effective_target = target[target!=-1]
            for label in range(LABEL_NUM):
                cur_tp = torch.logical_and(effective_pred==effective_target, effective_target==label).sum().item()
                cur_p = (effective_target==label).sum().item()
                val_tpr[label].update(cur_tp,cur_p) 
            val_acc.update(correct,effective_target.size()[0])  
    val_uar = np.sum([val_tpr[label].rate for label in range(LABEL_NUM)]) / LABEL_NUM   
    return val_loss.avg, val_acc.rate, val_uar

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_acc = RateMeter() 
    with tqdm(dataloader,unit= "batch") as tepoch:
        for input, target in tepoch:
            input = input.type(torch.cuda.FloatTensor)
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
            effective_pred = pred[target!=-1]
            effective_target = target[target!=-1]
            correct = (effective_pred == effective_target).sum().item()     
            cur_acc = correct/(effective_target.size()[0])
            train_loss.update(cur_loss)
            train_acc.update(correct,effective_target.size()[0]) 
            tepoch.set_postfix(loss=f"{(cur_loss):>0.6f}", accuracy=f"{(100*cur_acc):>0.1f}%")
    return train_loss.avg, train_acc.rate
            

def run(train,
        label_type,
        first_network,
        network,
        root_result_loc,
        root_data_loc,
        batch_size = 32,
        epochs = 50,
        weight_type = 4,
        run_idx=None,
        add_noise=False):
    input_size = 3
    learning_rate = 0.001
    decay_rate = 0.9
    if first_network == 'RES_BILSTM':
        data_loc = os.path.join(root_data_loc, "results_10runs_RES_BILSTM")
        fps = 8
        downsample_rate = fps // 2 # i.e. downsample to 2 hz
        raw_sample_len = 20000 # 20000/8 = 2500 s
        raw_seq_len = 16
        sample_len = int(raw_sample_len/downsample_rate)

    elif first_network == 'RES_LSTM':
        data_loc = os.path.join(root_data_loc, "results_10runs_2unidirectional_ltsm")
        fps = 8
        downsample_rate = fps // 2 # i.e. downsample to 2 hz
        raw_sample_len = 20000 # 20000/8 = 2500 s
        raw_seq_len = 16
        sample_len = int(raw_sample_len/downsample_rate)
    
    elif first_network == 'x3d-s':
        data_loc = os.path.join(root_data_loc, "results_10runs_x3d-s")    
        fps = 5
        downsample_rate = fps // 2 # i.e. downsample to 2 hz (2.5hz actually)
        raw_sample_len = 12500 # 2500 (s) * 5
        raw_seq_len = 13
        sample_len = int(raw_sample_len/downsample_rate)
        
    log_loc = os.path.join(root_result_loc, f"log_{network}_{first_network}_{raw_seq_len}_2stage_{label_type}")
    model_loc = os.path.join(root_result_loc, f"model_{network}_{first_network}_{raw_seq_len}_2stage_{label_type}")
    test_loc = os.path.join(root_result_loc, f"result_{network}_{first_network}_{raw_seq_len}_2stage_{label_type}")
    
    if train == 1:
        print("training models in normal mode\n")
        try:
            os.makedirs(log_loc)
        except:
            pass
        try:
            os.makedirs(model_loc)
        except:
            pass 
    else:
        print("testing models in normal mode\n")

    try:
        os.makedirs(test_loc)
    except:
        pass 
    
    if network == "double_lstm":
        model = double_LSTM(input_size=input_size,seq_len=sample_len,label_num=LABEL_NUM).cuda()
    else:
        model = single_LSTM(input_size=input_size,seq_len=sample_len,label_num=LABEL_NUM).cuda()
    model = torch.nn.DataParallel(model).cuda()
    #summary(model, input_size=(batch_size,seq_len, CHANNEL, HEIGHT, WIDTH))  

    if train != 0:    
        print("loading training set")
        sys.stdout.flush()
        start_time = time.time()
        train_set = probabilityDataset(
                    data_loc = data_loc,
                    gt_loc = os.path.join(root_data_loc, f"VideoData_independent_{fps}hz", "train_set/"),
                    mode = "train",
                    label_type = label_type,
                    raw_seq_len = raw_seq_len,
                    downsample_rate = downsample_rate,
                    raw_sample_len = raw_sample_len,
                    run_idx=run_idx,
                    add_noise=add_noise
                    )
        
        print(f"training set loaded. elapsed time: {time.time()-start_time:>0.2f} s\n")
        
        print("loading validation set")
        sys.stdout.flush()      
        start_time = time.time()   
        val_set = probabilityDataset(
                    data_loc = data_loc,
                    gt_loc = os.path.join(root_data_loc, f"VideoData_independent_{fps}hz", "val_set/"),
                    mode = "val",
                    label_type = label_type,
                    raw_seq_len = raw_seq_len,
                    downsample_rate = downsample_rate,
                    raw_sample_len = raw_sample_len,
                    run_idx=run_idx,
                    add_noise=False
                    ) 

        print(f"validation set loaded. elapsed time: {time.time()-start_time:>0.2f} s\n")
                                                  
        train_loader = data.DataLoader(
                        dataset = train_set,
                        batch_size = batch_size,
                        shuffle = True
            )  
            
        val_loader = data.DataLoader(
                        dataset = val_set,
                        batch_size = batch_size,
                        shuffle = False
            )                

        log = open(os.path.join(log_loc, "train_log.txt"), 'w')    
        print ("=======================Experimental Settings=======================")
        log.write("=======================Experimental Settings=======================\n")
        log.flush()
        sys.stdout.flush() 
        available_gpu = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]

        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"train second stage model\n"\
                  f"started at {start_time}\n"\
                  f"available GPU: {available_gpu}\n"\
                  f"first stage model: {first_network}\n" \
                  f"model: {network}\n" \
                  f"label type: {label_type}\n" \
                  f"batch size: {batch_size}  epochs: {epochs}\n"\
                  f"sample length: {sample_len}  downsample rate: {downsample_rate}\n"\
                  f"weight_type: {weight_type}"
        print(message)
        log.write(message+"\n")
        sys.stdout.flush()
        log.flush()
        
        weights,class_counts = class_weights(train_set.label_list[train_set.label_list!=-1],weight_type) 
        _,val_class_counts = class_weights(val_set.label_list[val_set.label_list!=-1],weight_type) 

        message = f"training set: {len(train_set)} video/patterns\n" \
                  f"training set class counts: {class_counts}\n" \
                  f"training set class weights: {weights}\n"\
                  f"validation set: {len(val_set)} video/patterns\n" \
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
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().cuda(),
                                      ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        best_val_uar = 0
        for epoch in range(epochs):
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
        print("no 'checkpoint_best.tar' is found in")
        print(model_loc)
        exit(0)
    model.load_state_dict(checkpoint['model_state_dict'])
           
    print("testing model...")
    start_time = time.time()
    sys.stdout.flush()
   
    mean_list = np.loadtxt(os.path.join(data_loc,f"train_mean_{label_type}.txt"))
    std_list = np.loadtxt(os.path.join(data_loc,f"train_std_{label_type}.txt"))
    
    test_2nd_stage(model = model,
                    data_loc = os.path.join(data_loc,f"{run_idx}", "result_test", "frame_features"), 
                    gt_loc = os.path.join(root_data_loc, f"VideoData_independent_{fps}hz", "test_set"),
                    test_save_loc = test_loc,
                    mean = mean_list[0],
                    std = std_list[0],
                    label_type = label_type,
                    downsample_rate = downsample_rate, 
                    raw_sample_len = raw_sample_len,
                    raw_seq_len = raw_seq_len,
                    test_stride = 1)       
    elapsed_time = time.time() - start_time
    print(f"Test finished, elapsed time: {elapsed_time:>6f} s")

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
            root_result_loc = os.path.join(RESULT_ROOT_LOC, "control", f"{args['first_network']}", f"{run_idx+1}"),
            batch_size = 32,
            epochs = 50,
            weight_type = 4,
            run_idx = run_idx+1,
            add_noise=False)