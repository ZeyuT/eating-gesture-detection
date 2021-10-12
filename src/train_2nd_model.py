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

from utils import class_weights,RateMeter,AverageMeter
from constants import LABEL_TABLE,MEAN,STD,LABEL_NUM
from tqdm.auto import tqdm
import multiprocessing as mp

class probabilityDataset(data.Dataset):
    def __init__(self,
                  data_loc,
                  gt_loc,
                  mode="train",
                  label_type="all",
                  run_times=10,
                  raw_seq_len=16,
                  raw_stride=1,
                  downsample_rate=4,
                  raw_sample_len=20000):
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
        # use features from 10 runs for training
        for model_idx in range(1,11):
            feature_loc = os.path.join(data_loc,f"{model_idx}/result_RES_LSTM_30_16_8_v4_{mode}/frame_features/")
            read_samples_args.append([feature_loc,gt_loc])
        pool = mp.Pool(20)
        ret = pool.map(self.read_raw_samples,read_samples_args)

        for data in ret:
            sample_list.append(data[0])
            label_list.append(data[1])
        self.sample_list = np.concatenate(sample_list)
        self.label_list = np.concatenate(label_list)
        
        pool.close()  
        pool.join() 
    
        # Standardization         
        if self.label_type=="bite":
            self.sample_list[self.label_list!=-1] = (self.sample_list[self.label_list!=-1]-MEAN[0])/STD[0]
        elif self.label_type=="drink":
            self.sample_list[self.label_list!=-1] = (self.sample_list[self.label_list!=-1]-MEAN[1])/STD[1]
        else:
            self.sample_list[self.label_list!=-1] = (self.sample_list[self.label_list!=-1]-MEAN[0:2])/STD[0:2]
        self.finish_initialize = True
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        return self.sample_list[idx],self.label_list[idx]
    
    def read_raw_samples(self, args):
        sample_loc,gt_loc = args[0], args[1]
        video_list = ["_".join(f.split(".")[0].split("_")[1:3]) for f in os.listdir(sample_loc)]
        samples = []
        labels = []
        count = 0
        for video_idx in video_list:
            cur_samples,cur_labels = self.read_features(sample_loc, gt_loc, video_idx)
            samples.append(cur_samples[:,::self.downsample_rate,:])
            labels.append(cur_labels[:,::self.downsample_rate])
            #count += 1
            #if count > 4:
                #break
        return np.concatenate(samples), np.concatenate(labels)
  
    def read_features(self, feature_loc, gt_loc, video_idx):
        f_gt = open(os.path.join(gt_loc,video_idx,"gt_frame_3labels.txt"),"r")
        gt_content = f_gt.readlines()
        f_gt.close()   
        
        f_feature = open(os.path.join(feature_loc,f"features_{video_idx}.txt"),"r")
        if self.label_type == "all":
            # default sample value is 0, and default label is -1( will be ingored when computing loss)
            samples = np.zeros((self.raw_seq_len,self.raw_sample_len,2)) 
            labels = np.ones((self.raw_seq_len,self.raw_sample_len)) * (-1)            
            for col_idx, line in enumerate(f_feature.readlines()):
                seq_features = line.split("\n")[0].split("\t")[1:]
                for row_idx in range(0,len(seq_features),3):
                    position_idx = row_idx // 3  # the frame index in the current sequence
                    frame_idx = col_idx * self.raw_stride + position_idx  # the frame index in the current frame list
                    samples[position_idx,frame_idx,0:2] = seq_features[row_idx:row_idx+2]
            frame_gt = np.array([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content])
            
        if self.label_type in ["bite","drink"]:
            samples = np.zeros((self.raw_seq_len,self.raw_sample_len,1)) 
            labels = np.ones((self.raw_seq_len,self.raw_sample_len)) * (-1)       
            target_position = 0 if self.label_type == "bite" else 1
            for col_idx, line in enumerate(f_feature.readlines()):
                seq_features = line.split("\n")[0].split("\t")[1:]
                for row_idx in range(0,len(seq_features),3):
                    position_idx = row_idx // 3  # the frame index in the current sequence
                    frame_idx = col_idx * self.raw_stride + position_idx  # the frame index in the current frame list
                    samples[position_idx,frame_idx] = seq_features[row_idx+target_position]
            frame_gt = np.array([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content])
            frame_gt = (frame_gt==LABEL_TABLE[self.label_type]).astype("int")
        f_feature.close()
            
        for idx in range(0,self.raw_seq_len):
            labels[idx,idx:idx+len(frame_gt)-self.raw_seq_len+1] = frame_gt[idx:idx+len(frame_gt)-self.raw_seq_len+1]
            
        return samples.astype("float"),labels.astype("int")

class single_LSTM(nn.Module):
    def __init__(self,input_size=3,seq_len=10000,label_num=3):
        super(single_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(affine=False,
                                          num_features=int(seq_len))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(64, label_num),
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
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64,
                            hidden_size=32,
                            num_layers=2,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(affine=False,
                                          num_features=int(seq_len))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(32, label_num),
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
                    label_type="all",
                    downsample_rate=4, 
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
        raw_samples,raw_labels,raw_frame_gt,raw_frame_names = read_features(data_loc,gt_loc,video_idx,label_type)
        # Besides of downsampling samples, 
        # also resample in the raw sequence with test_stride*downsample_rate fps. 
        # That is equivalent with applying test_stride stride on the downsampled raw data.  
        samples = raw_samples[::test_stride*downsample_rate,::downsample_rate,:]
        labels = raw_labels[::test_stride*downsample_rate,::downsample_rate]
        frame_gt = raw_frame_gt[::test_stride*downsample_rate]
        frame_name = raw_frame_names[::test_stride*downsample_rate]
        # Standardization
        if label_type=="bite":
            samples[labels!=-1] = (samples[labels!=-1]-MEAN[0])/STD[0]
        elif label_type=="drink":
            samples[labels!=-1] = (samples[labels!=-1]-MEAN[1])/STD[1]
        else:
            samples[labels!=-1] = (samples[labels!=-1]-MEAN[0:2])/STD[0:2]
            
        input = torch.from_numpy(samples).type(torch.cuda.FloatTensor)
        input = Variable(input).cuda()
        output = model(input)
        preds = output.detach().cpu().numpy().argmax(-1)
        if label_type in ["bite","drink"]:
            heat_map = np.zeros((len(labels[0]),2))
        else:
            heat_map = np.zeros((len(labels[0]),LABEL_NUM))
        for sample_idx in range(len(preds)):
            cur_label = labels[sample_idx]
            cur_pred = preds[sample_idx][cur_label!=-1]
            for frame_idx in range(len(cur_pred)):
                heat_map[sample_idx*test_stride + frame_idx][cur_pred[frame_idx]] += 1
            f_results = open(os.path.join(test_save_loc,"frame_preds",f"sub_preds_{video_idx}_{sample_idx}.txt"),'w')
            f_results.write("\n".join(["{}\t{}".format(int(i),int(j)) for i,j in (zip(cur_label[cur_label!=-1],cur_pred))]))
            f_results.close()
            
        final_preds = np.argmax(heat_map, axis=-1)
        '''
        # upsample final predictions to original 8hz
        upsampled_preds = np.zeros(len(raw_frame_gt))
        for pred_idx in range(len(upsampled_preds)):
            upsampled_preds[pred_idx] = final_preds[round(pred_idx/downsample_rate)]
        '''
        f_results = open(os.path.join(test_save_loc,"frame_preds",f"preds_{video_idx}.txt"),'w')
        f_results.write("\n".join(["{}\t{}\t{}".format(i,int(j),int(k)) for i,j,k in (zip(frame_name,frame_gt,final_preds))]))
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
    frame_names = [line.split("\n")[0].split("\t")[0] for line in gt_content]    
    f_gt.close()   
    
    f_feature = open(os.path.join(feature_loc,f"features_{video_idx}.txt"),"r")
    # default sample value is 0, and default label is -1( will be ingored when computing loss)
    if label_type == "all":
        samples = np.zeros((raw_seq_len,raw_sample_len,2)) 
        labels = np.ones((raw_seq_len,raw_sample_len)) * (-1)            
        for col_idx, line in enumerate(f_feature.readlines()):
            seq_features = line.split("\n")[0].split("\t")[1:]
            for row_idx in range(0,len(seq_features),3):
                position_idx = row_idx // 3  # the frame index in the current sequence
                frame_idx = col_idx * raw_stride + position_idx  # the frame index in the current frame list
                samples[position_idx,frame_idx,0:2] = seq_features[row_idx:row_idx+2]
        frame_gt = np.array(([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content]))
        
    if label_type in ["bite","drink"]:
        samples = np.zeros((raw_seq_len,raw_sample_len,1)) 
        labels = np.ones((raw_seq_len,raw_sample_len)) * (-1)       
        target_position = 0 if label_type == "bite" else 1
        for col_idx, line in enumerate(f_feature.readlines()):
            seq_features = line.split("\n")[0].split("\t")[1:]
            for row_idx in range(0,len(seq_features),3):
                position_idx = row_idx // 3  # the frame index in the current sequence
                frame_idx = col_idx * raw_stride + position_idx  # the frame index in the current frame list
                samples[position_idx,frame_idx] = seq_features[row_idx+target_position]
        frame_gt = np.array(([LABEL_TABLE[line.split("\n")[0].split("\t")[1]] for line in gt_content]))
        frame_gt = (frame_gt==LABEL_TABLE[label_type]).astype("int")
    f_feature.close()
        
    for idx in range(0,raw_seq_len):
        labels[idx,idx:idx+len(frame_gt)-raw_seq_len+1] = frame_gt[idx:idx+len(frame_gt)-raw_seq_len+1]
    
    return samples.astype("float"),labels.astype("int"),frame_gt,frame_names
                    
def main():       

    train = int(sys.argv[1])
    label_type = sys.argv[2]
    network = sys.argv[3]
    batch_size = 32
    downsample_rate = 4
    raw_sample_len = 20000
    raw_seq_len = 16
    sample_len = int(raw_sample_len/downsample_rate)
    epochs = 50
    weight_type = 4
    stride = 8
    global model_label_num
    if label_type in ["bite","drink"]:
        model_label_num = 2
        input_size = 1
    else:
        model_label_num = 3
        input_size = 2
        
    log_loc = f"log_{network}_{epochs}_{raw_seq_len}_{stride}_v{weight_type}_2stage_{label_type}"
    model_loc = f"model_{network}_{epochs}_{raw_seq_len}_{stride}_v{weight_type}_2stage_{label_type}"
    test_loc = f"result_{network}_{epochs}_{raw_seq_len}_{stride}_v{weight_type}_2stage_{label_type}"    
    
    if train == 1:
        print("training models in normal mode\n")
        try:
            os.mkdir(log_loc)
        except:
            pass
        try:
            os.mkdir(model_loc)
        except:
            pass 
    else:
        print("testing models in normal mode\n")

    try:
        os.mkdir(test_loc)
    except:
        pass 
    
    if network == "single_lstm":
        model = single_LSTM(input_size=input_size,seq_len=sample_len,label_num=model_label_num).cuda()
    elif network == "double_lstm":
        model = double_LSTM(input_size=input_size,seq_len=sample_len,label_num=model_label_num).cuda()

    model = torch.nn.DataParallel(model).cuda()
    #summary(model, input_size=(batch_size,seq_len, CHANNEL, HEIGHT, WIDTH))  

    if train != 0:    
        print("loading training set")
        sys.stdout.flush()
        start_time = time.time()
        train_set = probabilityDataset(
                    data_loc = "/scratch1/zeyut/eat_detection/results_10runs/",
                    gt_loc = "/scratch1/zeyut/eat_detection/all_labels/",
                    mode = "train",
                    label_type = label_type,
                    run_times = 10,
                    raw_seq_len = raw_seq_len,
                    downsample_rate = downsample_rate,
                    raw_sample_len = raw_sample_len
                    )
        while not train_set.finish_initialize:
            time.sleep(1)
        print(f"training set loaded. elapsed time: {time.time()-start_time:>0.2f} s\n")
        
        print("loading validation set")
        sys.stdout.flush()      
        start_time = time.time()   
        val_set = probabilityDataset(
                    data_loc = "/scratch1/zeyut/eat_detection/results_10runs/",
                    gt_loc = "/scratch1/zeyut/eat_detection/all_labels/",
                    mode = "val",
                    label_type = label_type,
                    run_times = 10,
                    raw_seq_len = raw_seq_len,
                    downsample_rate = downsample_rate,
                    raw_sample_len = raw_sample_len
                    ) 
        while not val_set.finish_initialize:
            time.sleep(1)
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
                  f"model: {network}\n" \
                  f"label type: {label_type}\n"\
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
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
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
        print("no 'checkpoint_best.tar' is found")
        exit(0)
    model.load_state_dict(checkpoint['model_state_dict'])
           
    print("testing model...")
    start_time = time.time()
    sys.stdout.flush()
    for model_idx in range(1,11):
        test_2nd_stage(model = model,
                        data_loc = f"/scratch1/zeyut/eat_detection/results_10runs/{model_idx}/result_RES_LSTM_30_16_8_v4_test/frame_features/", 
                        gt_loc = "/scratch1/zeyut/eat_detection/all_labels/",
                        test_save_loc = os.path.join(test_loc,f"{model_idx}"),
                        label_type = label_type,
                        downsample_rate = downsample_rate, 
                        test_stride = 1)       
    elapsed_time = time.time() - start_time
    print(f"Test finished, elapsed time: {elapsed_time:>6f} s")

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
            
def val_loop(dataloader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    val_acc = RateMeter()    
    val_tpr = [RateMeter() for _ in range(model_label_num)]  
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
            for label in range(model_label_num):
                cur_tp = torch.logical_and(effective_pred==effective_target, effective_target==label).sum().item()
                cur_p = (effective_target==label).sum().item()
                val_tpr[label].update(cur_tp,cur_p) 
            val_acc.update(correct,effective_target.size()[0])  
    val_uar = np.sum([val_tpr[label].rate for label in range(model_label_num)]) / model_label_num   
    return val_loss.avg, val_acc.rate, val_uar

    
if __name__ == "__main__":  
    main()