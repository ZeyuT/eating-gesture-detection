import sys
import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from constants import LABEL_NUM, LABEL_TABLE
from tqdm import tqdm
import math

class FrameSequenceDataset(data.Dataset):
    def __init__(self,root_path,video_list,seq_len,stride,model_type,transform,test_mode=False):
        'Initialization'
        self.transform = transform
        self.test_mode = test_mode
        self.sample_list = []
        self.label_list = []
        self.model_type = model_type
        self._get_data_list(root_path, video_list, seq_len, stride)
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        frame_list, labels = self.sample_list[idx], self.label_list[idx]
        frames = self._get_frames(frame_list)
        frames = torch.stack([transforms.functional.to_tensor(frame) for frame in frames])
        if not self.test_mode and self.transform is not None:
            frames = self.transform(frames)
        return frames, labels

    def _get_frames(self, frame_list):
        frames = []
        for frame_loc in frame_list:
            frames.append(Image.open(frame_loc).convert('RGB'))
        return frames

    def _get_data_list(self, root_path, video_list, seq_len, stride):
        for video in video_list:
            frame_locs = []
            frame_labels = []
            f = open(os.path.join(root_path,video,"gt_frame_3labels.txt"),"r")
            gt_frame = [str.split(line, "\t") for line in f.readlines()]
            for frame_info in gt_frame:
                frame_locs.append(os.path.join(root_path,video,frame_info[0]))
                cur_label_idx = LABEL_TABLE[str.split(frame_info[1], "\n")[0]]
                frame_labels.append(cur_label_idx)
            for i in range(0, len(frame_locs)-seq_len+1, stride):
                self.sample_list.append(frame_locs[i:i+seq_len])
                if self.model_type == 'seq2seq':
                    self.label_list.append(np.array(frame_labels[i:i+seq_len]))
                elif self.model_type == 'seq2one':
                    self.label_list.append(np.array(frame_labels[i+seq_len-1]))
        self.label_list = np.array(self.label_list)
        self.sample_list = np.array(self.sample_list)

class testDataset(data.Dataset):
    def __init__(self,data_path,seq_len,stride,transform):
        'Initialization'
        self.data_path = data_path
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.input_list = []
        self.video_labels = []
        self.frame_names = []
        self._get_data_list(data_path,seq_len,stride)   
        
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        frame_list = self.input_list[idx]
        frames = self._get_frames(frame_list)
        frames = torch.stack([transforms.functional.to_tensor(frame) for frame in frames])
        frames = self.transform(frames)
        return frames
    
    def _get_frames(self, frame_list):
        frames = []
        for frame_loc in frame_list:
            frames.append(Image.open(frame_loc).convert('RGB'))
        return frames
    
    def _get_data_list(self,data_path,seq_len,stride):
        frame_locs = []
        f = open(os.path.join(data_path,"gt_frame_3labels.txt"),"r")
        gt_frame = [str.split(line, "\t") for line in f.readlines()]
        sys.stdout.flush()
        frame_name_list = []
        for frame_info in gt_frame:
            frame_name_list.append(frame_info[0])
            frame_locs.append(os.path.join(data_path,frame_info[0]))
            cur_label_idx = LABEL_TABLE[str.split(frame_info[1], "\n")[0]]
            self.video_labels.append(cur_label_idx)
        for i in range(0, len(frame_locs)-seq_len+1, stride):
            self.input_list.append(frame_locs[i:i+seq_len])
            #self.frame_names.append(frame_name_list[i])
            
        self.video_labels = np.array(self.video_labels)       
        self.frame_names = np.array(frame_name_list) 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

class RateMeter(object):
    """Computes and stores the average rate (acc, TPR, etc)"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.correctCount = 0
        self.totalCount = 0
        self.rate = 0
    def update(self, correct, total):
        self.correctCount += correct
        self.totalCount += total
        if self.totalCount:
            self.rate = self.correctCount / self.totalCount
        else:
            self.rate = 0
            
def test_model(model, 
              test_video_list,
              root_path,
              test_save_loc, 
              seq_len, 
              model_type,
              inference_type,
              test_batch_size=1000,
              test_stride=1):
    model.eval()
    preprocess = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    if root_path.split("/")[-1] == "train_set":
        f_matrices = open(os.path.join(test_save_loc,"matrices_train.txt"),'w')
    elif root_path.split("/")[-1] == "val_set":
        f_matrices = open(os.path.join(test_save_loc,"matrices_val.txt"),'w')
    else:
        f_matrices = open(os.path.join(test_save_loc,"matrices_test.txt"),'w')
        
    print ("=======================Test Settings===============================")
    f_matrices.write("=======================Test Settings===============================\n")  
    message = f"{len(test_video_list)} videos in testing set\n"\
              f"testing batch size: {test_batch_size}\n"\
              f"test stride: {test_stride}"
    print(message)
    f_matrices.write(message+'\n')
    print ('===================================================================\n')
    f_matrices.write('===================================================================\n')
    f_matrices.flush()
    sys.stdout.flush()
    
    acc = RateMeter()
    tpr = [RateMeter() for _ in range(LABEL_NUM)]
    nonintake = RateMeter()
    for video_name in test_video_list:
        test_set = testDataset(
                    data_path=os.path.join(root_path,video_name),
                    seq_len=seq_len,
                    stride=test_stride,
                    transform=preprocess
                    )
        test_loader = data.DataLoader(
                            dataset=test_set,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=10,
                            pin_memory=True
                            )   
        video_labels = test_set.video_labels
        frame_names = test_set.frame_names
        pred_list = []
        prob_list = [] 
             
        with tqdm(test_loader,unit= "batch") as tbatch:
            tbatch.set_description(f"video {video_name}")
            with torch.no_grad():
                for input in tbatch:
                    input = input.type(torch.cuda.FloatTensor)
                    input = Variable(input).cuda()
                    output = model(input)
                    if inference_type == 'seq2one':
                        # The last of each frame sequence is the prediction target.
                        # the output dimension is: [seq, frame, class]
                        if model_type == 'seq2seq':
                            cur_prob = output.detach().cpu().numpy()[:,-1,:]
                        else:
                            cur_prob = output.detach().cpu().numpy()
                        pred_list.append(cur_prob.argmax(-1))
                    elif inference_type == 'seq2seq':
                        cur_prob = output.detach().cpu().numpy()
                        pred_list.append(cur_prob.argmax(-1))
                    prob_list.append(cur_prob)
        pred_list = np.concatenate(pred_list, axis=0)
        prob_list = np.concatenate(prob_list, axis=0)
        print('\n')
        if inference_type == 'seq2seq':   
            # build a heat map
            heat_map = np.zeros((len(video_labels),LABEL_NUM))
            for seq_idx in range(len(pred_list)):
                for frame_idx in range(len(pred_list[seq_idx])):
                    heat_map[test_stride*seq_idx+frame_idx][pred_list[seq_idx][frame_idx]] += 1
            video_preds = np.argmax(heat_map, axis=-1)
        elif inference_type == 'seq2one':
            # default pred value is 2, i.e. non-intake
            video_preds = np.ones(len(video_labels))*2.0
            video_preds[seq_len-1:seq_len-1+len(pred_list)] = pred_list           
        
        '''
        Save the raw prediction for visualization
        ''' 
        f_probs = open(os.path.join(test_save_loc,"frame_probs",f"probs_{video_name}.txt"),'w')
        for name, window_prob in zip(frame_names,prob_list):
            # frame's indexes start from 0. This is to align the convention of visualization
            frame_idx = int(name[6:-4])-1
            f_probs.write("{}".format(frame_idx))
            for prob_value in window_prob.flatten():
                f_probs.write("\t{0:.6f}".format(prob_value))
            f_probs.write("\n")
        f_probs.close()        
        '''
        Save the predictions
        '''
        f_results = open(os.path.join(test_save_loc,"frame_preds",f"preds_{video_name}.txt"),'w')
        f_results.write("\n".join(["{}\t{}\t{}".format(i,j,int(k)) for i,j,k in (zip(frame_names,video_labels,video_preds))]))
        f_results.close()  
        '''
        evaluate results
        ''' 
        cur_correct = np.sum(video_preds == video_labels)
        cur_acc = cur_correct/len(video_labels)
        acc.update(cur_correct,len(video_labels))
        cur_nonintake = np.sum(video_labels == 2)
        nonintake.update(cur_nonintake, len(video_labels))
        cur_uar = 0.0
        for label in range(LABEL_NUM):
            cur_tp = np.logical_and(video_preds == video_labels, video_labels==label).sum()
            cur_p = (video_labels == label).sum().item()
            if cur_p!=0:
                cur_uar += cur_tp/cur_p
            tpr[label].update(cur_tp,cur_p) 
        cur_uar = cur_uar/3
        message = f"video {video_name}   " \
                  f"acc: {(100*cur_acc):>0.2f}%   uar: {(100*cur_uar):>0.2f}%   " \
                  f"non_intake%: {100*cur_nonintake/len(video_labels):>0.2f}%"
        f_matrices.write(message+"\n")
        print(message) 
        sys.stdout.flush()  
        f_matrices.flush()  
    '''
    Write out global evaluation matrix`
    '''
    uar = np.sum([tpr[label].rate for label in range(LABEL_NUM)]) / LABEL_NUM
    message = f"summary\n" \
              f"acc: {(100*acc.rate):>0.2f}%   uar: {(100*uar):>0.2f}%\n" \
              f"{len(test_video_list)} videos in testing set\n" \
              f"{acc.totalCount} samples in testing set\n" \
              f"non_intake percentage: {100*nonintake.rate:>0.2f}%"
    print(message)
    f_matrices.write(message+"\n")
    sys.stdout.flush()  
    f_matrices.flush()
    f_matrices.close()

    
