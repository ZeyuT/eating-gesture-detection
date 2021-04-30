import sys
import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM, LABEL_TABLE

def class_weights(label_list, weight_type):
    """
    version 1: uniform weights
    version 2: 1/(1+LABEL_NUM*c(i)/n)
    version 3: a/c(i) (Inverse Number of Sample)
    version 4: a/c(i)**0.5 (Inverse of Square Root of Number of Samples)
    version 5: (1-beta) / (1-beta**c(i)) (Effective Number of Samples)
    https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
    All weights are normalized.
    Where n is the total pattern numeber. a=10000 is a constant to avoid super small number.
    """
    class_counts = []
    for label in range(LABEL_NUM):
        class_counts.append(np.sum(label_list==label))
    class_counts = np.array(class_counts)
    total = np.sum(class_counts)
    weights = []
    a = 10000
    for i in range(LABEL_NUM):
        if class_counts[i] == 0:
            weights.append(0)
        else:
            if weight_type == 1:    
                weights.append(1/LABEL_NUM)
            elif weight_type == 2:
                weights.append(1/(1+LABEL_NUM*class_counts[i]/total))
            elif weight_type == 3:
                weights.append(a/class_counts[i])
            elif weight_type == 4:
                weights.append(a/class_counts[i]**0.5)
            elif weight_type == 5:
                beta = 0.99999
                weights.append(a*(1-beta)/(1-beta**class_counts[i]))
    weights = np.array(weights) 
    weights = weights/np.sum(weights)
    return weights, class_counts
            
class FrameSequenceDataset(data.Dataset):
    def __init__(self,root_path,video_list,seq_len,stride,model_type,transform,test_mode=False):
        'Initialization'
        self.root_path = root_path
        self.model_type = model_type
        self.transform = transform
        self.test_mode = test_mode
        self.sample_list = []
        self.label_list = []
        self.seq_len = seq_len
        self._get_data_list(video_list, seq_len, stride, model_type)
        
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

    def _get_data_list(self, video_list, seq_len, stride, model_type):
        for video in video_list:
            frame_locs = []
            frame_labels = []
            f = open(os.path.join(self.root_path,video,"gt_frame_3labels.txt"),"r")
            gt_frame = [str.split(line, "\t") for line in f.readlines()]
            for frame_info in gt_frame:
                frame_locs.append(os.path.join(self.root_path,video,frame_info[0]))
                cur_label_idx = LABEL_TABLE[str.split(frame_info[1], "\n")[0]]
                frame_labels.append(cur_label_idx)
            for i in range(0, len(frame_locs)-seq_len, stride):
                self.sample_list.append(frame_locs[i:i+seq_len])
                if model_type == 1:
                    self.label_list.append(np.array(frame_labels[i:i+seq_len]))
                elif model_type == 2:
                    self.label_list.append(np.array(frame_labels[i+seq_len-1]))
        self.label_list = np.array(self.label_list)
        self.sample_list = np.array(self.sample_list)
        
def denormalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    """
    inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()

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
        self.avg = self.sum / self.count
            
def test_model(model, videos_test, test_loc, seq_len, model_type,test_stride=1):

    preprocess = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
    test_set = FrameSequenceDataset(
            root_path=FRAME_LOC+"test_set/",
            video_list=test_video_list,
            seq_len=seq_len,
            stride=stride,
            model_type=model_type,
            transform=None,
            test_mode=False
            )
        
    # test_stride is used for model_type 1
    test_batch = 1024
    f_matrices = open(test_loc + "/matrices.txt",'w')
    print("{} videos in testing set".format(len(videos_test)))
    print("testing batch size: {}\n".format(test_batch))
    total_True = 0
    total_pred = 0
    total_nonintake = 0
    sample_num = 0
    for video in videos_test:
        frames = []
        frame_labels = []
        frame_name = []
        f = open(FRAME_LOC + video + "/gt_frame_3labels.txt","r")
        #f = open("/home/zeyut/eat_detection/workspace/eating-gesture-detection/VideoData/p176_c1/gt_frame_3labels.txt","r")
        gt_frame = [str.split(line, "\t") for line in f.readlines()]
        for frame_info in gt_frame:
            frame_name.append(frame_info[0])
            cur_img = Image.open(FRAME_LOC + video + "/" + frame_info[0])
            cur_img = cur_img / 255.0 
            cur_img = transforms.functional.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(cur_img)
            frames.append(cur_img)
            #cur_labelIdx = int(frame_info[1])
            cur_labelIdx = numeralize_labels(str.split(frame_info[1], "\n")[0])
            frame_labels.append(cur_labelIdx)
        frame_labels = np.array(frame_labels)
        pred = []
        probs = []
        x_test = []
        if model_type == 1:
            for i in range(0, len(frames)-seq_len, test_stride):
                x_test.append(frames[i:i+seq_len])
                'Make predictions with batch size being test_batch'
                if len(x_test) >= test_batch:
                    x_test = np.reshape(x_test,(-1,seq_len,HEIGHT,WIDTH,CHANNEL))
                    cur_pred = np.squeeze(model.predict(x_test))
                    pred.append(np.argmax(cur_pred, axis=-1).astype("int"))
                    probs.append(cur_pred)
                    sample_num += len(x_test)
                    x_test = []
            if len(x_test) > 0:
                    x_test = np.reshape(x_test,(-1,seq_len,HEIGHT,WIDTH,CHANNEL))
                    cur_pred = np.squeeze(model.predict(x_test))
                    pred.append(np.argmax(cur_pred, axis=-1).astype("int"))
                    probs.append(cur_pred)
                    sample_num += len(x_test)
                    
            pred = np.concatenate(pred, axis=0)
            probs = np.concatenate(probs, axis=0)
            print("len:{}".format(len(probs)))
            'build prediction heat map'
            pred_heat = np.zeros((len(frame_labels),LABEL_NUM))
            for i in range(len(pred)):
                for j in range(len(pred[i])):
                    pred_heat[test_stride*i+j][pred[i][j]] += 1
            'get frame wise predictions with max vote strategy'
            frame_pred = np.argmax(pred_heat, axis=-1)
         
        elif model_type == 2 or model_type == 3:
            pred = []
            for i in range(0, len(frames)-seq_len):
                x_test.append(frames[i:i+seq_len])
                'Make predictions with batch size being test_batch'
                if len(x_test) >= test_batch:
                    x_test = np.reshape(x_test,(-1,seq_len,HEIGHT,WIDTH,CHANNEL))
                    cur_pred = np.squeeze(model.predict(x_test))
                    pred.append(np.argmax(cur_pred, axis=-1).astype("int"))
                    sample_num += len(x_test)
                    x_test = []
            if len(x_test) > 0:
                    x_test = np.reshape(x_test,(-1,seq_len,HEIGHT,WIDTH,CHANNEL))
                    cur_pred = np.squeeze(model.predict(x_test))
                    pred.append(np.argmax(cur_pred, axis=-1).astype("int"))
                    sample_num += len(x_test)
            pred = np.concatenate(pred, axis=0)
            frame_pred = np.zeros(len(frames))
            frame_pred[seq_len-1:seq_len-1+len(pred)] = pred
        
        """ for visualization"""
        f_probs = open(test_loc + "/probs_{}.txt".format(video),'w')
        for name, win_prob in zip(frame_name,probs):
            #print(name)
            frame_idx = int(name[6:-4])-1
            f_probs.write("{}".format(frame_idx))
            for num in win_prob.flatten():
                f_probs.write("\t{0:.6f}".format(num))
            f_probs.write("\n")
        f_probs.close()
        f_results = open(test_loc + "/pred_{}.txt".format(video),'w')
        f_results.write("\n".join(["{}\t{}\t{}".format(i,j,int(k)) for i,j,k in (zip(frame_name,frame_labels,frame_pred))]))
        f_results.close()   
        cur_True = np.sum(frame_pred == frame_labels)
        total_True += cur_True
        total_pred += len(frame_pred)
        cur_nonintake = np.sum(frame_labels == 2)
        total_nonintake += cur_nonintake
        f_matrices.write("video name: {}  non_intake percentage: {}\n".format(video, cur_nonintake/len(frame_pred)))
        print("video name: {}  non_intake percentage: {}\n".format(video, cur_nonintake/len(frame_pred))) 
        f_matrices.write("acc: {}\n".format(cur_True/len(frame_pred)))
        
    f_matrices.write('summary\n')
    print('summary')
    f_matrices.write('acc: {0:.6f}\n'.format(total_True/total_pred))
    print('acc: {0:.6f}\n'.format(total_True/total_pred))
    f_matrices.write("{} videos in testing set\n".format(len(videos_test)))
    print("{} videos in testing set".format(len(videos_test)))
    f_matrices.write("{} samples in testing set\n".format(sample_num))
    print("{} samples in testing set".format(sample_num))
    f_matrices.write("non_intake percentage: {}\n".format(total_nonintake/total_pred))
    print("non_intake percentage: {}\n".format(total_nonintake/total_pred)) 
    f_matrices.close()
    f.close()
