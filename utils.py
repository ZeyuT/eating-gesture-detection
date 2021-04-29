import sys
import os
import numpy as np
import cv2
import math
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM, LABEL_TABLE

def numeralize_labels(label):
    '''
    if label == "bite":
        return 0
    elif label == "drink":
        return 4
    elif label == "rest":
        return 2
    elif label == "utensiling":
        return 3
    elif label == "other":
        return 4
    # consider "unknown" being included in "other"
    elif label == "unknown":
        return 4
    '''
    return LABEL_TABLE[label]

def class_weights(label_list, weight_type):
    class_counts = []
    for label in range(LABEL_NUM):
        class_counts.append(np.sum(label_list==label))
    class_counts = np.array(class_counts)
    total = np.sum(class_counts)
    ret = []
    for i in range(LABEL_NUM):
        if class_counts[i] == 0:
            ret.append(0)
        else:
            if weight_type == 1:
                """
                version 1: n/(m*c(i))
                where n: total sample number, m: number of classes. c(i): number of samples belonging to the class
                """
                ret.append(total/(class_counts[i] * LABEL_NUM) / np.sum(total/(class_counts[class_counts!=0] * LABEL_NUM)))
            elif weight_type == 2:
                """
                version 2: 1/(1+c(i)/n)
                """
                ret.append(1/(1+LABEL_NUM*class_counts[i]/total) / np.sum(1/(1+LABEL_NUM*class_counts[class_counts!=0]/total)))
            elif weight_type == 3:
                """
                version 3: 1/c(i) / sum(1/c(i)) (Inverse Number of Sample)
                """
                ret.append(1/class_counts[i] / np.sum(1/class_counts[class_counts!=0]))
            elif weight_type == 4:
                """
                version 4: 1/c(i)**0.5 (Inverse of Square Root of Number of Samples)
                """
                ret.append(1/class_counts[i]**0.5 / np.sum(1/class_counts[class_counts!=0]**0.5))
            elif weight_type == 5:
                """
                version 5: (1-beta) / (1-beta**c(i)) (Effective Number of Samples)
                https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
                """
                beta = 0.99999
                ret.append((1-beta)/(1-beta**class_counts[i])/ np.sum((1-beta)/(1-beta**class_counts[class_counts!=0])))
            else:
                ret.append(1/LABEL_NUM)
    return np.array(ret)
            
def get_list(video_list, seq_len, stride, model_type):
    sample_list = []
    label_list = []
    label_counts = [0 for i in range(LABEL_NUM)]
    for video in video_list:
        frame_locs = []
        frame_labels = []
        f = open(FRAME_LOC + video + "/gt_frame_3labels.txt","r")
        gt_frame = [str.split(line, "\t") for line in f.readlines()]
        for frame_info in gt_frame:
            frame_locs.append(FRAME_LOC + video + "/" + frame_info[0])
            cur_labelIdx = numeralize_labels(str.split(frame_info[1], "\n")[0])
            #cur_labelIdx = int(frame_info[1])
            frame_labels.append(cur_labelIdx)
            label_counts[cur_labelIdx] += 1

        for i in range(0, len(frame_locs)-seq_len, stride):
            sample_list.append(frame_locs[i:i+seq_len])
            if model_type == 1:
                label_list.append(frame_labels[i:i+seq_len])
            elif model_type == 2 or model_type == 3:
                label_list.append(frame_labels[i+seq_len-1])
    return np.array(sample_list), np.array(label_list), label_counts
    
class testG():
    def __init__(self, sample_list, label_list, seq_len, model_type, batch_size=32, shuffle=True):
        'Initialization'
        self.sample_list = sample_list
        self.label_list = label_list
        self.seq_len = seq_len
        self.model_type = model_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def len(self):
        'Number of batch in the Sequence'
        return math.ceil(len(self.sample_list) / self.batch_size)

    def getitem(self, idx):
        'Gets batch at position idx'
        batch_x = []
        batch_y = []
        batch_sample_list = self.sample_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        batch_label_list = self.label_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        for sample_list in batch_sample_list:
            cur_x = []
            for frame_loc in sample_list:
                cur_img = cv2.imread(frame_loc, cv2.IMREAD_GRAYSCALE)/255.0
                cur_img = cv2.resize(cur_img, (WIDTH, HEIGHT))
                cur_x.append(cur_img)
            batch_x.append(cur_x) 
        batch_x = np.reshape(batch_x,(-1,self.seq_len,HEIGHT,WIDTH,CHANNEL))
        
        if self.model_type == 1:
            for label_list in batch_label_list:
                cur_y = []
                for label in label_list:
                    cur_y.append(label)
                batch_y.append(cur_y)  
            batch_y = np.reshape(batch_y, (-1,self.seq_len))
            
        elif self.model_type == 2 or self.model_type == 3:
            for label in batch_label_list:
                batch_y.append(label)         
            batch_y = np.array(batch_y)                   
        
        return batch_x, batch_y
        
    def on_epoch_end(self):
        if self.shuffle == True:
            self.sample_list, self.label_list = shuffle(self.sample_list,self.label_list)
                    
class DataGenerator(keras.utils.Sequence):
    def __init__(self, sample_list, label_list, seq_len, model_type, batch_size=32, max_delta=0.4, flip_ratio=0.5, shuffle=True):
        'Initialization'
        self.sample_list = sample_list
        self.label_list = label_list
        self.seq_len = seq_len
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_delta = max_delta
        self.flip_ratio = flip_ratio
        self.shuffle = shuffle
        self.on_epoch_end()
        self.batch_weights = np.ones(LABEL_NUM)

    def __len__(self):
        'Number of batch in the Sequence'
        return math.ceil(len(self.sample_list) / self.batch_size)

    def __getitem__(self, idx):
        'Gets batch at position idx'
        batch_x = []
        batch_y = []
        '''
        Randomly augment samples by horizontal flipping and adding a random brightness jitter
        each sample has 50% chance to be horizontal flipped
        each sample is added with a random brightness jitter
        '''
        batch_flip_flags = np.random.rand(self.batch_size) < self.flip_ratio
        batch_deltas = self.max_delta*(2*np.random.rand(self.batch_size)-1)
        batch_sample_list = self.sample_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        batch_label_list = self.label_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        for sample_list,cur_flip_flag,cur_delta in zip(batch_sample_list,batch_flip_flags,batch_deltas):
            cur_x = []
            for frame_loc in sample_list:
                cur_img = cv2.imread(frame_loc, cv2.IMREAD_UNCHANGED)
                cur_img = self.flip_augment(cur_img,cur_flip_flag)
                cur_img = self.brightness_augment(cur_img, cur_delta)
                cur_img = cur_img/255.0
                #cur_img = cv2.resize(cur_img, (WIDTH, HEIGHT))
                cur_x.append(cur_img)
            batch_x.append(cur_x) 
        batch_x = np.reshape(batch_x,(-1,self.seq_len,HEIGHT,WIDTH,CHANNEL))
        
        if self.model_type == 1:
            for label_list in batch_label_list:
                cur_y = []
                for label in label_list:
                    cur_y.append(label)
                batch_y.append(cur_y)  
            batch_y = np.reshape(batch_y, (-1,self.seq_len))
            
        elif self.model_type == 2 or self.model_type == 3:
            for label in batch_label_list:
                batch_y.append(label)         
            batch_y = np.array(batch_y)    
        del cur_img   
        return batch_x, batch_y
        
    def flip_augment(self, img, flip_flag): 
        if flip_flag:
            return img[:,::-1,:]
        else:
            return img
            
    def brightness_augment(self, img, delta): 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert to hsv
        hsv = np.array(hsv, dtype=np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * (1+delta) #scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255.0 #reset out of range values
        bgr = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        return bgr
         
    def on_epoch_end(self):
        if self.shuffle == True:
            self.sample_list, self.label_list = shuffle(self.sample_list,self.label_list)

def brightness_augment(img, delta): 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * (1+delta) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255.0 #reset out of range values
    bgr = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return bgr
            
def test_model(model, videos_test, test_loc, seq_len, model_type,test_stride=1):
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
            cur_img = cv2.imread(FRAME_LOC + video + "/" + frame_info[0], cv2.IMREAD_UNCHANGED)
            
            #gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            #gray_img = np.expand_dims(gray_img, axis=-1)
            #cur_img = np.concatenate([gray_img,gray_img,gray_img],axis=-1)
            
            cur_img = cur_img / 255.0 
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
