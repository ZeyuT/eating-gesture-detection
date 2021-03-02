import sys
import os
import numpy as np
import cv2
import math
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_TABLE,LABEL_NUM

def numeralize_labels(label):
    if label == "bite":
        return 0
    elif label == "drink":
        return 1
    elif label == "rest":
        return 2
    elif label == "utensiling":
        return 3
    elif label == "other":
        return 4
    elif label == "unknown":
        return 5
        
def get_list(video_list, seq_len, stride, model_type):
    sample_list = []
    label_list = []
    label_counts = [0 for i in range(LABEL_NUM)]

    for video in video_list:
        frame_locs = []
        frame_labels = []
        f = open(FRAME_LOC + video + "/gt_frame.txt","r")
        gt_frame = [str.split(line, "\t") for line in f.readlines()]
        for frame_info in gt_frame:
            frame_locs.append(FRAME_LOC + video + "/" + frame_info[0])
            #cur_labelIdx = numeralize_labels(str.split(frame_info[1], "\n")[0])
            cur_labelIdx = int(frame_info[1])
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
        for frame_list in batch_sample_list:
            cur_x = []
            for frame_loc in frame_list:
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
    def __init__(self, sample_list, label_list, seq_len, model_type, batch_size=32, shuffle=True):
        'Initialization'
        self.sample_list = sample_list
        self.label_list = label_list
        self.seq_len = seq_len
        self.model_type = model_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Number of batch in the Sequence'
        return math.ceil(len(self.sample_list) / self.batch_size)

    def __getitem__(self, idx):
        'Gets batch at position idx'
        batch_x = []
        batch_y = []
        batch_sample_list = self.sample_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        batch_label_list = self.label_list[idx * self.batch_size:
                                      (idx + 1) * self.batch_size]
        for frame_list in batch_sample_list:
            cur_x = []
            for frame_loc in frame_list:
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
    
def test_model(model, videos_test, test_loc, seq_len, stride, model_type):
    test_batch = 1024
    f_matrices = open(test_loc + "/matrices.txt",'w')
    print("{} videos in testing set".format(len(videos_test)))
    print("testing batch size: {}\n".format(test_batch))
    total_TP = 0
    total_pred = 0
    sample_num = 0
    for video in videos_test:
        frames = []
        frame_labels = []

        f = open(FRAME_LOC + video + "/gt_frame.txt","r")
      
        gt_frame = [str.split(line, "\t") for line in f.readlines()]
        for frame_info in gt_frame:
            cur_img = cv2.imread(FRAME_LOC + video + "/" + frame_info[0], cv2.IMREAD_GRAYSCALE)/255.0 
            cur_img = cv2.resize(cur_img, (WIDTH, HEIGHT))
            frames.append(cur_img)
            cur_labelIdx = int(frame_info[1])
            #cur_labelIdx = numeralize_labels(str.split(frame_info[1], "\n")[0])
            frame_labels.append(cur_labelIdx)
        frame_labels = np.array(frame_labels)
        pred = []
        x_test = []
        if model_type == 1:
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
            'build prediction heat map'
            pred_heat = np.zeros((len(frame_labels),6))
            for i in range(len(pred)):
                for j in range(len(pred[i])):
                    pred_heat[stride*i+j,pred[i,j]] += 1
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
        
        f_results = open(test_loc + "/pred_{}.txt".format(video),'w')
        f_results.write("\n".join(["frame{:05d}.ppm\t{}\t{}".format(i,j,k) for i,(j,k) in enumerate(zip(frame_labels,frame_pred))]))
        f_results.close()   
        cur_TP = np.sum((frame_pred == frame_labels))

        total_TP += cur_TP
        total_pred += len(frame_pred)
        
        f_matrices.write('video name: {}\n'.format(video))
        print('video name: {}'.format(video))  
        f_matrices.write('acc: {}\n'.format(cur_TP/len(frame_pred)))
        print('acc: {}\n'.format(cur_TP/len(frame_pred)))  
        
    f_matrices.write('summary\n')
    print('summary')
    f_matrices.write('acc: {0:.6f}\n'.format(total_TP/total_pred))
    print('acc: {0:.6f}\n'.format(total_TP/total_pred))
    f_matrices.write("{} videos in testing set\n".format(len(videos_test)))
    print("{} videos in testing set".format(len(videos_test)))
    f_matrices.write("{} samples in testing set".format(sample_num))
    print("{} samples in testing set".format(sample_num))
    f_matrices.close()
    f.close()
