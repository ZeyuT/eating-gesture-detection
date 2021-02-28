import sys
import os
import subprocess
import numpy as np
import cv2
import time

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K

from models import CNN3D_Model, CNNLSTM_Model
from utils import get_list, test_model, DataGenerator, testG
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_TABLE
            
def class_weights(label_counts):
    ret = []
    total = float(sum(label_counts))
    for count in label_counts:
        ret.append(total/(count * len(label_counts)))
    return np.array(ret)
        
def weighted_sparse_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.sparse_categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_sparse_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer="adam")
    """
    num_class = len(weights)
    weights = K.constant(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # one-hot encode labels
        # to_categorical does not work here. It takes array/list as arguement, but I cannot find a easy way to convert 
        # tensor y_true to array
        # y_true_encoded = to_categorical(y_true, num_classes=weights.shape[0])
        # so hand write one-hot encoding
        temp = []
        for i in range(num_class):
            temp.append(tf.where(tf.equal(y_true, i), 1.0, 0.0))
        y_true_encoded = tf.concat(temp, axis=-1)
        # clip to prevent NaN"s and Inf"s
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true_encoded * K.log(y_pred) * weights
        loss = -K.mean(loss, -1)
        return loss
    
    return loss
    
if __name__ == "__main__":  
    
    print("start")
    sys.stdout.flush()
    tf.keras.backend.set_floatx("float32")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # following codes get the excution stuck    
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print("GPU is NOT AVAILABLE") 
    '''
    train = 0: test model only
          = 1: train and test model on raw video data
          = 2: debug mode on simulated data
          = 3: train and test model on raw simulated data
          = 4: test model on raw simulated data
    '''
    train = int(sys.argv[1])
    if train == 2:
        #for debugging
        seq_len = 2
        stride = 1
        batch_size = 8
        epochs = 1
        network = "CNN3D_Model"
    else:
        batch_size = int(sys.argv[2])
        epochs = int(sys.argv[3])
        network = sys.argv[4]
        seq_len = int(sys.argv[5])
        stride = int(sys.argv[6])
    
    if network == "CNNLSTM_Model":
        model_type = 3
    elif network == "CNN3D_Model":
        model_type = 2
        
    print("model: {}".format(network))
    print("batch size: {}  epochs: {}".format(batch_size, epochs))
    print("sequence length: {}  stride: {}\n".format(seq_len, stride))
    sys.stdout.flush()

    
    log_loc = "./log_{}_{}_{}_{}".format(network,epochs,seq_len,stride)
    model_loc = "./model_{}_{}_{}_{}".format(network,epochs,seq_len,stride)
    test_loc = "./test_{}_{}_{}_{}".format(network,epochs,seq_len,stride)
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
    
    if train >= 2: 
        train_video_list = ["train6"]
        test_video_list = ["test"]
    else:
        video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith("p")]
        video_list.sort(reverse=False)
        train_split_ratio = 0.8
        train_video_list = video_list[0:int(len(video_list)*train_split_ratio)]
        test_video_list = video_list[int(len(video_list)*train_split_ratio):]
        
    train_sample_list, train_label_list, label_counts = get_list(train_video_list, seq_len, stride, model_type)
    weights = class_weights(label_counts)
    print("{} videos in training set".format(len(train_video_list)))
    print("{} patterns in training set".format(len(train_sample_list)))
    print("class sizes: {}".format(label_counts))
    print("class weights: {}".format(weights)) 
    # loss function depends on the actual NN
    loss = weighted_sparse_categorical_crossentropy(weights)
    
    input_ori = Input((seq_len,HEIGHT,WIDTH,CHANNEL), name="ori",dtype=K.floatx())
    if model_type == 1 or model_type == 3:
        model = CNNLSTM_Model(input_ori)
    elif model_type == 2:
        model = CNN3D_Model(input_ori)
    model.compile(
                  loss = loss,
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics = [tf.keras.metrics.CategoricalAccuracy()]
                  )  
    model.summary()
    
    if train > 0 and train < 4:
        if train == 2:
            # for debugging
            print(train_sample_list.shape)
            print(train_label_list.shape)
            train_gen = testG(train_sample_list, train_label_list, seq_len, model_type, batch_size=batch_size)
            count = 0
            for idx in range(train_gen.len()):
                x, y = train_gen.getitem(idx)
                for i,seq in enumerate(x):
                    for j,img in enumerate(seq):
                        img = np.squeeze(img)
                        #cv2.imwrite("./test/{}_{}.jpg".format(i,j),img*255)
                        print(img.shape)
                        count += 1
                print(x.shape, y.shape)
                print(y)
                exit(0)
        else:
            train_gen = DataGenerator(train_sample_list, train_label_list, seq_len, model_type, batch_size=batch_size)
        
        elapsed_time = time.time() - start_time
        print("Finished training data generator preparation, elapsed time: {0:.6f} s".format(elapsed_time)) 
        
        print("Training {} model".format(network))  
        start_time = time.time()
        sys.stdout.flush()         
        csv_logger = callbacks.CSVLogger("{}/train.log".format(log_loc))
        
    
        """
        early_stopping  = callbacks.EarlyStopping(monitor="train_accuray", min_delta=0.0001, patience=10, 
                                                    verbose=2, mode="auto", baseline=None,                                                                                                         restore_best_weights=True)
        """

        hist = model.fit(train_gen,
                          epochs= epochs, 
                          verbose = 2,
                          callbacks = [csv_logger],
                          #use_multiprocessing=True,
                          #workers=16,
                          shuffle = False) # Already shuffled in generator at the end of each epoch      
        del train_gen
        model.save_weights("{}/model.h5".format(model_loc))        
        elapsed_time = time.time() - start_time
        print("Finished training model, elapsed time: {0:.6f} s".format(elapsed_time)) 
    
    else:
        model.load_weights("{}/model.h5".format(model_loc))        
    
    print("Testing model...")
    start_time = time.time()
    sys.stdout.flush()
    
    test_model(model, test_video_list, test_loc, seq_len, stride, model_type)
    
    elapsed_time = time.time() - start_time
    print("Test finished, elapsed time: {0:.6f} s".format(elapsed_time))