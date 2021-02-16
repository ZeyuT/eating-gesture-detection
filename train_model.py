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
from constants import FRAME_LOC,WIDTH,HEIGHT,LABEL_TABLE
            

if __name__ == "__main__":  
    
    print('start')
    sys.stdout.flush()
    tf.keras.backend.set_floatx('float32')
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # following codes get the excution stuck    
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print("GPU is NOT AVAILABLE") 
    """
    'for debugging '
    seq_len = 30
    stride = 6
    batch_size = 8
    epochs = 1
    network = "CNNLSTM_Model"
    """
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    network = sys.argv[3]
    seq_len = int(sys.argv[4])
    stride = int(sys.argv[5])
    train = int(sys.argv[6])
    
    if network == "CNNLSTM_Model":
        model_type = 1
    elif network == "CNN3D_Model":
        model_type = 2
        
    print('model: {}'.format(network))
    print('batch size: {}  epochs: {}'.format(batch_size, epochs))
    print('sequence length: {}  stride: {}\n'.format(seq_len, stride))
    sys.stdout.flush()

    
    log_loc = './log_{}_{}_{}_{}'.format(network,epochs,seq_len,stride)
    model_loc = './model_{}_{}_{}_{}'.format(network,epochs,seq_len,stride)
    test_loc = './test_{}_{}_{}_{}'.format(network,epochs,seq_len,stride)
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


        
    print('Preparing training sample list...')
    start_time = time.time()
    sys.stdout.flush()
    
    video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith('p')]
    video_list.sort(reverse=False)
    train_split_ratio = 0.5
    
    #video_list = video_list[0:2]
    
    train_video_list = video_list[0:int(len(video_list)*train_split_ratio)]
    test_video_list = video_list[int(len(video_list)*train_split_ratio):]

    train_sample_list, train_label_list = get_list(train_video_list, seq_len, stride, model_type)
    
    input_ori = Input((seq_len,HEIGHT,WIDTH,3), name='ori',dtype=K.floatx())
    if model_type == 1:
        model = CNNLSTM_Model(input_ori)
    elif model_type == 2:
        model = CNN3D_Model(input_ori)
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])  
    model.summary()
    
    if train:
        train_gen = DataGenerator(train_sample_list, train_label_list, seq_len, model_type, batch_size=batch_size)
        """
        'for debugging '
        print(train_sample_list.shape)
        print(train_label_list.shape)
        train_gen = testG(train_sample_list, train_label_list, seq_len, model_type, batch_size=batch_size)
        for idx in range(train_gen.len()):
    
            x, y = train_gen.getitem(idx)
            print(x.shape, y.shape)
            break
        """
        elapsed_time = time.time() - start_time
        print('Finished training data generator preparation, elapsed time: {0:.6f} s'.format(elapsed_time)) 
        print("{} videos in training set".format(len(train_video_list)))
        print("{} patterns/samples in training set".format(len(train_sample_list)))
    
        print('Train {} model'.format(network))  
        start_time = time.time()
        sys.stdout.flush()         
                
        csv_logger = callbacks.CSVLogger('{}/train.log'.format(log_loc))
    
        '''
        early_stopping  = callbacks.EarlyStopping(monitor='train_accuray', min_delta=0.0001, patience=10, 
                                                    verbose=2, mode='auto', baseline=None,                                                                                                         restore_best_weights=True)
        '''
    
        hist = model.fit(train_gen,
                          epochs= epochs, 
                          verbose = 2,
                          callbacks = [csv_logger],
                          use_multiprocessing=True,
                          workers=16,
                          shuffle = False) # Already shuffled in generator at the end of each epoch      
        del train_gen
        elapsed_time = time.time() - start_time
        print('Finished Training Model, elapsed time: {0:.6f} s'.format(elapsed_time)) 
    
    else:
        model.load_weights('{}/model.h5'.format(model_loc))        
    
    print('Testing model...')
    start_time = time.time()
    sys.stdout.flush()
    
    test_model(model, test_video_list, test_loc, seq_len, stride, model_type)
    
    elapsed_time = time.time() - start_time
    print('Test finished, elapsed time: {0:.6f} s'.format(elapsed_time))
    