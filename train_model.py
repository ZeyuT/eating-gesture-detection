import sys
import os
import subprocess
import numpy as np
import cv2
import time
import math
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K

from models import CNN3D_Model, CNNLSTM_Model, RESLSTM_Model
from utils import class_weights, get_list, test_model, DataGenerator, testG
from constants import FRAME_LOC,WIDTH,HEIGHT,CHANNEL,LABEL_NUM

from math import e
                    
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
    '''
    num_class = len(weights)


    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_true_encoded = tf.one_hot(tf.cast(y_true,tf.int32), num_class)
        # clip to prevent NaN"s and Inf"s
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calculation
        loss = y_true_encoded * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    '''   
    '''
    A simpler version using Keras sparse_categorical_crossentropy
    https://github.com/tensorflow/models/blob/master/official/nlp/modeling/losses/weighted_sparse_categorical_crossentropy.py
    '''    
    weights = tf.cast(weights, tf.float32)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    def loss(y_true, y_pred):
        raw_losses = scce(y_true, y_pred)
        return tf.math.divide_no_nan(
                tf.reduce_sum(raw_losses * weights), tf.reduce_sum(weights))
            
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
            
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA 
    '''
    train = 0: test model only
          = 1: train and test model on raw video data
          = 2: debug mode on simulated data
          = 3: train and test model on raw simulated data
          = 4: test model on raw simulated data
          = 5: continue training on raw video data and test model
    model_type  = 1: seq2seq frame-wise prediction
                = 2: seq2one frame prediction
    '''
    train = int(sys.argv[1])
    if train == 2:
        #for debugging
        seq_len = 2
        stride = 1
        batch_size = 4
        epochs = 1
        network = "CNNLSTM_Model"
        weight_type = 0
    else:
        batch_size = int(sys.argv[2])
        epochs = int(sys.argv[3])
        network = sys.argv[4]
        seq_len = int(sys.argv[5])
        stride = int(sys.argv[6])
        video_num = int(sys.argv[7])
        weight_type = int(sys.argv[8])
    if network == "CNNLSTM_Model" or network == "RESLSTM_Model":
        model_type = 1
    elif network == "CNN3D_Model":
        model_type = 2
        
    print("model: {}".format(network))
    print("batch size: {}  epochs: {}".format(batch_size, epochs))
    print("sequence length: {}  stride: {}\n".format(seq_len, stride))
    print("weight_type: {}\n".format(weight_type))
    sys.stdout.flush()

    #v{x}: version x for class weight calculation
    #bcb: batch class balance. needs to find a way to convert y_true to numpy. leave it on todo list.
    if train == 5:
        print("continue training models\n")
        log_loc = "./log_{}_{}_{}_{}_{}_v{}".format(network,30+epochs,seq_len,stride,video_num,weight_type)
        model_loc = "./model_{}_{}_{}_{}_{}_v{}".format(network,30+epochs,seq_len,stride,video_num,weight_type)
        test_loc = "./test_{}_{}_{}_{}_{}_v{}_60videos".format(network,30+epochs,seq_len,stride,video_num,weight_type)
    else:
        log_loc = "./log_{}_{}_{}_{}_{}_v{}".format(network,epochs,seq_len,stride,video_num,weight_type)
        model_loc = "./model_{}_{}_{}_{}_{}_v{}".format(network,epochs,seq_len,stride,video_num,weight_type)
        test_loc = "./test_{}_{}_{}_{}_{}_v{}".format(network,epochs,seq_len,stride,video_num,weight_type)    
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
    
    if train in [2,3,4]: 
        train_video_list = ["train"]
        test_video_list = ["test"]
    else:
        video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith("p")]
        video_list.sort(reverse=False)
        video_list = video_list[0:video_num]
        train_split_ratio = 0.8
        train_video_list = video_list[0:int(len(video_list)*train_split_ratio)]
        test_video_list = video_list[int(len(video_list)*train_split_ratio):]
        #test_video_list = train_video_list
        #test_video_list = ['p026_c1']
        #test_video_list = ["p207_c3","p176_c1","p179_c3","p177_c2","p176_c2"]
    train_sample_list, train_label_list, label_counts = get_list(train_video_list, seq_len, stride, model_type)
    weights = class_weights(train_label_list,weight_type)
    print("{} videos in training set".format(len(train_video_list)))
    print("{} patterns in training set".format(len(train_sample_list)))
    print("class sizes: {}".format(label_counts))
    print("class weights: {}".format(weights)) 

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
                    #img = np.squeeze(img)
                    #cv2.imwrite("./test/{}_{}.jpg".format(i,j),img*255)
                    print(img.shape)
                    count += 1
            exit(0)
    else:
        train_gen = DataGenerator(train_sample_list, train_label_list, seq_len, model_type, batch_size=batch_size)
        def generator():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_gen, use_multiprocessing=True)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_x, batch_y = next(multi_enqueuer.get()) 
                yield batch_x, batch_y
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=(tf.TensorShape([None,None,None,None,None]),
                                                                tf.TensorShape([None,None])))

        dataset = dataset.with_options(options)
                                                       
    elapsed_time = time.time() - start_time
    print("Finished training data generator preparation, elapsed time: {0:.6f} s".format(elapsed_time)) 
        
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # loss function depends on the actual NN
        loss = weighted_sparse_categorical_crossentropy(weights)
        input_ori = Input((seq_len,HEIGHT,WIDTH,CHANNEL), name="ori",dtype=K.floatx())
        if  network == "CNNLSTM_Model":
            model = CNNLSTM_Model(input_ori)
        elif network == "CNN3D_Model":
            model = CNN3D_Model(input_ori)
        elif network == "RESLSTM_Model":
            model = RESLSTM_Model(input_ori)
            
        model.compile(
                      loss = loss,
                      #loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                      metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                      )  
    model.summary()
    
    if train !=0 and train != 4:        
        print("Training {} model".format(network))  
        start_time = time.time()
        sys.stdout.flush()      

        #Resume training   
        if train == 5:
            model.load_weights("./model_{}_{}_{}_{}_{}_v{}/model.h5".format(network,30,seq_len,stride,video_num,weight_type))

        csv_logger = callbacks.CSVLogger("{}/train.log".format(log_loc))
        
        early_stopping  = callbacks.EarlyStopping(monitor="loss", min_delta=0.0001, patience=10, 
                                                    verbose=2, mode="auto", baseline=None, restore_best_weights=True)

        hist = model.fit(dataset,
                          epochs= epochs, 
                          steps_per_epoch = math.ceil(len(train_label_list) / batch_size),
                          verbose = 1,
                          callbacks = [csv_logger,early_stopping],
                          shuffle = False) # Already shuffled in generator at the end of each epoch      

        model.save_weights("{}/model.h5".format(model_loc))        
        elapsed_time = time.time() - start_time
        print("Finished training model, elapsed time: {0:.6f} s".format(elapsed_time)) 
    
    else:
        model.load_weights("{}/model.h5".format(model_loc))        
    
    print("Testing model...")
    start_time = time.time()
    sys.stdout.flush()
    
    test_model(model, test_video_list, test_loc, seq_len, model_type)
    
    elapsed_time = time.time() - start_time
    print("Test finished, elapsed time: {0:.6f} s".format(elapsed_time))