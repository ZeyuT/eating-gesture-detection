import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU,Conv2D,Activation,MaxPooling2D,concatenate,ConvLSTM2D,BatchNormalization,LSTM,GRU,TimeDistributed
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras import Model,initializers, regularizers
from constants import LABEL_NUM
def Conv3D_Block(input_tensor, out_channels):
    # first layer
    x = Conv3D(filters=out_channels, kernel_size=[3,3,3], padding="same",
                kernel_initializer="he_normal",bias_initializer="zeros")(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling3D(pool_size=[2,2,2], strides=[2, 2, 2], padding="same")(x)
    
    # second layer
    x = Conv3D(filters=out_channels, kernel_size=[3,3,3], padding="same",
                kernel_initializer="he_normal",bias_initializer="zeros")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling3D(pool_size=[2,2,2], strides=[2, 2, 2], padding="same")(x)
    return x
        
def TimeDistributed_Conv2D_Block(input_tensor, out_channels):
    # first layer
    x = TimeDistributed(Conv2D(filters=out_channels, kernel_size=[3,3], padding="same",
                        kernel_initializer="he_normal",
                        bias_initializer="zeros"))(input_tensor)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=[2,2], strides=[2,2], padding="same"))(x)
    
    # second layer
    x = TimeDistributed(Conv2D(filters=out_channels, kernel_size=[3,3], padding="same",
                        kernel_initializer="he_normal",
                        bias_initializer="zeros"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=[2,2], strides=[2,2], padding="same"))(x)   
    return x
    
def CNN3D_Model(input_tensor,n_filters=8):
    x = Conv3D_Block(input_tensor,n_filters)
    x = Conv3D_Block(x,n_filters*2)
    #x = Conv3D_Block(x,n_filters*4)
    
    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(LABEL_NUM)(x)
    
    outputs = Activation("softmax")(x)
    model = Model(inputs=[input_tensor], outputs=[outputs], name="CNN-3D")
    return model

         
def CNNLSTM_Model(input_tensor,n_filters=8):        
    x = TimeDistributed_Conv2D_Block(input_tensor,n_filters)
    x = TimeDistributed_Conv2D_Block(x,n_filters)
    x = TimeDistributed_Conv2D_Block(x,n_filters*2)
    x = TimeDistributed_Conv2D_Block(x,n_filters*2)
    x = TimeDistributed_Conv2D_Block(x,n_filters*4)   

    x = TimeDistributed(Flatten())(x)

    x = TimeDistributed(Dense(64))(x)
    x = TimeDistributed(LeakyReLU())(x)
    #x = TimeDistributed(Dropout(0.2))(x)
    '''
    x = LSTM(units = 128,
            kernel_initializer='he_normal', bias_initializer='zeros',
            return_sequences=True)(x)
    
    x = LSTM(units = 64, 
            kernel_initializer='he_normal', bias_initializer='zeros',
            return_sequences=True)(x)
    '''
    x = LSTM(units = 32, 
            kernel_initializer='he_normal', bias_initializer='zeros',
            return_sequences=True)(x)
           
    x = TimeDistributed(Dense(LABEL_NUM))(x)
    #x = Dense(LABEL_NUM)(x)
    outputs = Activation("softmax")(x)
    model = Model(inputs=[input_tensor], outputs=[outputs], name="CNN-LSTM")
    return model