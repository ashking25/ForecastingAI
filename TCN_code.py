from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import keras
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed,RepeatVector,merge,Conv1D,MaxPooling1D
from keras.layers import LSTM,Reshape,Flatten,Permute,Input
from keras.layers import Add,Conv2D,MaxPooling2D,Dropout,BatchNormalization
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from keras.initializers import RandomNormal


def ResidualBlock(inputs,n_outputs,k,d,dropout_rate):
    """ Residual block
    inputs: previous layer
    n_outputs: size of the layers (number of features)
    k: kernel size
    d: dilation rate
    dropout_rate: percentage to dropout

    return
    f: final layer in residual block"""
    input_dim = int(inputs.shape[2])
    c1 = Conv1D(n_outputs,kernel_size=k,dilation_rate=d,\
                input_shape=(input_dim,1),activation='relu',\
               padding='same',kernel_initializer=RandomNormal(mean=0,stddev=0.01))(inputs)
    b1 = BatchNormalization()(c1)
    d1 = Dropout(dropout_rate,noise_shape=(1,1,n_outputs))(b1)
    c2 = Conv1D(n_outputs,kernel_size=k,dilation_rate=d,\
                activation='relu',padding='same',kernel_initializer=RandomNormal(mean=0,stddev=0.01))(d1)
    b2 = BatchNormalization()(c2)
    d2 = Dropout(dropout_rate,noise_shape=(1,1,n_outputs))(b2)
    e = Dense(n_outputs, activation=None)(inputs)
    f = Add()([e,d2])
    g = Dense(n_outputs, activation='relu',name='ResidBlock_'+str(d),kernel_initializer=RandomNormal(mean=0,stddev=0.01))(f)
    return g


def TCN(input_dim,time_steps,layers,features,dilation_rate=2.,kernel_size=3, dropout=0.):
    """ Temporal Convolution Neural Network
        By using dilaiton with enough layers, all values should be causally
        connnected"""

    num_channels = [features]*layers
    layers = []
    num_levels = len(num_channels)
    inputs = Input(shape=input_dim)

    for i in range(num_levels):
        dilation_size = int(dilation_rate ** i)

        out_channels = num_channels[i]
        if i == 0:
            in_channels = input_dim
            mod = ResidualBlock(inputs,out_channels,kernel_size,dilation_size,dropout)
        else:
            in_channels = num_channels[i-1]
            mod = ResidualBlock(mod,out_channels,kernel_size,dilation_size,dropout)
    cEnd = Conv1D(1,kernel_size=1,dilation_rate=1,activation='sigmoid',\
               padding='same',kernel_initializer=RandomNormal(mean=0,stddev=0.01))(mod)
    bcEnd = BatchNormalization()(cEnd)
    mod1 = Flatten()(bcEnd)
    mod2 = Dense(1,activation='linear',kernel_initializer=RandomNormal(mean=0,stddev=0.01),bias_initializer=keras.initializers.Constant(value=14.))(mod1) # the last output should be able to reach all of y values
    model = Model(input=[inputs], output=mod2)
    return model

def dataloader(batch_size=10,nstart=0,num_eq=1000,num_days=30,PATH='/Users/ashking/QuakeFinders/data/mocks'):
    """ Build generator to load the data progressively """
    while True:
        number_EQ    = np.random.randint(nstart,num_eq,num_eq*num_days) # draw a random distribution of earth quakes
        number_days  = np.random.randint(0,num_days,num_eq*num_days) # draw randomly from 0-30 days before the quakes
        num_batch    = int(num_eq*num_days/batch_size) # number of batches

        for i in range(num_batch):
            start    = i*batch_size
            y        = []
            data     = []

            for j,(EQ,day) in enumerate(zip(number_EQ[start:start+batch_size],number_days[start:start+batch_size])):
                data += [np.load(PATH+'/EQ'+str(EQ)+'_'+str(day)+'daysuntilEQ.npy')]
                y    += [day]

            # This isn't normalized! We should think about this
            data = np.reshape(data,(len(data),np.shape(data)[1],1))
            y    = np.array(y)

            yield (data,y)


if 1 == 2:
    LAYERS    = 12
    N_HIDDEN  = 1024 # hidden layer, i.e. num of features
    INPUT_DIM = 24*3600 # seconds in a day, number of channels -1
    TIME_STEPS    = 1
    KERNEL_SIZE   = 12
    layers_effect = 1+2*(KERNEL_SIZE-1)*(2**LAYERS-1)
    BATCH_SIZE    = 10
    EPOCHS        = 2
    STEPS_PER_EPOCH = int(1000*30/BATCH_SIZE)

    train_gen = dataloader(batch_size=BATCH_SIZE)
    test_gen  = dataloader(batch_size=BATCH_SIZE) # its not clear to me this will
                        #give seperate test data, so be careful in interpretations

    #print 'the number of effected points',layers_effect

    model = TCN(INPUT_DIM,TIME_STEPS,LAYERS,N_HIDDEN,kernel_size=KERNEL_SIZE,dropout=0.)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.01, amsgrad=False)

    #Compile model
    model.compile(loss='mean_squared_error',  metrics=['accuracy'],optimizer=adam)
    #print model.summary()
    # Record whats going on using TensorBoard
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()),histogram_freq=1,write_images=True)

    # Fit using Fit_generator
    #model.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,verbose=1,\
    #    validation_data=test_gen,validation_steps=5,callbacks=[tensorboard])
