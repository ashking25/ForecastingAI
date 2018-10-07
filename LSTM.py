### LSTM ###
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import normalize,MinMaxScaler
import matplotlib
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization,Dense,TimeDistributed,RepeatVector,merge,Conv1D,MaxPooling1D
from keras.layers import LSTM,Reshape,Flatten,Permute,Input,Conv2D,MaxPooling2D,Dropout,Add
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal

def ResidualBlock_LSTM(inputs,n_outputs,k,d,dropout_rate):
    """ Residual block
    inputs: previous layer
    n_outputs: size of the layers (number of features)
    k: kernel size
    d: dilation rate
    dropout_rate: percentage to dropout

    return
    f: final layer in residual block"""
    input_dim = int(inputs.shape[2])
    c1 = TimeDistributed(Conv1D(n_outputs,kernel_size=k,dilation_rate=d,\
                input_shape=(input_dim,1),activation='relu',\
               padding='same',kernel_initializer=RandomNormal(mean=0,stddev=0.01)))(inputs)
    b1 = TimeDistributed(BatchNormalization())(c1)
    d1 = TimeDistributed(Dropout(dropout_rate,noise_shape=(1,1,n_outputs)))(b1)
    c2 = TimeDistributed(Conv1D(n_outputs,kernel_size=k,dilation_rate=d,\
                activation='relu',padding='same',\
                kernel_initializer=RandomNormal(mean=0,stddev=0.01)))(d1)
    b2 = TimeDistributed(BatchNormalization())(c2)
    d2 = TimeDistributed(Dropout(dropout_rate,noise_shape=(1,1,n_outputs)))(b2)
    e = TimeDistributed(Dense(n_outputs, activation=None))(inputs)
    f = Add()([e,d2])
    g = TimeDistributed(Dense(n_outputs, activation='relu',name='ResidBlock_'+str(d),\
              kernel_initializer=RandomNormal(mean=0,stddev=0.01)))(f)
    return g



def reshapedata(data,timesteps,featuresize):
    """ reshape data into N x timesteps x featuresize"""
    samples = np.size(data)/timesteps/featuresize
    newdata = np.reshape(data[:int(timesteps*featuresize*samples)],(int(samples),int(timesteps),int(featuresize)))
    return newdata

def dataloader_2(timesteps,feature_length,lookback=3,batch_size=10,nstart=0,num_eq=1000,num_days=30,PATH='/Users/ashking/QuakeFinders/data/mocks'):
    """ Build generator to load the data progressively """
    while True:
        number_EQ    = np.random.randint(nstart,num_eq,num_eq*num_days) # draw a random distribution of earth quakes
        number_days  = np.random.randint(0,num_days-lookback,num_eq*num_days) # draw randomly from 0-30 days before the quakes
        num_batch    = int(num_eq*num_days/batch_size) # number of batches

        for i in range(num_batch):
            start    = int(i*batch_size)
            y        = []
            data     = []

            for j,(EQ,day) in enumerate(zip(number_EQ[start:int(start+batch_size)],number_days[start:int(start+batch_size)])):
                newdata = ()
                for s in range(lookback):
                    dataset0 = np.load(PATH+'/EQ'+str(EQ)+'_'+str(day+lookback-1-s)+'daysuntilEQ.npy')
                    if np.size(newdata) <= 0:
                        newdata = reshapedata(dataset0,timesteps,feature_length)
                    else:
                        newdata = np.append(newdata,reshapedata(dataset0,timesteps,feature_length),axis=1)
                data += [newdata]
                y    += [day]

            # This isn't normalized! We should think about this
            data = np.reshape(data,(len(data),int(timesteps*lookback),int(feature_length),1))
            y    = np.array(y)

            yield (data,y)


def myLSTM(input_dim,time_steps,layers,features,pooling=2,kernel_size=7, dropout=0.):
    """ LSTM with Convolution Neural Network to reduce size"""

    num_channels = [features]*layers
    layers = []
    num_levels = len(num_channels)
    inputs = Input(shape=input_dim)

    for i in range(num_levels):
        dilation_size = 1
        out_channels = num_channels[i]
        if i == 0:
            in_channels = input_dim
            mod = ResidualBlock_LSTM(inputs,out_channels,kernel_size,dilation_size,dropout)
            pool = TimeDistributed(MaxPooling1D(pool_size=(pooling)))(mod)
        else:
            in_channdels = num_channels[i-1]
            mod = ResidualBlock_LSTM(pool,out_channels,kernel_size,dilation_size,dropout)
            pool = TimeDistributed(MaxPooling1D(pool_size=(pooling)))(mod)

    cEnd = TimeDistributed(Conv1D(1,kernel_size=1,dilation_rate=1,activation='sigmoid',\
               padding='same',kernel_initializer=RandomNormal(mean=0,stddev=0.01)))(pool)
    resh = Reshape((timesteps*lookback,1*int(data_length/pooling**num_levels)),\
        input_shape=(timesteps*lookback,int(data_length/pooling**num_levels),1))(cEnd)
    lstm1=LSTM(10 ,return_sequences=True,activation='tanh')(resh)
        #,input_shape=(timesteps*lookback,int(data_length/pool**nCNN),features)))
    lstm2=LSTM(features, activation='tanh')(lstm1)
    #mod1 = Flatten()(cEnd)
    mod2 = Dense(1,activation='linear',kernel_initializer=RandomNormal(mean=0,stddev=0.01))(lstm2) # the last output should be able to reach all of y values
    model = Model(input=[inputs], output=mod2)
    return model


### Data params ###
BATCH_SIZE    = 2
EPOCHS        = 200
STEPS_PER_EPOCH = 50# int(900/BATCH_SIZE)#*30
timesteps     = 100
data_length   = int(3600*24/timesteps)
lookback      = 3
input_dim     = (int(timesteps*lookback),int(data_length),1)
dropout       = 0
features      = 100
layers        = 4



### Data ###
train_gen = dataloader_2(timesteps,data_length,lookback=lookback,batch_size=BATCH_SIZE,
    num_eq=900,PATH='../data/mocks')
test_gen  = dataloader_2(timesteps,data_length,lookback=lookback,batch_size=25,
    nstart=901,num_eq=1000,PATH='/home/ashking/quake_finder/data/mocks') #
#test_data = next(test_gen)
#train_data = next(train_gen)



### Model ###
model2 = myLSTM(input_dim,timesteps,layers,features)

#Optimizer
adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
#backends
tensorboard = TensorBoard(log_dir="../data/mocks/logs/{}".format(time()),histogram_freq=10,write_images=True)

#Compile model
model2.compile(loss='mean_squared_error',  metrics=['accuracy'],optimizer=adam)
#Callbacks
#tensorboard = TensorBoard(log_dir="../data/mocks/logs/LSTM_{}".format(time()))
#filepath = '../data/mocks/logs/model_LSTM_v0_{epoch:02d}-{val_loss:.2f}.hdf5'
#callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
print model2.summary()

### Fit Model ###
#model2.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,verbose=2,\
#        validation_data=test_data,callbacks=[tensorboard,callbacks])
