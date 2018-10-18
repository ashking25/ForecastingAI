#Make hybrid model with TCN and LSTM
from time import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D
from keras.layers import LSTM, Reshape, Flatten, Input
from keras.layers import Add, Dropout, BatchNormalization, Concatenate
from keras.models import *
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal
from keras import backend as K

def binary_lstm_accuracy(y_true, y_pred):
    #want the accuracy just for the lstm output
    shape = K.shape(y_pred)
    return K.mean(K.mean(K.slice(K.equal(y_true, K.round(y_pred)),
        (0,0), (shape[0],1)), axis=-1), axis=-1)

def mean_total_squared_error(y_true,y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true), axis=-1),axis=-1)


def reshapedata(data, timesteps, featuresize):
    """ reshape data into N x timesteps x featuresize"""
    samples = np.size(data)/timesteps/featuresize
    newdata = np.reshape(data[:int(timesteps*featuresize*samples)], (int(samples), int(timesteps), int(featuresize)))
    return newdata

def dataloader_2(timesteps, feature_length, lookback=3, batch_size=10, nstart=0, num_eq=1000, num_days=30, PATH='/Users/ashking/QuakeFinders/data/mocks'):
    """ Build generator to load the data progressively """
    while True:
        number_EQ    = np.random.randint(nstart, num_eq, num_eq*num_days) # draw a random distribution of earth quakes
        number_days  = np.random.randint(0, num_days-lookback, num_eq*num_days) # draw randomly from 0-30 days before the quakes
        num_batch    = int(num_eq*num_days/batch_size) # number of batches

        for i in range(num_batch):
            start    = int(i*batch_size)
            y        = []
            data     = []

            for j, (EQ, day) in enumerate(zip(number_EQ[start:int(start+batch_size)], number_days[start:int(start+batch_size)])):
                newdata = ()
                ynew = [day]
                for s in range(lookback):
                    dataset0 = np.load(PATH+'/EQ'+str(EQ)+'_'+str(day+lookback-1-s)+'daysuntilEQ.npy')
                    if np.size(newdata) <= 0:
                        newdata = reshapedata(dataset0, timesteps, feature_length)
                    else:
                        newdata = np.append(newdata, reshapedata(dataset0, timesteps, feature_length), axis=1)
                    ynew.append(day+lookback-1-s)

                data += [newdata]
                y    += [ynew]

            # This isn't normalized! We should think about this
            data = np.reshape(data, (len(data), int(timesteps*lookback), int(feature_length), 1))
            y    = np.array(y)

            yield (data, y)


def ResidualBlock(inputs, n_outputs, k, d, dropout_rate):
    """ Residual block
    inputs: previous layer
    n_outputs: size of the layers (number of features)
    k: kernel size
    d: dilation rate
    dropout_rate: percentage to dropout

    return
    f: final layer in residual block"""
    input_dim = int(inputs.shape[3])
    c1 = TimeDistributed(Conv1D(n_outputs, kernel_size=k, dilation_rate=d, \
                input_shape=(input_dim, 1), activation='relu', \
               padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(inputs)
    b1 = TimeDistributed(BatchNormalization())(c1)
    d1 = TimeDistributed(Dropout(dropout_rate, noise_shape=(1, 1, n_outputs)))(b1)
    c2 = TimeDistributed(Conv1D(n_outputs, kernel_size=k, dilation_rate=d, \
                activation='relu', padding='same', \
                kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(d1)
    b2 = TimeDistributed(BatchNormalization())(c2)
    d2 = TimeDistributed(Dropout(dropout_rate, noise_shape=(1, 1, n_outputs)))(b2)
    e = TimeDistributed(Dense(n_outputs, activation=None))(inputs)
    f = Add()([e, d2])
    g = TimeDistributed(Dense(n_outputs, activation='relu', name='ResidBlock_'+str(d), \
              kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(f)
    return g


def my_model(input_dim, time_steps, layers, features, n_hidden,
            dilation_rate=1, pooling=1, kernel_size=7, dropout=0.):
    """ LSTM with Temporal Convolution Neural Network """

    num_channels = [n_hidden]*layers
    layers = []
    num_levels = len(num_channels)
    inputs = Input(shape=input_dim)

    for i in range(num_levels):
        dilation_size = int(dilation_rate**2)
        out_channels = num_channels[i]
        if i == 0:
            mod = ResidualBlock(inputs, out_channels, kernel_size, dilation_size, dropout)
        else:
            mod = ResidualBlock(mod, out_channels, kernel_size, dilation_size, dropout)

    cEnd = TimeDistributed(Conv1D(1, kernel_size=1, dilation_rate=1, activation='sigmoid', \
               padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(mod)
    bEnd = TimeDistributed(BatchNormalization())(cEnd)
    resh = Reshape((timesteps*lookback, data_length), \
        input_shape=(timesteps*lookback, data_length, 1))(bEnd)
    mod2 = TimeDistributed(Dense(1, activation='linear', \
        kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(resh) # the last output should be able to reach all of y values

    #lstm1=LSTM(features , return_sequences=True, activation='tanh')(mod2)
        #, input_shape=(timesteps*lookback, int(data_length/pool**nCNN), features)))
    lstm2=LSTM(features, activation='linear')(mod2)
    #mod1 = Flatten()(cEnd)
    #mod2 = Dense(1, activation='linear', kernel_initializer=RandomNormal(mean=0, stddev=0.01))(lstm2) # the last output should be able to reach all of y values
    resh2 = Flatten()(mod2)
    mod_end = Concatenate()([lstm2,resh2])
    model = Model(input=[inputs], output=mod_end)
    return model

### Data params ###
batch_size = 2
epochs = 200
steps_per_epoch = 50# int(900/BATCH_SIZE)#*30
timesteps = 1
data_length = int(3600*24/timesteps)
lookback = 3
input_dim = (int(timesteps*lookback), int(data_length), 1)
dropout = 0
features = 1 # number of features in lstm
n_hidden = 50 # number of featurs in TCN
kernel_size  = 7
dilation_rate = 2
layers = int(np.ceil(np.log((input_dim[1]-1.)/(2.*(kernel_size-1))+1)/np.log(dilation_rate)))

### Data ###
train_gen = dataloader_2(timesteps, data_length, lookback=lookback, batch_size=batch_size,
    num_eq=900, PATH='../data/mocks')
test_gen  = dataloader_2(timesteps, data_length, lookback=lookback, batch_size=25,
    nstart=901, num_eq=1000, PATH='/home/ashking/quake_finder/data/mocks') #
test_data = next(test_gen)
train_data = next(train_gen)


### Model ###
model2 = my_model(input_dim, timesteps, layers, features, n_hidden,
        dilation_rate=dilation_rate, kernel_size=kernel_size, dropout=dropout)

#Optimizer
adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
    decay=0.005, amsgrad=False)
model2.compile(loss=mean_total_squared_error,  metrics=[binary_lstm_accuracy,
    'accuracy'], optimizer=adam)

print(model2.summary())
print('layers', layers)

tensorboard = TensorBoard(log_dir="../data/mocks/logs/hybrid_l"+str(layers)+\
    "_k"+str(kernel_size)+"_nh"+str(n_hidden)+"_d"+str(dilation_rate)+"_f"+str(features)+"_sqerr", \
    histogram_freq=0, write_images=True)

filepath = "../data/mocks/logs/model_hybrid_l"+str(layers)+\
    "_k"+str(kernel_size)+"_nh"+str(n_hidden)+"_d"+str(dilation_rate)+"_f"+str(features)+"_sqerr.hdf5"

callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
    verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)

model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2, \
        validation_data=test_data, callbacks=[tensorboard, callbacks])
