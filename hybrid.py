#Make hybrid model with TCN and LSTM
from time import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D
from keras.layers import LSTM, Reshape, Flatten, Input, GlobalMaxPooling1D
from keras.layers import Add, Dropout, BatchNormalization, Concatenate
from keras.models import *
from keras.callbacks import TensorBoard
from keras.initializers import RandomNormal
from keras import backend as K
from keras import regularizers


def custom_activation(x):
    return (K.sigmoid(x)*30.)-1

def binary_lstm_accuracy(y_true, y_pred):
    #want the accuracy just for the lstm output
    shape = K.shape(y_pred)
    return K.mean(K.mean(K.slice(K.equal(y_true, K.round(y_pred)),
        (0,0), (shape[0],1)), axis=-1), axis=-1)

def binary_tcn_accuracy(y_true, y_pred):
    #want the accuracy just for the lstm output
    shape = K.shape(y_pred)
    return K.mean(K.mean(K.slice(K.equal(y_true, K.round(y_pred)),
        (0,shape[1]-1), (shape[0],1)), axis=-1), axis=-1)

def mean_total_squared_error(y_true,y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true), axis=-1),axis=-1)

def mean_tcn_squared_error(y_true,y_pred):
    shape = K.shape(y_pred)
    return K.mean(K.sum(K.slice(K.square(y_pred - y_true),
        (0,shape[1]-1), (shape[0],1)), axis=-1),axis=-1)


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
                    ynew.append((day+lookback-1-s))#(day+lookback-1-s)/float(num_days)) # normalize from 0-1

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
    f = Add()([e, d1])
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
    mod2 = TimeDistributed(Dense(1, activation='relu', \
        kernel_initializer=RandomNormal(mean=0, stddev=0.01)))(resh) # the last output should be able to reach all of y values

    glob_pool = TimeDistributed(GlobalMaxPooling1D())(mod)
    #dense1 = Dense(128, activation='relu')(resh)

    #lstm1=LSTM(features , return_sequences=True, activation='tanh')(dense1)
    #lstm2=LSTM(1, activation='relu')(glob_pool)
    mod1 = Flatten()(glob_pool)
    mod3 = Dense(1, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.01))(mod1) # the last output should be able to reach all of y values

    resh2 = Flatten()(mod2)
    mod_end = Concatenate()([mod3,resh2])
    model = Model(input=[inputs], output=mod_end)
    return model

### Data params ###
batch_size = 2
epochs = 200
steps_per_epoch = 50# int(900/BATCH_SIZE)#*30
timesteps = 1
data_length = int(3600*24/timesteps)
lookback = 5
#input_dim = (None, int(data_length), 1)
input_dim = (int(lookback*timesteps), int(data_length), 1)
dropout = 0
features = 1 # number of features in lstm
n_hidden = 64 # number of featurs in TCN
kernel_size  = 7
dilation_rate = 4
layers = int(np.ceil(np.log((input_dim[1]-1.)/(2.*(kernel_size-1))+1)/np.log(dilation_rate)))
#Optimizer
lr = 0.003
### Model ###
if True:

    model2 = my_model(input_dim, timesteps, layers, features, n_hidden,
            dilation_rate=dilation_rate, kernel_size=kernel_size, dropout=dropout)


    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.01, amsgrad=False)
    model2.compile(loss=mean_total_squared_error,
        metrics=[binary_lstm_accuracy, binary_tcn_accuracy, mean_tcn_squared_error,
        'accuracy'], optimizer=adam)
else:
    model2= load_model('../data/mocks/logs/model_hybrid_look1_l7_k7_nh64_d4_f1_lr0.003_sqerr.hdf5',
        custom_objects={'binary_lstm_accuracy':binary_lstm_accuracy,'binary_tcn_accuracy':binary_tcn_accuracy,\
        'mean_total_squared_error':mean_total_squared_error,'mean_tcn_squared_error':mean_tcn_squared_error})

print(model2.summary())
print('layers', layers)

### Data ###
train_gen = dataloader_2(timesteps, data_length, lookback=lookback, batch_size=batch_size,
    num_eq=900, PATH='../data/mocks')
test_gen  = dataloader_2(timesteps, data_length, lookback=lookback, batch_size=25,
    nstart=901, num_eq=1000, PATH='/home/ashking/quake_finder/data/mocks') #
test_data = next(test_gen)
train_data = next(train_gen)


#tensorboard = TensorBoard(log_dir="../data/mocks/logs/hybrid_l"+str(layers)+\
#    "_k"+str(kernel_size)+"_nh"+str(n_hidden)+"_d"+str(dilation_rate)+"_f"+
#    str(features)+"_lr"+str(lr)+"_sqerr", \
#    histogram_freq=0, write_images=True)

filepath = "../data/mocks/logs/model_hybrid_dense_look"+str(lookback)+"_l"+str(layers)+\
    "_k"+str(kernel_size)+"_nh"+str(n_hidden)+"_d"+str(dilation_rate)+"_f"+\
        str(features)+"_lr"+str(lr)+"_sqerr.hdf5"

callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
    verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)



#model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=20, verbose=2, \
#        validation_data=test_data, callbacks=[callbacks])


K.set_value(model2.optimizer.lr, 2e-3)
model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
        verbose=2, validation_data=test_data, callbacks=[callbacks])
