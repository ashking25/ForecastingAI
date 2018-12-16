import numpy as np
import keras
from keras.callbacks import TensorBoard
import sys
import os
from astropy.time import Time, TimeDelta
import pandas as pd
from scipy.signal import decimate
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D, Flatten, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.callbacks import Callback
from keras import regularizers



def load_data(PATH,detector,year,day,lookback,freq=1):
    times = Time(day[:4]+'-'+day[4:6]+'-'+day[6:8])
    t = Time(times)
    all_values =[]
    for l in range(lookback):
        delta_t = TimeDelta(lookback-l-1,format='jd')
        previous_day = (t-delta_t).value
        previous_day = previous_day[:4]+previous_day[5:7]+previous_day[8:10]
        values=[]
        for letter in ['d','e','n']:
            mag = np.load(PATH+'/'+year+'/'+letter+'_mag_volts_rs50/'+detector+'/'+previous_day+'.npy')
            derivative = np.append(mag[:-1]-mag[1:],0)
            #derivative = derivative[:int(len(derivative)/2.)]
            # down sample
            #derivative = decimate(derivative, 10) # downsample by a factor of 10
            #derivative = decimate(derivative, 5) # twice if you want to downsample over a factor of 13
            derivative = np.transpose(derivative)
            reshaped = np.reshape(derivative,(freq,int(len(derivative)/freq)))
            values.append(reshaped)
        values = np.array(values)
        values = np.reshape(values, (values.shape[1], values.shape[2], 1, values.shape[0]))
        if len(all_values)==0:
            all_values = values
        else:
            all_values = np.append(all_values,values,axis=-1)
    return all_values


def dataloader(freq=5, lookback=1, batch_size=10, train_percent=0.8, test=False,
    file='list_of_data_lookback5.txt',PATH='/home/ashking/quake_finder/data_qf/'):

    """ Build generator to load the data progressively """
    while True:
        files = np.loadtxt(PATH[:-8]+'ForecastingAI/'+file,dtype='str')
        if test:
            files = files[int(train_percent*len(files)):]
        else:
            files = files[:int(train_percent*len(files))]
        random_files = np.random.choice(files,size=len(files),replace=False)
        for i in range(int(len(random_files)/batch_size)):
            data = []
            y = []
            for r in random_files[i*batch_size:(i+1)*batch_size]:
                day = r[-12:-4]
                times = Time(day[:4]+'-'+day[4:6]+'-'+day[6:8])
                t = Time(times)
                year = r[7:14]
                detector = r[-17:-13]
                if len(data) == 0:
                    data = load_data(PATH+'level2',detector, year, day, lookback, freq=freq)
                else:
                    data = np.append(data,load_data(PATH+'level2', detector, year, day,\
                    lookback, freq=freq), axis=0)
            y = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[-1]))
            #y = data[:,:,0,0]
            for j in range(int(data.shape[0]/batch_size)):
                #print('shape',np.shape(data))
                #print('shape y',np.shape(y))
                yield (data[j*batch_size:(j+1)*batch_size], y[j*batch_size:(j+1)*batch_size])

def auto_conv_encoder(input_dim, features, kernel, pool=2):
    inputs = Input(shape=input_dim)

    # encoder
    conv1 = Conv2D(features, kernel, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((pool, 1), padding='same')(conv1)
    conv2 = Conv2D(features*2, kernel, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((pool, 1), padding='same')(conv2)
    conv3 = Conv2D(features*4, kernel, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((pool, 1), padding='same')(conv3)
    conv4 = Conv2D(features*8, kernel, activation='relu', padding='same',
                   activity_regularizer=regularizers.l1(1e-7))(pool3)
    pool4 = MaxPooling2D((pool,1), padding='same')(conv4)
    #conv5 = Conv2D(1, kernel, activation='relu', padding='same',
    #               activity_regularizer=regularizers.l1(1e-7))(pool4)
    # decoder
    conv9 = Conv2D(features*8, kernel, activation='relu', padding='same')(pool4)
    pool9 = UpSampling2D((pool, 1))(conv9)
    conv10 = Conv2D(features*4, kernel, activation='relu', padding='same')(pool9)
    pool10 = UpSampling2D((pool, 1))(conv10)
    conv11 = Conv2D(features*2, kernel, activation='relu', padding='same')(pool10)
    pool11 = UpSampling2D((pool, 1))(conv11)
    conv12 = Conv2D(features, kernel, activation='relu', padding='same')(pool11)
    pool12 = UpSampling2D((pool, 1))(conv12)
    conv13 = Conv2D(3, kernel, activation='linear', padding='same')(pool12)

    flat = Flatten()(conv13)

    model = Model(input=[inputs], output=flat)

    return model
    
if __name__ == "__main__":
    kernel = (7,1)
    features = 32 # hidden layer, i.e. num of features
    lr = 1e-5
    freq = 50
    input_dim = (24*3600*50./freq, 1, 3) # seconds in a day, number of channels -1
    batch_size = 2
    fit_batch_size = batch_size*freq
    epochs = 1000
    lookback = 1
    steps_per_epoch = 100

    train_gen = dataloader(freq=freq, lookback=lookback, batch_size=batch_size, train_percent=.05,
                           test=False, file='list_of_data_lookback5.txt')
        #PATH='../data/mocks')
    print('test')
    test_gen  = dataloader(freq=freq, lookback=lookback, batch_size=4, train_percent=0.95,
            test=False, file='list_of_validate_data_lookback5.txt')
    test_data = next(test_gen)
    train_data = next(train_gen)

    #print('test shape',np.shape(test_data[0]))
    model2 = auto_conv_encoder(input_dim, features, kernel)
    #model2 = load_model('../data/mocks/logs/auto_conv_encoder_lr3e-05_f16_k7_sqerr.hdf5')

    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.01)

    model2.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)#, sample_weight_mode="temporal")

    print(model2.summary())

    #tensorboard = TensorBoard(log_dir="../data/mocks/logs/auto_conv_encoder_lr"+str(lr)+\
    #    "_f"+str(features)+"_k"+str(kernel[0])+"_sqerr", histogram_freq=0, write_images=False)

    filepath = "../data/mocks/logs/auto_conv_encoder_regul_sample_weights_lr"+str(lr)+\
        "_f"+str(features)+"_k"+str(kernel[0])+"_sqerr.hdf5"

    callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
        save_best_only=True, save_weights_only=False, mode='auto', period=10)


    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=20,
                verbose=1, validation_data=test_data, callbacks=[callbacks])

    #K.set_value(model2.optimizer.lr, lr)
    #model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
    #        verbose=2, validation_data=test_data, callbacks=[callbacks])
