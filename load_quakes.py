import numpy as np
import keras
from keras.callbacks import TensorBoard
#from hybrid import my_model
import sys
import os
from astropy.time import Time, TimeDelta
import pandas as pd
from hybrid import my_model

def find_lookback(PATH,day,lookback):
    """ determine if all the days are present in the file"""
    path_n = np.copy(PATH)
    str(path_n).replace('d_mag','n_mag')
    path_e = np.copy(PATH)
    str(path_e).replace('d_mag','e_mag')
    times = Time(day[:4]+'-'+day[4:6]+'-'+day[6:8])
    t = Time(times)
    for l in range(lookback):
        delta_t = TimeDelta(l,format='jd')
        previous_day = (t-delta_t).value
        previous_day = previous_day[:4]+previous_day[5:7]+previous_day[8:10]
        if not os.path.isfile(PATH+'/'+previous_day+'.npy'):
            return False
        elif not os.path.isfile(str(path_n)+'/'+previous_day+'.npy'):
            return False
        elif not os.path.isfile(str(path_e)+'/'+previous_day+'.npy'):
            return False
    return True

def make_data_list(PATH):
    f = open(PATH+'list_of_data_lookback5.txt','w')
    #### ls d_mag*npy > list_of_data.txt
    with open(PATH+'list_of_data.txt','r') as fp:
        line = fp.readline()
        while line:
            path = line[:-14]
            day = line[-13:-5]
            lookback = 5
            path_n = np.copy(path)
            str(path_n).replace('d_mag','n_mag')
            path_e = np.copy(path)
            str(path_e).replace('d_mag','e_mag')
            if find_lookback(PATH+path,day,lookback):
                if find_lookback(PATH+str(path_n),day,lookback):
                    if find_lookback(PATH+str(path_e),day,lookback):
                        f.write(line)
            line = fp.readline()
    f.close()

def load_data(PATH,detector,year,day,lookback,freq=5):
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
            derivative = np.transpose(derivative)
            reshaped = np.reshape(derivative,(freq,len(derivative)/freq))
            values.append(reshaped)
        values = np.array(values)
        values = np.reshape(values,(1,values.shape[1],values.shape[2],values.shape[0]))
        if len(all_values)==0:
            all_values = values
        else:
            all_values = np.append(all_values,values,axis=1)
    return all_values


def set_eq_value(detector, day, lookback, Magnitude=4.8, EQEs=0, threshold_days = 0, max_value=360,
    PATH='/Users/ashking/QuakeFinders/data/'):
    #thresold_days is the number of days after an earth quake to consider, <0 is after earthquake

    df = pd.read_csv(PATH+'Earthquakes_previous_year.csv')
    df2 = pd.read_csv(PATH+'Earthquakes.csv')
    df = df.append(df2)
    times = [Time(str.replace(df['EQ Datetime'].values[i],' ','T')) for i in range(len(df))]
    t = Time(times)
    df['MJD'] = t.mjd
    df['delta_t']=df['MJD']-day.mjd
    df_local = df.loc[(df['Station ID']==int(detector))
        & (df['EQ Es'] >= EQEs) & (df['EQ Magnitude'] >= Magnitude) &
        (df['delta_t'] >= threshold_days)]
    if len(df_local) > 0:
        if np.min(df_local['delta_t'].values)< max_value:
            return np.min(df_local['delta_t'].values)
        else:
            return max_value
    else:
        return max_value


def dataloader(freq=5, lookback=3, batch_size=10, train_percent=0.8, test=False,
    PATH='/Users/ashking/QuakeFinders/data/'):

    """ Build generator to load the data progressively """

    files = np.loadtxt(PATH+'list_of_data_lookback5.txt',dtype='str')
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
            year = r[10:17]
            detector = r[-17:-13]
            freq = 5
            lookback = 5
            if len(data) == 0:
                data = load_data(PATH+'qf/level2',detector, year, day, lookback, freq=freq)
            else:
                data = np.append(data,load_data(PATH+'qf/level2', detector, year, day,\
                lookback, freq=freq), axis=0)

            y += [set_eq_value(detector, t, lookback, PATH=PATH)]
        yield (data, np.array(y))


if __name__ == "__main__":
    batch_size = 2
    epochs = 200
    steps_per_epoch = 10# int(900/BATCH_SIZE)#*30
    timesteps = 5
    freq = 50
    data_length = int(3600*24*freq/timesteps)
    lookback = 5
    #input_dim = (None, int(data_length), 1)
    input_dim = (int(lookback*timesteps), int(data_length), 3)
    dropout = 0
    features = 1 # number of features in lstm
    n_hidden = 64 # number of featurs in TCN
    kernel_size  = 7
    dilation_rate = 4
    layers = int(np.ceil(np.log((input_dim[1]-1.)/(2.*(kernel_size-1))+1)/np.log(dilation_rate)))
    #Optimizer
    lr = 0.003
    ### Model ###

    model2 = my_model(input_dim, timesteps, lookback, layers, features, n_hidden,
            dilation_rate=dilation_rate, kernel_size=kernel_size, dropout=dropout)


    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.01, amsgrad=False)
    model2.compile(loss='mse', metrics=['accuracy'], optimizer=adam)

    print (model2.summary())
    print('layers', layers)


    ### Data ###
    print 'train'
    train_gen = dataloader(lookback=lookback, batch_size=batch_size, test=False)
        #PATH='../data/mocks')
    print 'test'
    test_gen  = dataloader(lookback=lookback, batch_size=10, test=True)#,
        #nstart=901, num_eq=1000, PATH='/home/ashking/quake_finder/data/mocks') #
    test_data = next(test_gen)
    train_data = next(train_gen)

    ### Fit ###
    filepath = "../data/mocks/logs/model_hybrid_denseoutput_dense_look"+str(lookback)+"_l"+str(layers)+\
        "_k"+str(kernel_size)+"_nh"+str(n_hidden)+"_d"+str(dilation_rate)+"_f"+\
            str(features)+"_lr"+str(lr)+"_sqerr.hdf5"

    callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
        verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)



    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=2, verbose=2, \
            validation_data=test_data)#, callbacks=[callbacks])
