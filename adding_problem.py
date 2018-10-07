# Adding problem
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
from TCN_code import ResidualBlock,TCN

# Problem size
T = 600
N = 60000

###  load data  ###
data =[]
for i in range(N):
    data += [np.transpose(np.load('../data/adding_problem/data_'+str(i)+'.npy'))]
data = np.array(data)
y = np.sum(data[:,:,0]*data[:,:,1],axis=-1)

frac = 50000
train_data = data[:frac]
train_y = y[:frac]

test_data = data[frac:]
test_y = y[frac:]


###  Setup Parameters  ###
LAYERS    = 8 # n
N_HIDDEN  = 24 # hidden layer, i.e. num of features
INPUT_DIM = (T,2) # seconds in a day, number of channels -1
TIME_STEPS    = 1
KERNEL_SIZE   = 8 # k
layers_effect = 1+2*(KERNEL_SIZE-1)*(2**LAYERS-1)
BATCH_SIZE    = 32
EPOCHS        = 100
STEPS_PER_EPOCH = int(N/BATCH_SIZE)
DROPOUT   = 0.


### Create Model ###
model = TCN(INPUT_DIM,TIME_STEPS,LAYERS,N_HIDDEN,kernel_size=KERNEL_SIZE,\
            dropout=DROPOUT)
adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999,\
                             epsilon=None,decay=0.0, amsgrad=False)
model.load_weights('../data/adding_problem/tcn_weights.h5')
model.compile(loss='mean_squared_error',  metrics=['accuracy'],\
              optimizer=adam)
tensorboard = TensorBoard(log_dir="../data/adding_problem/logs/{}".format(time()),histogram_freq=25,write_images=True)


### Run Model ###
model.fit(train_data,train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
    shuffle=True,validation_data=(test_data,test_y),callbacks=[tensorboard])

model.save_weights('../data/adding_problem/tcn_weights_50k.h5')
