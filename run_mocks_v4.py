### run TCN on quake mocks

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
from TCN_code import ResidualBlock,TCN,dataloader

LAYERS    = 14
N_HIDDEN  = 100 # hidden layer, i.e. num of features
INPUT_DIM = (24*3600,1) # seconds in a day, number of channels -1
TIME_STEPS    = 1
KERNEL_SIZE   = 3
layers_effect = 1+2*(KERNEL_SIZE-1)*(2**LAYERS-1)
BATCH_SIZE    = 3
EPOCHS        = 200
STEPS_PER_EPOCH =50# int(900/BATCH_SIZE)#*30

train_gen = dataloader(batch_size=BATCH_SIZE,num_eq=900,PATH='/home/ashking/quake_finder/data/mocks')
test_gen  = dataloader(batch_size=25,nstart=901,num_eq=1000,PATH='/home/ashking/quake_finder/data/mocks') #
test_data = next(test_gen)
train_data = next(train_gen)


#print 'the number of effected points',layers_effect
 
model = TCN(INPUT_DIM,TIME_STEPS,LAYERS,N_HIDDEN,kernel_size=KERNEL_SIZE,dropout=0.)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
    decay=0.01, amsgrad=False)

#Compile model
# run either this line or the following line, not both
#for i in range(2):
    #model = load_model('../data/mocks/tcn_model_nhid100_'+str(3+i)+'.h5')
model.compile(loss='mean_squared_error',  metrics=['accuracy'],optimizer=adam)
print(model.summary())
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
filepath = 'logs/model_v4_{epoch:02d}-{val_loss:.2f}.hdf5'
callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)

model.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,verbose=2,\
        validation_data=test_data,callbacks=[tensorboard,callbacks])
#model.save('../data/mocks/tcn_model_nhid100_'+str(4+i)+'.h5')
#model.save_weights('../data/mocks/tcn_weights_nhid100_3.h5')

