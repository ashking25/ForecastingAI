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



N_HIDDEN  = 50 # hidden layer, i.e. num of features
INPUT_DIM = (24*3600,1) # seconds in a day, number of channels -1
TIME_STEPS    = 1
KERNEL_SIZE   = 15
#layers_effect = 1+2*(KERNEL_SIZE-1)*(2**LAYERS-1)
DILATION      = 4.
LAYERS        = int(np.ceil(np.log((INPUT_DIM[0] -1.)/(2.*(KERNEL_SIZE-1))+1)/np.log(DILATION)))
BATCH_SIZE    = 2
EPOCHS        = 250
STEPS_PER_EPOCH =50# int(900/BATCH_SIZE)#*30

train_gen = dataloader(batch_size=BATCH_SIZE,num_eq=900,PATH='/home/ashking/quake_finder/data/mocks')
test_gen  = dataloader(batch_size=25,nstart=901,num_eq=1000,PATH='/home/ashking/quake_finder/data/mocks') #
test_data = next(test_gen)
train_data = next(train_gen)


#print 'the number of effected points',layers_effect
 
model = TCN(INPUT_DIM,TIME_STEPS,LAYERS,N_HIDDEN,dilation_rate=DILATION,kernel_size=KERNEL_SIZE,dropout=0.)
#adam = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None,
#    decay=0.0, amsgrad=False)

#Compile model
# run either this line or the following line, not both
#for i in range(2):
#model = load_model("../data/mocks/logs/model_v4_TCN_l"+str(LAYERS)+"_k"+str(KERNEL_SIZE)+"_nh"+str(N_HIDDEN)+"_d"+str(DILATION)+"_sqerr.hdf5")

adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
    decay=0.01, amsgrad=False)
model.compile(loss='mean_squared_error',  metrics=['accuracy'],optimizer=adam)
print(model.summary())
print('layers',LAYERS)
tensorboard = TensorBoard(log_dir="../data/mocks/logs/TCN_l"+str(LAYERS)+"_k"+str(KERNEL_SIZE)+"_nh"+str(N_HIDDEN)+"_d"+str(DILATION)+"_sqerr",histogram_freq=0,write_images=True)
filepath = "../data/mocks/logs/model_v4_TCN_l"+str(LAYERS)+"_k"+str(KERNEL_SIZE)+"_nh"+str(N_HIDDEN)+"_d"+str(DILATION)+"_sqerr.hdf5"
callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)

model.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,verbose=2,\
        validation_data=test_data,callbacks=[tensorboard,callbacks])
#model.save('../data/mocks/tcn_model_nhid100_'+str(4+i)+'.h5')
#model.save_weights('../data/mocks/tcn_weights_nhid100_3.h5')

