### run tcn code with encoder layers
from TCN_code import ResidualBlock
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D
from keras.layers import Flatten, Input, BatchNormalization, Reshape
from keras.models import load_model
from keras.callbacks import Callback
from keras.initializers import RandomNormal
from keras import backend as K
import numpy as np
from keras import regularizers


def auto_conv_encoder_only(model1, inputs, features, kernel, pool=2):
    # encoder
    conv1 = Conv2D(features, kernel,  padding='same', weights=model1.layers[1].get_weights(), trainable=False)(inputs)
    pool1 = MaxPooling2D((pool,1), weights=model1.layers[2].get_weights(), trainable=False)(conv1)
    conv2 = Conv2D(features*2, kernel, padding='same', weights=model1.layers[3].get_weights(), trainable=False)(pool1)
    pool2 = MaxPooling2D((pool,1), weights=model1.layers[4].get_weights(), trainable=False)(conv2)
    conv3 = Conv2D(features*4, kernel, padding='same',weights=model1.layers[5].get_weights(), trainable=False)(pool2)
    pool3 = MaxPooling2D((pool,1), weights=model1.layers[6].get_weights(), trainable=False)(conv3)
    conv4 = Conv2D(features*8, kernel, padding='same', weights=model1.layers[7].get_weights(), trainable=False)(pool3)
    pool4 = MaxPooling2D((pool,1), weights=model1.layers[8].get_weights(), trainable=False)(conv4)

    return pool4



def TCN(input_dim, time_steps, layers, features, features_enc, kernel_enc,
    model1, dilation_rate=2., kernel_size=3, dropout=0.):
    """ Temporal Convolution Neural Network
        By using dilaiton with enough layers, all values should be causally
        connnected"""

    num_channels = [features]*layers
    layers = []
    num_levels = len(num_channels)
    inputs = Input(shape=input_dim)

    encode = auto_conv_encoder_only(model1, inputs, features_enc, kernel_enc)
    encoder_shape = K.int_shape(encode)
    reshape = Reshape((encoder_shape[1], encoder_shape[-1]))(encode)

    for i in range(num_levels):
        dilation_size = int(dilation_rate ** i)

        out_channels = num_channels[i]*2**(i//2)
        if i == 0:
            mod = ResidualBlock(reshape, out_channels, kernel_size, dilation_size, dropout)
        else:
            mod = ResidualBlock(mod, out_channels, kernel_size, dilation_size, dropout)
    cEnd = Conv1D(1, kernel_size=1, dilation_rate=1, activation='sigmoid',\
               padding='same', kernel_initializer=RandomNormal(mean=0, stddev=0.01),
               activity_regularizer=regularizers.l1(1e-7))(mod)
    bcEnd = BatchNormalization()(cEnd)
    mod1 = Flatten()(bcEnd)
    mod2 = Dense(1,activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.01),
            bias_initializer=keras.initializers.Constant(value=14.),
            max_value=40, threshold=-10.0)(mod1) # the last output should be able to reach all of y values
    model = Model(input=[inputs], output=mod2)
    return model


def dataloader(batch_size=10, nstart=0, num_eq=1000, num_days=30, PATH='', weights=False):
    """ Build generator to load the data in chunks """
    while True:
        number_EQ = np.random.randint(nstart, num_eq, num_eq*num_days) # draw a random distribution of events
        number_days = np.random.randint(0, num_days, num_eq*num_days) # draw randomly from 0-30 days before the event
        num_batch = int(num_eq*num_days/batch_size) # number of batches

        for i in range(num_batch):
            start = i*batch_size
            y = []
            data = []
            sample_weights = []

            for j, (EQ, day) in enumerate(zip(number_EQ[start:start+batch_size],
                number_days[start:start+batch_size])):
                data += [np.load(PATH+'/EQ'+str(EQ)+'_'+str(day)+'daysuntilEQ.npy')]
                y += [day]
                sample_weights+=[30-day]

            data = np.array(data)
            y = np.array(y)
            sample_weights = np.array(sample_weights)
            data0 = np.reshape(data,(len(data),data.shape[1],1,1))
            if weights:
                yield (data0, y, sample_weights)
            else:
                yield (data0, y)

if __name__ == "__main__":
    kernel_enc = (7,1)
    features_enc = 16 # hidden layer, i.e. num of features
    lr = 0.00003
    input_dim_enc = (24*3600,1,1) # seconds in a day, number of channels -1
    batch_size = 2
    epochs = 500
    steps_per_epoch = 100

    n_hidden  = 64 # hidden layer, i.e. num of features
    input_dim = (5400,128) # seconds in a day, number of channels -1
    time_steps = 1
    kernel = 15
    dilation = 2.
    layers = int(np.ceil(np.log(((input_dim)[0]-1.)/(2.*(kernel-1))+1)/np.log(dilation)))

    model1 = load_model('../data/mocks/logs/auto_conv_encoder_regul_lr3e-05_f16_k7_sqerr.hdf5')

    train_gen = dataloader(batch_size=batch_size, num_eq=900,
        PATH='/home/ashking/quake_finder/data/mocks')
    test_gen  = dataloader(batch_size=25, nstart=901, num_eq=1000,
        PATH='/home/ashking/quake_finder/data/mocks') #

    test_data = next(test_gen)
    train_data = next(train_gen)


    model2 = TCN(input_dim_enc, time_steps, layers, n_hidden, features_enc, kernel_enc,
        model1, dilation_rate=dilation, kernel_size=kernel, dropout=0.)

    adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.01, amsgrad=False)

    model2.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)

    print(model2.summary())
    print('layers',layers)


    filepath = "../data/mocks/logs/auto_tcn_encoder_sample_weights_lr"+str(lr)+\
        "_f"+str(n_hidden)+"_k"+str(kernel)+"_l"+str(layers)+"_sqerr.hdf5"

    callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
        save_best_only=True, save_weights_only=False, mode='auto', period=10)

    #lr_tracker = showLR()
    # cycle through, i think starting again is good for some reason
    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=20,
            verbose=2, validation_data=test_data, callbacks=[callbacks])

    K.set_value(model2.optimizer.lr, 1e-7)
    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
            verbose=2, validation_data=test_data, callbacks=[callbacks])
