import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D, Flatten, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.callbacks import Callback
from keras import backend as K
from keras import regularizers
class showLR( Callback ) :
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print(" epoch={:02d}, lr={:.5f}".format( epoch, lr ))


def dataloader(batch_size=10, nstart=0, num_eq=1000, num_days=30, PATH='', conv=False, weights=False, yvals='series'):
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
                sample_weights += [30-day]


            data = np.array(data)
            if yvals == 'days':
                y = np.array(y)
            elif yvals == 'series':
                y = np.copy(data)
                y[np.where(np.abs(y) < 2.5)] = 0
            sample_weights = np.array(sample_weights)
            if conv:
                data0 = np.reshape(data,(len(data),data.shape[1],1,1))
                if weights:
                    yield (data0, y, sample_weights)
                else:
                    yield (data0, y)
            else:
                if weights:
                    yield (data, y, sample_weights)
                else:
                    yield (data, y)


def auto_encoder(input_dim, features):
    inputs = Input(shape=(input_dim,))
    dense = Dense(features, activation='relu')(inputs)
    out = Dense(input_dim, activation='linear')(dense)
    model = Model(input=[inputs], output=out)
    return model


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

    # decoder
    conv9 = Conv2D(features*8, kernel, activation='relu', padding='same')(pool4)
    pool9 = UpSampling2D((pool, 1))(conv9)
    conv10 = Conv2D(features*4, kernel, activation='relu', padding='same')(pool9)
    pool10 = UpSampling2D((pool, 1))(conv10)
    conv11 = Conv2D(features*2, kernel, activation='relu', padding='same')(pool10)
    pool11 = UpSampling2D((pool, 1))(conv11)
    conv12 = Conv2D(features, kernel, activation='relu', padding='same')(pool11)
    pool12 = UpSampling2D((pool, 1))(conv12)
    conv13 = Conv2D(1, kernel, activation='linear', padding='same')(pool12)

    flat = Flatten()(conv13)

    model = Model(input=[inputs], output=flat)

    return model

if __name__ == "__main__":
    kernel = (7,1)
    features = 16 # hidden layer, i.e. num of features
    lr = 0.00003
    input_dim = (24*3600,1,1) # seconds in a day, number of channels -1
    batch_size = 2
    epochs = 1000
    steps_per_epoch = 100

    train_gen = dataloader(batch_size=batch_size, num_eq=900,
        PATH='/home/ashking/quake_finder/data/mocks',conv=True, weights= True)
    test_gen  = dataloader(batch_size=25, nstart=901, num_eq=1000,
        PATH='/home/ashking/quake_finder/data/mocks',conv=True, weights=True) #

    test_data = next(test_gen)
    train_data = next(train_gen)

    model2 = auto_conv_encoder(input_dim, features, kernel)

    #model2 = load_model('../data/mocks/logs/auto_conv_encoder_lr3e-05_f16_k7_sqerr.hdf5')

    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.00)

    model2.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)#, sample_weight_mode="temporal")

    print(model2.summary())

    #tensorboard = TensorBoard(log_dir="../data/mocks/logs/auto_conv_encoder_lr"+str(lr)+\
    #    "_f"+str(features)+"_k"+str(kernel[0])+"_sqerr", histogram_freq=0, write_images=False)

    filepath = "../data/mocks/logs/auto_conv_encoder_regul_sample_weights_lr"+str(lr)+\
        "_f"+str(features)+"_k"+str(kernel[0])+"_sqerr.hdf5"

    callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
        save_best_only=True, save_weights_only=False, mode='auto', period=10)


    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=20,
                verbose=2, validation_data=test_data, callbacks=[callbacks])

    K.set_value(model2.optimizer.lr, lr)
    model2.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
            verbose=2, validation_data=test_data, callbacks=[callbacks])
