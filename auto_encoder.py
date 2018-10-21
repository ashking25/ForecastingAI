import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import TensorBoard

def dataloader(batch_size=10, nstart=0, num_eq=1000, num_days=30, PATH=''):
    """ Build generator to load the data in chunks """
    while True:
        number_EQ = np.random.randint(nstart, num_eq, num_eq*num_days) # draw a random distribution of events
        number_days = np.random.randint(0, num_days, num_eq*num_days) # draw randomly from 0-30 days before the event
        num_batch = int(num_eq*num_days/batch_size) # number of batches

        for i in range(num_batch):
            start = i*batch_size
            y = []
            data = []

            for j, (EQ, day) in enumerate(zip(number_EQ[start:start+batch_size],
                number_days[start:start+batch_size])):
                data += [np.load(PATH+'/EQ'+str(EQ)+'_'+str(day)+'daysuntilEQ.npy')]
                y += [day]

            yield (data, data)


def auto_encoder(input_dim, features):
    inputs = Input(shape=(None,input_dim))
    dense = Dense(features, activation='relu')(inputs)
    out = Dense(input_dim, activation='linear')(dense)
    model = Model(input=[inputs], output=out)
    return model


if __name__ == "__main__":
    features = 500 # hidden layer, i.e. num of features
    lr = 0.002
    input_dim = (24*3600) # seconds in a day, number of channels -1
    batch_size = 2
    epochs = 250
    steps_per_epoch = 50

    train_gen = dataloader(batch_size=batch_size, num_eq=900,
        PATH='/home/ashking/quake_finder/data/mocks')
    test_gen  = dataloader(batch_size=25, nstart=901, num_eq=1000,
        PATH='/home/ashking/quake_finder/data/mocks') #

    test_data = next(test_gen)
    train_data = next(train_gen)

    model = auto_encoder(input_dim, features)

    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.00, amsgrad=False)

    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)

    print(model.summary())

    tensorboard = TensorBoard(log_dir="../data/mocks/logs/auto_encoder_lr"+str(lr)+\
        "_f"+str(features)+"_sqerr", histogram_freq=10, write_images=True)

    filepath = "../data/mocks/logs/auto_encoder_lr"+str(lr)+\
        "_f"+str(features)+"_sqerr.hdf5"

    callbacks = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
        save_best_only=True, save_weights_only=False, mode='auto', period=10)

    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
        verbose=2, validation_data=test_data, callbacks=[tensorboard,callbacks])
