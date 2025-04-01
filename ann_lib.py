import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D, MaxPooling2D,
                                     Concatenate, concatenate, Activation, Conv2DTranspose,
                                     LeakyReLU, Reshape, Embedding, BatchNormalization, ReLU, Dense)
from tensorflow import pad, constant
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import numpy as np
import keras

def generate_unet(N_filters, learning_rate, input_shape, filter_size,
                  activation, final_activation, loss_function):
    '''
    Initialize the U-Net neural network architecture
    '''
    paddings = tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]])


    inputs = Input(input_shape)

    # Encoder
    Epad1 = tf.pad(inputs, paddings, "SYMMETRIC")
    Ec1 = Conv2D(N_filters[0], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad1)
    EA1 = Activation(activation)(Ec1)
    Epad2 = tf.pad(EA1, paddings, "SYMMETRIC")
    Ec2 = Conv2D(N_filters[0], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad2)
    EA2 = Activation(activation)(Ec2)

    Ep1 = MaxPooling2D((2, 2))(EA2)
    Epad3 = tf.pad(Ep1, paddings, "SYMMETRIC")
    Ec3 = Conv2D(N_filters[1], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad3)
    EA3 = Activation(activation)(Ec3)
    Epad4 = tf.pad(EA3, paddings, "SYMMETRIC")
    Ec4 = Conv2D(N_filters[1], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad4)
    EA4 = Activation(activation)(Ec4)

    Ep2 = MaxPooling2D((2, 2))(EA4)
    Epad5 = tf.pad(Ep2, paddings, "SYMMETRIC")
    Ec5 = Conv2D(N_filters[2], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad5)
    EA5 = Activation(activation)(Ec5)
    Epad6 = tf.pad(EA5, paddings, "SYMMETRIC")
    Ec6 = Conv2D(N_filters[2], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad6)
    EA6 = Activation(activation)(Ec6)

    Ep3 = MaxPooling2D((2, 2))(EA6)
    Epad7 = tf.pad(Ep3, paddings, "SYMMETRIC")
    Ec7 = Conv2D(N_filters[3], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad7)
    EA7 = Activation(activation)(Ec7)
    Epad8 = tf.pad(EA7, paddings, "SYMMETRIC")
    Ec8 = Conv2D(N_filters[3], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad8)
    EA8 = Activation(activation)(Ec8)

    Ep4 = MaxPooling2D((2, 2))(EA8)
    Epad9 = tf.pad(Ep4, paddings, "SYMMETRIC")
    Ec9 = Conv2D(N_filters[4], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad9)
    EA9 = Activation(activation)(Ec9)
    Epad10 = tf.pad(EA9, paddings, "SYMMETRIC")
    Ec10 = Conv2D(N_filters[4], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Epad10)
    EA10 = Activation(activation)(Ec10)

    # Decoder
    Du1 = UpSampling2D((2, 2), interpolation='bilinear')(EA10)
    DpadI1 = tf.pad(Du1, paddings, "SYMMETRIC")
    DcI1 = Conv2D(N_filters[5], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(DpadI1)
    DAI1 = Activation(activation)(DcI1)
    Dcc1 = concatenate([DAI1, EA8])
    Dpad1 = tf.pad(Dcc1, paddings, "SYMMETRIC")
    Dc1 = Conv2D(N_filters[5], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad1)
    DA1 = Activation(activation)(Dc1)
    Dpad2 = tf.pad(DA1, paddings, "SYMMETRIC")
    Dc2 = Conv2D(N_filters[5], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad2)
    DA2 = Activation(activation)(Dc2)

    Du2 = UpSampling2D((2, 2), interpolation='bilinear')(DA2)
    DpadI2 = tf.pad(Du2, paddings, "SYMMETRIC")
    DcI2 = Conv2D(N_filters[6], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(DpadI2)
    DAI2 = Activation(activation)(DcI2)
    Dcc2 = concatenate([DAI2, EA6])
    Dpad2 = tf.pad(Dcc2, paddings, "SYMMETRIC")
    Dc3 = Conv2D(N_filters[6], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad2)
    DA3 = Activation(activation)(Dc3)
    Dpad3 = tf.pad(DA3, paddings, "SYMMETRIC")
    Dc4 = Conv2D(N_filters[6], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad3)
    DA4 = Activation(activation)(Dc4)

    Du3 = UpSampling2D((2, 2), interpolation='bilinear')(DA4)
    DpadI3 = tf.pad(Du3, paddings, "SYMMETRIC")
    DcI3 = Conv2D(N_filters[7], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(DpadI3)
    DAI3 = Activation(activation)(DcI3)
    Dcc3 = concatenate([DAI3, EA4])
    Dpad4 = tf.pad(Dcc3, paddings, "SYMMETRIC")
    Dc5 = Conv2D(N_filters[7], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad4)
    DA5 = Activation(activation)(Dc5)
    Dpad5 = tf.pad(DA5, paddings, "SYMMETRIC")
    Dc6 = Conv2D(N_filters[7], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad5)
    DA6 = Activation(activation)(Dc6)

    Du4 = UpSampling2D((2, 2), interpolation='bilinear')(DA6)
    DpadI4 = tf.pad(Du4, paddings, "SYMMETRIC")
    DcI4 = Conv2D(N_filters[8], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(DpadI4)
    DAI4 = Activation(activation)(DcI4)
    Dcc4 = concatenate([DAI4, EA2])
    Dpad6 = tf.pad(Dcc4, paddings, "SYMMETRIC")
    Dc7 = Conv2D(N_filters[8], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad6)
    DA7 = Activation(activation)(Dc7)
    Dpad7 = tf.pad(DA7, paddings, "SYMMETRIC")
    Dc8 = Conv2D(N_filters[8], (filter_size, filter_size), kernel_initializer='he_normal', padding='valid')(Dpad7)
    DA8 = Activation(activation)(Dc8)

    out = Conv2D(N_filters[9], (1, 1), activation=final_activation, padding='valid')(DA8)

    optimizer = Adam(learning_rate=learning_rate,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=1e-07,
                     amsgrad=False,
                     name="Adam")

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=optimizer, loss=loss_function)
    model.summary()

    return model

def valueWeightedMSEHigh():
    def valueWeightedMSE(y_true, p_pred):
        print('yshape: ', y_true.shape)
        fac = K.max(K.max(y_true, axis=1), axis=1) - K.min(K.min(y_true, axis=1), axis=1)
        # fac = K.ones_like(fac_t)

        LOSS = K.mean(K.square(y_true - p_pred) / fac)
        return LOSS
    return(valueWeightedMSE)

def targetMSEHigh():
    '''
    Mean squared error in the near field.
    '''
    def targetMSELow(y_true, p_pred):
        y_obs = y_true[...]
        mu_pred = p_pred[...]
        print(y_obs.shape)

        mask = (y_obs != -1)

        mask = K.cast(mask, float)

        mu_pred *= mask
        y_obs *= mask

        LOSS = K.sum(K.sum(K.sum((K.square(y_obs - mu_pred)),
                                 axis=-1),
                           axis=-1),
                     axis=-1) / K.sum(mask)
        return LOSS
    return(targetMSELow)

class IntermediatePlotCallbackDecoder(keras.callbacks.Callback):
    '''
    Callback to make intermediate plots after each epoch.
    '''
    def __init__(self, x_test, y_test, path, N_epochs, steps):
        self.x_test = x_test
        self.y_test = y_test
        self.path = path
        self.N_epochs = N_epochs
        self.steps = steps
        self.Nperiods = int(self.y_test.shape[-1] / 2)

    def on_epoch_end(self, epoch, logs={}):
        # Make a plot every N epochs, where is chosen such that during training
        # plots are made 10 times.
        N = int(self.N_epochs / self.steps)
        if epoch % N == 0:
            y_pred = self.model.predict(self.x_test)
            print(y_pred.shape)

            for i_ex in range(len(self.y_test)):
                ii_valid = np.where(self.y_test[i_ex, ..., 0] != self.y_test[i_ex, 0, 0, 0])
                iy0 = np.min(ii_valid[0])
                iy1 = np.max(ii_valid[0])
                ix0 = np.min(ii_valid[1])
                ix1 = np.max(ii_valid[1])

                fig, ax = plt.subplots(4, 11, figsize=(11*4, 12))

                for i_channel in range(self.y_test.shape[-1]):
                    vmin = np.min(self.y_test[i_ex, ..., i_channel])
                    vmax = np.max(self.y_test[i_ex, ..., i_channel])

                    if i_channel >= self.Nperiods:
                        ip_channel = i_channel-self.Nperiods
                        l0 = 2
                        l1 = 3
                    else:
                        ip_channel = i_channel
                        l0 = 0
                        l1 = 1
                    ax[l0, ip_channel].pcolormesh(self.y_test[i_ex, iy0:iy1, ix0:ix1, i_channel], vmin=vmin, vmax=vmax)
                    ax[l1, ip_channel].pcolormesh(y_pred[i_ex, iy0:iy1, ix0:ix1, i_channel], vmin=vmin, vmax=vmax)

                fig.tight_layout()
                plt.savefig(self.path + '%04i_%05i.png' % (i_ex, epoch), dpi=60)
                plt.close()

class IntermediatePlotCallbackUnet(keras.callbacks.Callback):
    '''
    Callback to make intermediate plots after each epoch.
    '''
    def __init__(self, x_test, y_test, path, N_epochs, steps):
        self.x_test = x_test
        self.y_test = y_test
        self.path = path
        self.N_epochs = N_epochs
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):

        # Make a plot every N epochs, where is chosen such that during training
        # plots are made 10 times.
        N = int(self.N_epochs / self.steps)
        if epoch % N == 0:
            y_pred = self.model.predict(self.x_test)

            for i_ex in range(len(self.y_test)):
                ii_valid = np.where(self.y_test[i_ex, ..., 0] != self.y_test[i_ex, 0, 0, 0])
                iy0 = np.min(ii_valid[0])
                iy1 = np.max(ii_valid[0])
                ix0 = np.min(ii_valid[1])
                ix1 = np.max(ii_valid[1])

                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                vmin = np.min(self.y_test[i_ex, ..., 0])
                vmax = np.max(self.y_test[i_ex, ..., 0])

                ax[0].pcolormesh(self.y_test[i_ex, iy0:iy1, ix0:ix1, 0], vmin=vmin, vmax=vmax)
                ax[1].pcolormesh(y_pred[i_ex, iy0:iy1, ix0:ix1, 0], vmin=vmin, vmax=vmax)

                fig.tight_layout()
                plt.savefig(self.path + '%04i_%05i.png' % (i_ex, epoch), dpi=60)
                plt.close()

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_files, y_files, X_ch_indices, norm_dicts, rotate=True, do_weight=True, batch_size=8, dim=(256,256), n_channels_out=2, shuffle=True):
        'Initialization'
        self.X_files = X_files
        self.y_files = y_files
        self.X_ch_indices = X_ch_indices
        self.norm_dicts = norm_dicts
        self.rotate = rotate
        self.do_weight = do_weight
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.on_epoch_end()

        norm_dict = norm_dicts[0].copy()
        for nd in self.norm_dicts[1:]:
            for key in norm_dict.keys():
                norm_dict[key]['min'] = np.minimum(norm_dict[key]['min'], nd[key]['min'])
                norm_dict[key]['max'] = np.maximum(norm_dict[key]['max'], nd[key]['max'])
        self.mean_min = norm_dict['mean']['min']
        self.mean_max = norm_dict['mean']['max']
        self.std_min = norm_dict['std']['min']
        self.std_max = norm_dict['std']['max']
        self.X_min = norm_dict['X']['min'][self.X_ch_indices]
        self.X_max = norm_dict['X']['max'][self.X_ch_indices]
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X_files_temp = [self.X_files[k] for k in indexes]
        y_files_temp = [self.y_files[k] for k in indexes]
        # print(X_files_temp)
        # print(y_files_temp)

        # Generate data
        X, y, weights = self.__data_generation(X_files_temp, y_files_temp)

        return (X, y, weights)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X_files_temp, y_files_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Calc weights
        weights = (np.array(['control' in file for file in X_files_temp], dtype=np.float32) * 6 + 1).reshape(self.batch_size, 1, 1, 1)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, len(self.X_ch_indices)), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out), dtype=np.float32)
        # Generate data
        for (i, xfile), yfile in zip(enumerate(X_files_temp), y_files_temp):
            # Store sample
            X[i,] = np.load(xfile)[..., self.X_ch_indices]

            # Store class
            y[i,] = np.load(yfile)

        # Scale data
        
        X = (X - self.X_min) / (self.X_max - self.X_min)
        y[..., 0] = (y[..., 0] - self.mean_min) / (self.mean_max - self.mean_min)
        y[..., 1] = (y[..., 1] - self.std_min) / (self.std_max - self.std_min)

        # rotate sample
        if self.rotate:
            nrot = np.random.randint(4)
            y = np.rot90(y, k=nrot, axes=(1, 2))
            X = np.rot90(X, k=nrot, axes=(1, 2))
        # if self.do_weight:
        return X, y, weights
        # else:
            # return X, y
    
def relative_mse(y_true, y_pred): 
    loss = tf.math.reduce_mean(((y_true-y_pred) / tf.math.reduce_std(y_true, axis=(1, 2), keepdims=True))**2)
    return loss

def relative_mse_numpy(y_true, y_pred): 
    loss = tf.math.reduce_mean(  ((y_true-y_pred) / tf.math.reduce_std(y_true, axis=(1, 2), keepdims=True))**2  )

    loss = np.mean(((y_true - y_pred) / np.std(y_true, axis=(1, 2)))**2)
    return loss