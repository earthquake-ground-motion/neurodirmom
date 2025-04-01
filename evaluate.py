import numpy as np
import sys
import configparser
from ann_lib import relative_mse_numpy
from processing_lib import norm_X, denorm_yNorm
import glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import tensorflow as tf
from tensorflow.keras.models import load_model
from processing_lib import norm_X, denorm_yNorm
import pandas as pd

tf.config.run_functions_eagerly(False)

CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
session = Session(config=CONFIG)

# %%% Config %%%
config_dict = {'rrup': 0,
               'rjb': 1,
               'ry0': 2,
               'cosaz': 3,
               'sinaz': 4,
               'gc2t': 5,
               'gc2u': 6,
               'SOF': 7,
               'M': 8,
               'T': 9,
               'dip': 10}

config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
config_ID = config_file.split('/')[-1][:-4]

N_filters = np.array(config['ann']['N_filters'].split(','), dtype=int)
activation = config['ann']['activation']
final_activation = config['ann']['final_activation']
filter_size = int(config['ann']['filter_size'])

learning_rate = float(config['training']['learning_rate'])
N_epochs = int(config['training']['N_Epochs'])
batch_size = int(config['training']['batch_size'])
loss_function = config_dict[config['training']['loss_function']]

data_source_dir = config['data']['source_dir']
tr_sets = config['data']['tr_sets'].split(',')
val_sets = config['data']['val_sets'].split(',')
X_ch_indices = np.array([config_dict[param] for param in config['data']['X_params'].split(',')])

out_dir = config['output']['out_dir']
out_dir = out_dir + config_ID +'/'


# %%% Load model %%%
model_dir = out_dir + 'model/'
model = load_model(model_dir, compile=False)

# %%% Read data %%%
# GEN
X_files_GEN = np.sort(glob.glob(data_source_dir + 'samples_GEN_perT/X_*.npy'))
y_files_GEN = np.sort(glob.glob(data_source_dir + 'samples_GEN_perT/y_*.npy'))

val_mask_GEN = np.load(data_source_dir + 'samples_GEN_perT/GEN_val_mask.npy')

val_mask_GEN = val_mask_GEN.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_GEN = np.load(data_source_dir + 'samples_GEN_perT/norm_dict.npy',
                        allow_pickle=True)[()]

# NZ
X_files_NZ = np.sort(glob.glob(data_source_dir + 'samples_NZ_perT/X_*.npy'))
y_files_NZ = np.sort(glob.glob(data_source_dir + 'samples_NZ_perT/y_*.npy'))

val_mask_NZ = np.load(data_source_dir + 'samples_NZ_perT/NZ_val_mask.npy')

val_mask_NZ = val_mask_NZ.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_NZ = np.load(data_source_dir + 'samples_NZ_perT/norm_dict.npy',
                       allow_pickle=True)[()]

# TUR 
X_files_TUR = np.sort(glob.glob(data_source_dir + 'samples_TUR_perT/X_*.npy'))
y_files_TUR = np.sort(glob.glob(data_source_dir + 'samples_TUR_perT/y_*.npy'))

val_mask_TUR = np.load(data_source_dir + 'samples_TUR_perT/TUR_val_mask.npy')

val_mask_TUR = val_mask_TUR.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_TUR = np.load(data_source_dir + 'samples_TUR_perT/norm_dict.npy',
                        allow_pickle=True)[()]

# Merge
X_files_val = []
if 'NZ' in val_sets:
    X_files_val.extend(list(X_files_NZ[val_mask_NZ].flatten()))
if 'TUR' in val_sets:
    X_files_val.extend(list(X_files_TUR[val_mask_TUR].flatten()))
if 'GEN' in val_sets:
    X_files_val.extend(list(X_files_GEN[val_mask_GEN].flatten()))

y_files_val = []
if 'NZ' in val_sets:
    y_files_val.extend(list(y_files_NZ[val_mask_NZ].flatten()))
if 'TUR' in val_sets:
    y_files_val.extend(list(y_files_TUR[val_mask_TUR].flatten()))
if 'GEN' in val_sets:
    y_files_val.extend(list(y_files_GEN[val_mask_GEN].flatten()))

norm_dicts = [norm_dict_GEN, norm_dict_NZ, norm_dict_TUR]

# %%% Evaluate mean performance %%%

losses = []
residuals_mean = []
residuals_std = []
means = []
stds = []
rrups = []
for X_file, y_file in zip(X_files_val, y_files_val):
    print(X_file)
    X = np.load(X_file)
    y = np.load(y_file)
    X_norm = norm_X(X, norm_dicts)
    print(X.shape, X_norm.shape, X_norm[..., X_ch_indices].shape)
    y_hat_norm = model.predict(X_norm[..., X_ch_indices][None, ...])
    y_hat = denorm_yNorm(y_hat_norm, norm_dicts)

    near_field = X[..., 0] < 100
    residuals_mean.append(y[near_field, 0] - y_hat[0, near_field, 0])
    residuals_std.append(y[near_field, 1] - y_hat[0, near_field, 1])
    means.append(y[near_field, 0])
    stds.append(y[near_field, 1])
    rrups.append(X[near_field, 0])
    T = X[0, 0, 9]
    mag = X[0, 0, 8]
    SOF = X[0, 0, 7]
    dip = X[0, 0, 10]
    if SOF > 0.5:
        SOF = 'SS'
    else:
        SOF = 'DS'
    if 'TUR' in X_file:
        DS = 'TUR'
    elif 'GEN' in X_file:
        DS = 'GEN'
    elif 'NZ' in X_file:
        DS = 'NZ'

    print(y.shape, y_hat.shape)
    loss = relative_mse_numpy(y[None, ...], y_hat)
    losses.append([y_file, T, mag, SOF, dip, loss])
losses = np.array(losses)
loss_db = pd.DataFrame(losses, columns=['Filename', 'Period (s)', 'Magnitude', 'SOF', 'Dip', 'Loss'])
loss_db.to_csv(out_dir + 'loss_db.csv')
np.save(out_dir + 'residuals_mean100km.npy', np.array(residuals_mean, dtype=object))
np.save(out_dir + 'residuals_std100km.npy', np.array(residuals_std, dtype=object))
np.save(out_dir + 'rrups100km.npy', np.array(rrups, dtype=object))
np.save(out_dir + 'means100km.npy', np.array(means, dtype=object))
np.save(out_dir + 'stds100km.npy', np.array(stds, dtype=object))
