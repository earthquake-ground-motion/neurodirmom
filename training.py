import sys
import configparser
import numpy as np
from ann_lib import generate_unet, DataGenerator, relative_mse
from processing_lib import init_path
import glob
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import tensorflow as tf
import gc

tf.config.run_functions_eagerly(False)

CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
session = Session(config=CONFIG)

# %%% Config %%%

config_dict = {'relative_mse': relative_mse,
               'rrup': 0,
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

input_dim = (256, 256, len(X_ch_indices))
output_dim = (256, 256, 2)
out_dir = out_dir + config_ID +'/'
init_path(out_dir)

# %%% Read data %%%
# GEN
X_files_GEN = np.sort(glob.glob(data_source_dir + 'samples_GEN_perT/X_*.npy'))
y_files_GEN = np.sort(glob.glob(data_source_dir + 'samples_GEN_perT/y_*.npy'))

train_mask_GEN = np.load(data_source_dir + 'samples_GEN_perT/GEN_train_mask.npy')
val_mask_GEN = np.load(data_source_dir + 'samples_GEN_perT/GEN_val_mask.npy')

train_mask_GEN = train_mask_GEN.repeat(11) # Go from per-rupture to per-rupture and per-T
val_mask_GEN = val_mask_GEN.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_GEN = np.load(data_source_dir + 'samples_GEN_perT/norm_dict.npy',
                        allow_pickle=True)[()]

# NZ
X_files_NZ = np.sort(glob.glob(data_source_dir + 'samples_NZ_perT/X_*.npy'))
y_files_NZ = np.sort(glob.glob(data_source_dir + 'samples_NZ_perT/y_*.npy'))

train_mask_NZ = np.load(data_source_dir + 'samples_NZ_perT/NZ_train_mask.npy')
val_mask_NZ = np.load(data_source_dir + 'samples_NZ_perT/NZ_val_mask.npy')

train_mask_NZ = train_mask_NZ.repeat(11) # Go from per-rupture to per-rupture and per-T
val_mask_NZ = val_mask_NZ.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_NZ = np.load(data_source_dir + 'samples_NZ_perT/norm_dict.npy',
                        allow_pickle=True)[()]

# TUR 
X_files_TUR = np.sort(glob.glob(data_source_dir + 'samples_TUR_perT/X_*.npy'))
y_files_TUR = np.sort(glob.glob(data_source_dir + 'samples_TUR_perT/y_*.npy'))

train_mask_TUR = np.load(data_source_dir + 'samples_TUR_perT/TUR_train_mask.npy')
val_mask_TUR = np.load(data_source_dir + 'samples_TUR_perT/TUR_val_mask.npy')

train_mask_TUR = train_mask_TUR.repeat(11) # Go from per-rupture to per-rupture and per-T
val_mask_TUR = val_mask_TUR.repeat(11) # Go from per-rupture to per-rupture and per-T

norm_dict_TUR = np.load(data_source_dir + 'samples_TUR_perT/norm_dict.npy',
                        allow_pickle=True)[()]

# Merge
X_files_train = []
if 'NZ' in tr_sets:
    X_files_train.extend(list(X_files_NZ[train_mask_NZ].flatten()))
if 'TUR' in tr_sets:
    X_files_train.extend(list(X_files_TUR[train_mask_TUR].flatten()))
if 'GEN' in tr_sets:
    X_files_train.extend(list(X_files_GEN[train_mask_GEN].flatten()))

y_files_train = []
if 'NZ' in tr_sets:
    y_files_train.extend(list(y_files_NZ[train_mask_NZ].flatten()))
if 'TUR' in tr_sets:
    y_files_train.extend(list(y_files_TUR[train_mask_TUR].flatten()))
if 'GEN' in tr_sets:
    y_files_train.extend(list(y_files_GEN[train_mask_GEN].flatten()))

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

# %%% Start training %%%

training_generator = DataGenerator(X_files=X_files_train,
                                   y_files=y_files_train,
                                   X_ch_indices=X_ch_indices,
                                   norm_dicts=[norm_dict_GEN, norm_dict_NZ, norm_dict_TUR],
                                   batch_size=batch_size)

validation_generator = DataGenerator(X_files=X_files_val,
                                     y_files=y_files_val,
                                     X_ch_indices=X_ch_indices,
                                     norm_dicts=[norm_dict_GEN, norm_dict_NZ, norm_dict_TUR],
                                     batch_size=1,
                                     rotate=False)

model = generate_unet(N_filters,
                      learning_rate,
                      input_dim,
                      filter_size,
                      activation,
                      final_activation,
                      loss_function)

tensorboard_callback = TensorBoard(log_dir=out_dir + 'log/')

checkpoint_callback = ModelCheckpoint(out_dir+'model',
                                      monitor='val_loss',
                                      mode='min',
                                      save_best_only=True,
                                      save_weights_only=False)

callbacks_list = [tensorboard_callback,
                  checkpoint_callback]

model.fit(x=training_generator,
          epochs=N_epochs,
          callbacks=callbacks_list,
          validation_data=validation_generator,
          use_multiprocessing=True,
          workers=16)