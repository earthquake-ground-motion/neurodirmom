import numpy as np
import h5py
import glob
from joblib import Parallel, delayed
from processing_lib import process_rupture_file, init_path

# %%% Config %%%

do_NZ = False
do_GEN = False
do_TUR = True

# %%% Turkey ruptures %%%        

if do_TUR:
    rupture_files = np.sort(glob.glob('data/raw/TUR/Turkey_ERF_Simulation_V1/*.hdf5'))
    master_file = 'data/raw/turkey_ruptures/Turkey_ERF_Simulation_V1.hdf5'
    save_dir = 'data/samples_TUR_perT/'
    init_path(save_dir)

    with h5py.File(master_file, 'r') as master_db:
        grid_llat = np.array(master_db['grid']['lat'])
    master_dim = grid_llat.shape

    out = Parallel(n_jobs=32)(delayed(process_rupture_file)(file, master_dim, save_dir=save_dir) for file in rupture_files)
    X_min = np.min(np.stack([out[i][0] for i in range(len(rupture_files))]), axis=0)
    X_max = np.max(np.stack([out[i][1] for i in range(len(rupture_files))]), axis=0)
    min_amp_mean = np.min([out[i][2] for i in range(len(rupture_files))])
    max_amp_mean = np.max([out[i][3] for i in range(len(rupture_files))])
    min_amp_std = np.min([out[i][4] for i in range(len(rupture_files))])
    max_amp_std = np.max([out[i][5] for i in range(len(rupture_files))])
        
    norm_dict = {}
    norm_dict['mean'] = {'min': min_amp_mean,
                         'max': max_amp_mean}
    norm_dict['std'] = {'min': min_amp_std,
                        'max': max_amp_std}
    norm_dict['X'] = {'min': X_min,
                      'max': X_max}

    np.save(save_dir + 'norm_dict.npy', norm_dict)

if do_GEN:
    rupture_files = np.concatenate((np.sort(glob.glob('data/raw/GEN/generic_henning_N/*.hdf5')),
                                    np.sort(glob.glob('data/raw/GEN/generic_henning_R/*.hdf5')),
                                    np.sort(glob.glob('data/raw/GEN/generic_henning_SS/*.hdf5'))))
    master_file = 'data/raw/GEN/generic_henning_N.hdf5'
    save_dir = 'data/samples_GEN_perT/'
    init_path(save_dir)
                  
    with h5py.File(master_file, 'r') as master_db:
        grid_llat = np.array(master_db['grid']['lat'])
    master_dim = grid_llat.shape

    out = Parallel(n_jobs=32)(delayed(process_rupture_file)(file, master_dim, save_dir=save_dir) for file in rupture_files)
    X_min = np.min(np.stack([out[i][0] for i in range(len(rupture_files))]), axis=0)
    X_max = np.max(np.stack([out[i][1] for i in range(len(rupture_files))]), axis=0)
    min_amp_mean = np.min([out[i][2] for i in range(len(rupture_files))])
    max_amp_mean = np.max([out[i][3] for i in range(len(rupture_files))])
    min_amp_std = np.min([out[i][4] for i in range(len(rupture_files))])
    max_amp_std = np.max([out[i][5] for i in range(len(rupture_files))])
        
    norm_dict = {}
    norm_dict['mean'] = {'min': min_amp_mean,
                         'max': max_amp_mean}
    norm_dict['std'] = {'min': min_amp_std,
                        'max': max_amp_std}
    norm_dict['X'] = {'min': X_min,
                      'max': X_max}

    np.save(save_dir + 'norm_dict.npy', norm_dict)

if do_NZ:
    rupture_files = np.sort(glob.glob('data/raw/NZ_09May2023/NZL_2022_ERF_simulation_UNET/FLT_*.hdf5'))
    master_file = 'data/raw/NZ_09May2023/NZL_2022_ERF_simulation_UNET.hdf5'
    save_dir = 'data/samples_NZ_perT/'
    init_path(save_dir)

    with h5py.File(master_file, 'r') as master_db:
        grid_llat = np.array(master_db['grid']['lat'])
    master_dim = grid_llat.shape

    out = Parallel(n_jobs=32)(delayed(process_rupture_file)(file, master_dim, save_dir=save_dir) for file in rupture_files)
    X_min = np.min(np.stack([out[i][0] for i in range(len(rupture_files))]), axis=0)
    X_max = np.max(np.stack([out[i][1] for i in range(len(rupture_files))]), axis=0)
    min_amp_mean = np.min([out[i][2] for i in range(len(rupture_files))])
    max_amp_mean = np.max([out[i][3] for i in range(len(rupture_files))])
    min_amp_std = np.min([out[i][4] for i in range(len(rupture_files))])
    max_amp_std = np.max([out[i][5] for i in range(len(rupture_files))])
        
    norm_dict = {}
    norm_dict['mean'] = {'min': min_amp_mean,
                         'max': max_amp_mean}
    norm_dict['std'] = {'min': min_amp_std,
                        'max': max_amp_std}
    norm_dict['X'] = {'min': X_min,
                      'max': X_max}

    np.save(save_dir + 'norm_dict.npy', norm_dict)