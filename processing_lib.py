import numpy as np
import h5py
import os
import shutil

def process_rupture_file(file, master_dim, return_data=False, return_range=True, save_data=True, save_dir=None):
    with h5py.File(file, 'r') as file_db:
        lx = file_db.attrs['lx']
        ux = file_db.attrs['ux']
        ly = file_db.attrs['ly']
        uy = file_db.attrs['uy']

        rake = file_db.attrs['rake']
        SS_flag = int((np.abs(rake) < 30) | (np.abs(rake) > 150 ))
        mag = file_db.attrs['mag']
        dip = file_db.attrs['dip']

        rrup = file_db['global_properties']['rrup'].reshape(master_dim)[ly:uy, lx:ux]
        rjb = file_db['global_properties']['rjb'].reshape(master_dim)[ly:uy, lx:ux]
        ry0 = file_db['global_properties']['ry0'].reshape(master_dim)[ly:uy, lx:ux]
        az = file_db['global_properties']['azimuth'].reshape(master_dim)[ly:uy, lx:ux]
        gc2t = file_db['global_properties']['gc2t'].reshape(master_dim)[ly:uy, lx:ux]
        gc2u = file_db['global_properties']['gc2u'].reshape(master_dim)[ly:uy, lx:ux]
        
        amp = file_db['amplification'][()].reshape((256, 256, 11, 250))

    periods = np.array([0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])

    X_min = np.ones(11) * np.inf
    X_max = np.ones(11) * np.inf * (-1)

    min_amp_mean = np.inf
    max_amp_mean = -np.inf
    min_amp_std = np.inf
    max_amp_std = -np.inf
    for iT, T in enumerate(periods):
        X = np.zeros((256, 256, 11))
        X[..., 0] = rrup
        X[..., 1] = rjb
        X[..., 2] = ry0
        X[..., 3] = (np.cos(np.deg2rad(az)) + 1) / 2
        X[..., 4] = (np.sin(np.deg2rad(az)) + 1) / 2
        X[..., 5] = gc2t
        X[..., 6] = gc2u
        X[..., 7] = SS_flag
        X[..., 8] = mag
        X[..., 9] = T
        X[..., 10] = dip

        X_min = np.minimum(X_min, np.min(X, axis=(0, 1)))
        X_max = np.maximum(X_max, np.max(X, axis=(0, 1)))
        
        ## Get amplification

        amp_mean = np.mean(amp, axis=3)
        amp_std = np.std(amp, axis=3)

        min_amp_mean = np.minimum(min_amp_mean, np.min(amp_mean))
        max_amp_mean = np.maximum(max_amp_mean, np.max(amp_mean))
        min_amp_std = np.minimum(min_amp_std, np.min(amp_std))
        max_amp_std = np.maximum(max_amp_std, np.max(amp_std))

        y = np.concatenate((amp_mean[..., iT:iT+1], amp_std[..., iT:iT+1]), axis=2)
        
        if save_data and save_dir:
            filename = file.split('/')[-1].split('.')[0]
            np.save(save_dir + 'X_' + filename + '_%02i.npy' % (iT), X)
            np.save(save_dir + 'y_' + filename + '_%02i.npy' % (iT), y)
    
    if return_range and not return_data:
        out = [X_min, X_max, min_amp_mean, max_amp_mean, min_amp_std, max_amp_std]
    elif return_data and not return_range:
        out = [X, y]
    elif return_data and return_range:
        out = [[X_min, X_max, min_amp_mean, max_amp_mean, min_amp_std, max_amp_std], [X, y]]

    return out

def norm_X(X, norm_dicts):
    d_dummy = norm_dicts[0]
    Nx = len(d_dummy['X']['min'])

    Xmin = np.ones(Nx) * np.inf
    Xmax = np.ones(Nx) * np.inf * (-1)

    for norm_dict in norm_dicts:
        Xmin = np.minimum(Xmin, norm_dict['X']['min'])
        Xmax = np.maximum(Xmax, norm_dict['X']['max'])
    
    X_norm = (X - Xmin) / (Xmax-Xmin)

    return X_norm

def denorm_yNorm(y_norm, norm_dicts):
    mean_max = -np.inf
    mean_min = np.inf
    std_max = -np.inf
    std_min = np.inf

    for norm_dict in norm_dicts:
        mean_max = np.maximum(mean_max, norm_dict['mean']['max'])
        mean_min = np.minimum(mean_min, norm_dict['mean']['min'])
        std_max = np.maximum(std_max, norm_dict['std']['max'])
        std_min = np.minimum(std_max, norm_dict['std']['min'])
    
    y_min = np.array([mean_min, std_min])
    y_max = np.array([mean_max, std_max])

    y = y_norm * (y_max-y_min) + y_min

    return y

def init_path(path):
    '''
    Creates a directory if it does not exist.

    :input path: Path to be created
    '''
    if os.path.exists(path):
        print('path exists!')
        pass
    else:
        os.makedirs(path)


def renew_path(path):
    '''
    Removes an existing path and recreates it.

    :input path: Path to be recreated.
    '''
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)