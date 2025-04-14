# neurodirmom
Moment Modifier Directivity Model via a U-NET: Code and Model to support Lilienkamp &amp; Weatherill (2025)

```
Lilienkamp, H., Weatherill, G. (2025) "Efficient incorporation of rupture directivity in to probabilistic seismic hazard analysis using a deep learning-based approach", Earthquake Spectra, in press
```

The repository associated with the release of the Lilienkamp & Weatherill (2025) manuscript is available from at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15118499.svg)](https://doi.org/10.5281/zenodo.15118499)

The code within this repository was developed by Henning Lilienkamp (formerly of GFZ Helmholtz Centre for Geosciences)

For further information please contact:

```
Graeme Weatherill
GFZ Helmholtz Centre for Geosciences
Telegrafenberg
14473 Potsdam
Germany

https://www.gfz.de/en/staff/graeme.weatherill/sec26
```


## Dependencies:

The program runs in Python 3.10 (or greater) and requires the following dependencies (versions will largely depend on the tensorflow version)

```
Numpy
Scipy
Pandas
Matplotlib
h5py
joblib
Keras
Tensorflow
```



## Installation

Where possible, we recommend running this program within a virtual environment. The installation will require than an HDF5 library is available on your operating system. See [https://www.h5py.org/](https://www.h5py.org/) for more details.

To setup and install the virtual environment:

```
>> python -m venv /path/to/neurodirmom/virtualenv
>> source /path/to/neurodirmom/virtualenv/bin/activate
(venv) >> pip install --upgrade pip 
```

General installation of the requirements can be undertaken by:

```
(venv) >> pip install numpy scipy pandas matplotlib h5py joblib keras tensorflow
```
**Please note that this method does not resolve dependency conflicts and may result in errors on certain operating systems and versions! See pip or Python packaging documentation for details**


If the dependencies are installed successfully then `neurodirmom` can be downloaded from: [https://github.com/earthquake-ground-motion/neurodirmom.git](https://github.com/earthquake-ground-motion/neurodirmom.git) or cloned with Git on the command line via

```
(venv) >> git clone https://github.com/earthquake-ground-motion/neurodirmom.git
```

## Execution

Execution of the code requires a configuration file structured according to the `example_config.ini` file contained within the repository:

```
[ann]
N_filters = 16,32,64,128,256,128,64,32,16,2
activation = leaky_relu
final_activation = linear
filter_size = 3

[training]
learning_rate = 0.0001
N_epochs = 1000
batch_size = 8
loss_function = relative_mse

[data]
source_dir = NEURODIRMOM/data/
tr_sets = GEN,NZ,TUR
val_sets = GEN,NZ,TUR
X_params = rrup,rjb,gc2u,ry0,gc2t,gc2u,M,SOF,T

[output]
out_dir = NEURODIRMOM/results/
```

For more information on the execution of the algorithm and the settings of the U-Net, please refer to the supplementary material of the manuscript.

The training process is executed via

```
(venv) >> cd neurodirmom
(venv) >> python training.py example_config.ini 
```



 





