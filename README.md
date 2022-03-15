# lkf_tools
Tools to detect and track deformation features (leads and pressure ridges) in sea-ice deformation data.

## Getting Started
Download/clone this repository.

### Installing python
First you need to install conda to install the python environment needed for this package. This can easily be done using a [miniforge](https://github.com/conda-forge/miniforge).

After installing conda with a miniforge you can install the python environment using:
```
conda env create -f environment.yml
```
and activate the environment:
```
conda activate lkf_tools
```

## Generate LKF data-set

There is a [tutorial notebook](notebooks/tutorial_gen_dataset.ipynb) that illustrates how to generate a LKF data-set from a netcdf file. This tutorial uses model output from the [SIREx model output repository](https://doi.org/10.5281/zenodo.5555329) and also uses the SIREx sampling strategies that are described in detail in this [preprint](https://www.essoar.org/doi/10.1002/essoar.10507396.1). The tutorial shows you how to:
* download and read in the netcdf file
* detect LKFs in the netcdf file
* run the tracking algorithm on the detected LKFs
* some basic plotting routines of the extracted LKFs


## Algorithm description

An in-depth description of the algorithm can be found here:
```
Hutter, N., Zampieri, L., and Losch, M.: Leads and ridges in Arctic sea ice from RGPS data and a new tracking algorithm, The Cryosphere Discuss., https://doi.org/10.5194/tc-2018-207, accepted for publication, 2018. 
```


## Author

Nils Hutter
nils.hutter@awi.de

## DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2560078.svg)](https://doi.org/10.5281/zenodo.2560078)


## License
GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

