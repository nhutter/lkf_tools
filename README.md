# lkf_tools
Tools to detect and track deformation features (leads and pressure ridges) in sea-ice deformation data.

## Getting Started

### Installing python
install python and install packages scipy.ndimage and skimage.morphology

### Download RGPS example data

RGPS data in Lagrangian and Eulerian format need to be downloaded from Ron Kwok's homepage:
https://rkwok.jpl.nasa.gov/radarsat/index.html

RGPS data needs to be unzip. The data needs to be orgnaized in a seperate directory for each winter that are named w9798, w9899, ...


## Generate LKF data-set

Use gen_dataset.py to generate LKF data-sets, which performs three steps for each year:
* run the LKF detection on RGPS deformation data
* interpolate Lagrangian drift data to Eulerian grid
* run the LKF tracking algorithm


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

