# lkf_tools
Tools to detect and track deformation features (leads and pressure ridges) in sea-ice deformation data.

## Getting Started

### Installing python
install python and install packages scipy.ndimage and skimage.morphology

### Download RGPS example data

RGPS data in Lagrangian and Eulerian format need to be downloaded from Ron Kwok's homepage:
``
https://rkwok.jpl.nasa.gov/radarsat/index.html`
```
RGPS data needs to be unzip. The data needs to be orgnaized in a seperate directory for each winter that are named w9798, w9899, ...


## Generate LKF data-set

Use gen_dataset.py to generate LKF data-sets, which performs three steps for each year:
* run the LKF detection on RGPS deformation data
* interpolate Lagrangian drift data to Eulerian grid
* run the LKF tracking algorithm


## Author

Nils Hutter
nils.hutter@awi.de

## License
