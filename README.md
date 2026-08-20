# Inverse Landscape Modelling Python (ILMP)

Landscape evolution model for river profile, comogenic nuclide and low-temperature thermochronologic data predictions.

## Installation Requirement:
1) python3.10 (mamba create -n ILMP python=3.10)
2) numpy, scipy, matplotlib, pandas (conda)
3) sbi, Pebble, shapely (pip)

## Model Description:

### Getting Started

The main python files is main_forward_ilmp.py. This file allow to perform 1D forward landscape evolution modelling and allow to predict river node elevations, cosmogenic nuclide concentrations and low-temperature thermochronology depending parameters (uplift, erodibility, ...). The default parameters assume a constant uplift and erodibility. The user can complexify the model by adding functions like Tilting_to_Uplift and Lithology_to_Erodibility to simulate tilting in the uplift or change in the erodibility depending the lithology. A default river network of the Neckar catchment is provided

### Inverse Modelling

The python file to perform inverse modelling is main_inverse_modelling.py. The user can set unknown parameters to be tested in order to replicate observed data of the river network (elevation. cosmogenic nuclide concentrations and thermochronological ages). Unknown parameters have to set as fp[0], fp[1], ... in the default parameter and functions. Prior knowledge of the unknow parameters can be changed in the Prior class.

### River Network Generation

Catchment river network can be generated and ready to be used by the two main python script main_forward_ilmp.py and main_inverse_modelling.py. The python script catchment_dictionnary_functions.py allow to convert a .csv river network data file extracted from LSTopoTools (Mudd et al., 2014) to a dictionary .pkl which can be used by the ILMP code. A default river network of the Neckar catchment is provided in the data folder.

## Citation

Manuscript: Bernard et al., Estimation of Denudation Parameters and River Capture Events from Neural Network Inverse Modelling of River Profiles and Thermo- and Geochronology Data, 2024

Numerical code: https://doi.org/10.5281/zenodo.10473523
