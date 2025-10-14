# Calibration of Multi-Spectral Sensor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15224597.svg)](https://doi.org/10.5281/zenodo.15224597) 

## Overview

This repository contains the code, data, and results to calibrate of the multi-spectral sensor AS7341 to Photosynthetically Active Radiation.

## Project Description
Photosynthetically Active Radiation (PAR) sensors are essential tools in plant stress monitoring, ecophysiology, and forest growth modeling. Recently, multi-spectral PAR sensors have emerged as promising alternatives to conventional single-channel systems, offering comparable accuracy and additional spectral information.  
However, existing calibration methods rely on single-channel references under variable ambient light conditions, which limits their reliability across diverse lighting environments. This study introduces a novel calibration methodology conducted in a controlled light environment to ensure consistent, reproducible calibration independent of external light variability. The Partial Least Squares (PLS) regression model was tailored for this application by integrating the optical properties of the system components through a customized preprocessing framework. PLS effectively manages collinearity among spectral channels, generating individual calibration coefficients while compensating for spectral leakage, particularly in the near-infrared range. The proposed calibration approach is validated through field experiments under varying weather conditions, demonstrating a normalized RMSE of just 3.92% compared to a commercial PAR sensor. 

## Repository Structure

├── data/ # Raw and processed data sets  
├── src/ # Source code for analysis  
├── supplementary_material/ # Additional documentation of the AS7341 characteristic  
├── LICENSE # License for project  
└── README.md # Project overview

## Installation

To reproduce the analysis, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Multi-Spectral-PAR-Sensor.git
cd PATH/your-repo-name/src
pip install -r requirements.txt
```
Consider to use an own environment.

## Usage

1. **Prepare the Data**:
   - Use the raw measurement data in the `/data` directory or place there your input data files. Ensure the data format matches the requirements of the library.

2. **Run the Notebook**:
   - Open `Calibration_Execution.ipynb` in Jupyter Notebook or Visual Studio Code.
   - Follow the steps in the notebook to load the data, apply the calibration functions from `Calibration.py`, and view the results.

3. **Output**:
   - The results of the calibration will be displayed in the notebook and saved to an output file in the `/data` directory.


## Citation of Publication
If you use this project, please cite our publication.

## Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15224597.svg)](https://doi.org/10.5281/zenodo.15224597) 

## Acknowledgments
This work was conducted in the framework of the collaborative research centre ECOSENSE with funding from the German Research Foundation SFB 1537/1 ECOSENSE.  
Further information: [ECOSENSE website](https://uni-freiburg.de/ecosense/)  
DOI to [ECOSENSE Grand Proposal](https://doi.org/10.3897/rio.10.e129357)
