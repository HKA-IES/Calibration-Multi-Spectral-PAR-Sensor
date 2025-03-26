# Calibration of Multi-Spectral Sensor

## Overview

This repository contains the code, data, and results for the publication:  
**"Calibration of Multi-Spectral Photosynthetically Active Radiation Sensor"**  
Authors: Johannes Klueppel, Megan Jurzcak, Ulrike Wallrabe, Laura Maria Comella  
Published in: Journal Name (Year)  
DOI: [Link to DOI](https://doi.org/10.5281/zenodo.14981748)

## Project Description
**Add abstract when published.**

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
Run the ``preprocessing`` script to convert the raw data to normalized data.  
Run ``PAR_PLS`` script to calculate the coefficients.  
Run the ``graphs`` script to show the results based on the mean values.  
In one of the first jupyter cell, the sensor number can be chosen.

## Results
Check our publication on: ...

## Citation
**Add when published.**

If you use this project, please cite our publication:

```
@article{YourName2024,
  title={Title of Your Scientific Publication},
  author={Your Name and Co-author Names},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxx}
}
```

## Acknowledgments
This work was conducted in the framework of the collaborative research centre ECOSENSE with funding from the German Research Foundation SFB 1537/1 ECOSENSE.  
Further information: [ECOSENSE website](https://uni-freiburg.de/ecosense/)  
DOI to [ECOSENSE Grand Proposal](https://doi.org/10.3897/rio.10.e129357)
