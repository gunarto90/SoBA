# SoBA
Social tie classification using location based social network data, implemented using Python.

If you are interested to apply this work in your research or project, please cite the following work.

Gunarto Sindoro Njoo, Min-Chia Kao, Kuo-Wei Hsu, Wen-Chih Peng: Exploring Check-in Data to Infer Social Ties in Location Based Social Networks. PAKDD (1) 2017: 460-471

## Dataset
Dataset used in the experiments are obtained from LBSN data: Gowalla and Brighkite (http://snap.stanford.edu/data/).

The processed dataset used throughout the SoBA experiments can be found in this link (http://bit.ly/pakdd2017-dataset-social).

### base.py
The functions needed in other python files are provided here. It is a library for this project.

### classes.py
Provided the definition required for the classes used throughout this project

### colocation.py
Co-location generator, the first phase in our work, generating the co-location between check-ins of two users. The co-location implemented in this work apply two parameters: location distance and time difference. As for the location distance, we applied 0m (same venue id) to serve as same location, whereas this parameter can be modified by using the haversine distance between check-ins' venue. Time difference can varies from minutes, hours, days, or even weeks.

### colocation_filter.py 
This code is intended to generate a new check-in and friend dataset based on the users who have at least 2 co-location with other users (Useful for testing the CCS2017 method)
@inproceedings{BHPZ17,
author = {Michael Backes and Mathias Humbert and Jun Pang and Yang Zhang},
title = {walk2friends: Inferring Social Links from Mobility Profiles.},
booktitle = {Proceedings of the 24th ACM SIGSAC Conference on Computer and Communications Security (CCS)},
year = {2017},
publisher = {ACM}
}

### extract_checkin.py
Similar to colocation_filter.py except that this code extract all the records from the original data (Useful for testing the CCS2017 method)

### SOBA.py
Feature extractions of our method as well as the evaluation of our method are implemented here. 

### pgt.py
Competitor of this work, using PGT framework. H. Wang, Z. Li and W. C. Lee, "PGT: Measuring Mobility Relationship Using Personal, Global and Temporal Factors," 2014 IEEE International Conference on Data Mining, Shenzhen, 2014, pp. 570-579.

### general_utilities.py
The functions needed in other python files are provided here. It is a library for this project.

### soc_evaluation.py
The evaluation file for the proposed framework.

### pgt_evaluation.py
The evaluation file for PGT method.

### plot.py
Python codes to generate plots.
