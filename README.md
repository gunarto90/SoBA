# Distinguishing Friends from Strangers in Location-based Social Networks using Co-location

We aim to differentiate the friends from strangers in the location-based social network (LBSN). To generate a fine result, we make use of the concept of "co-location" which is a simulation of the meeting between two people from the check-ins data. However, in the "co-location" data, there are too much noise and it is difficult to find the real friend. Therefore, we extract few key features and employ a Random Forests to learn the model that can distinguish the friends from strangers well.

This code is the implementation of the following publications.
> Njoo, G. S., Hsu, K. W., & Peng, W. C. (2018). Distinguishing friends from strangers in location-based social networks using co-location. Pervasive and Mobile Computing, 50, 114-123.

> Njoo, G. S., Kao, M. C., Hsu, K. W., & Peng, W. C. (2017, May). Exploring check-in data to infer social ties in location based social networks. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 460-471). Springer, Cham.

Baseline compared in this model is PGT
> PGT: Measuring Mobility Relationship Using Personal, Global and Temporal Factors (ICDM 2014)
