# MDS_CatEnc
Categorical columns encoding example through MultiDimensionalScaling

Data used can be downloaded here: 
https://www.kaggle.com/ishajain30/super-heroes/data

The code transforms categorical data into points in a multi-dimensional space reflecting the distance given by the statistics for each modality across the dataset.

In the example at the end we encode hero's name according to all the other variables. Anyway, its main function could be to map categorical columns (e.g. race or publisher from the same dataset). Therefore, the example is just intended to show how the code works, spatializing superhero's according to all their information.

It uses sklearn's MDS
