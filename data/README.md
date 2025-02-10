# Data folder
This folder contains the data for this project.

## Original dataset
The original CUB-200-2011 dataset is available on [Kaggle](https://www.kaggle.com/datasets/wenewone/cub2002011) or from the original [download link](https://www.vision.caltech.edu/datasets/cub_200_2011/).

## Filtered dataset
As indicated by the authors of CBM/CEM, the CUB-200-2011 dataset has noisy concept labels as they were labeled by non-experts.
Therefore, they applied a filtering technique to filter from 312 concepts to 112. The same filtering has been applied in this project.

The filtered data can be found the the ```train.pkl```, ```val.pkl```, and ```test.pkl``` files.

## Configuration
The cub.yaml file contains the configuration for the dataloader from CEM.