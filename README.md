# Explainable Reinforcement-Learning from Human-Feedback for Concept Bottleneck Models (XRLHF_CBMs)
GitHub repository to manage the UniTrento course project for "Advanced Topics in Machine Learning & Optimization (145871)"

## Description of the project
This project (code and report) is used to evaluate the course. The project is to dive into a possible pathway into combining a machine learning model to debug buggy machine learning model. This is done by using a ranker to rank explanations that are produces by Concept Bottleneck Models (CBMs), one buggy CBM and one oracle CBM is needed. Please see the report for a more detailed description of the project.

### Setup of this repository
This repository is set up in the following way:
- appendix folder: this folder contains the example output and distribution plots seen in the appendix of the report.
- data folder: please see README.md in this folder.
- models folder: this folder contains the final version of the buggy and oracle models.
- notebooks folder: the folder with the running notebooks. There are multiple versions, explained below:
  - buggy folder: this folder contains code used to create the buggy machine learning model trained on confounded training data.
    - evaluation folder: this folder contains code for evaluating the oracle and buggy models. Furthermore, there is also a notebook to inspect the output of the CBMs for any split of the data.
  - oracle folder: this folder contains code used to create the oracle
    - V1: trained on the non-confounded train and non-confounded test splits.
    - V2: trained on the confounded train and non-confounded test splits.
  - ranker folder: this folder contains code for training the ranker model
- src folder: this folder contains all code that was used/imported into the notebooks. This could be further improved by cleaning and better parameter passing.

## Installation
The development of this project mainly happend on Kaggle because of the Graphical Processing Unit (GPU) needed to speed up vision backbone (InceptionV3) and the learning of the model.

Therefore, there is a ```requirements_kaggle.txt``` file which contains the Kaggle packages. The development was done on Python 3.10.12. The example notebook worked straight out-of-the-box on Kaggle. There is also a ```pyproject.toml``` file which was manually created and imported all the necessary dependencies from the imports that were used in the example CEM notebook, but this was done in Python 3.8.10. The ```requirements.txt``` was exported by Poetry.