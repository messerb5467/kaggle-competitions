#!/bin/env bash
conda install -c anaconda scikit-learn -y
conda install -c anaconda pandas -y
conda install -c anaconda matplotlib -y
conda install -c anaconda seaborn -y
conda install -c anaconda statsmodels -y 
conda install -c conda-forge xgboost -y
conda install -c anaconda tensorflow-estimator -y
conda install -c anaconda keras -y
conda install -c pytorch pytorch -y
conda install -c main pip -y
conda install -c anaconda jupyter_core -y
conda install -c conda-forge notebook -y
conda install -c conda-forge hyperopt -y
yes | pip install kaggle
