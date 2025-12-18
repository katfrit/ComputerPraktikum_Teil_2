# KNN Classification — Project Overview

This project implements a **binary classification** method using the *k-nearest neighbors (kNN)* algorithm.

## Method Summary

1. ...

## Program Usage
`classify.py -f <folds> -k <kmax> -d <mode> <datasetname>`

## Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `-h` | Show help and exit | — |
| `-f <folds>` | Number of folds *l* | 5 |
| `-k <kmax>` | Maximum value for *k*, defining K = {1,…,kmax} | 200 |
| `-d <mode>` | Fold mode: 0 = random, 1 = deterministic | 0 |
| `datasetname` | Name of the dataset (without file extension) | — |

## Input Files

The program expects CSV files like:

`../classification-data/<datasetname>.csv`

Each line must have the format:

`<label>, <feature1>, <feature2>, ...`

## Content of `classify.py`

- Parse command-line arguments  
- Load dataset  
- Generate folds  
- Create the set of candidate k-values  
- Prepare the structure for the later kNN classification (to be implemented)
