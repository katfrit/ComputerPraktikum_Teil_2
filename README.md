# KNN Classification — Project Overview

This project implements a **binary classification** method using the *k-nearest neighbors (kNN)* algorithm.

The program supports different fold generation modes and includes a naive kNN implementation intended for **debugging and validation**.

---

## Method Summary

1. Load a labeled dataset from a CSV file
2. Optionally restrict the dataset size for quick tests
3. Split the data into *l* folds
4. Perform *l*-fold cross-validation
5. Evaluate kNN classifiers for all k ∈ {1, …, kmax}
6. Compute and report the average classification error for each k

---

## Program Usage

```bash
classify.py [-h] [-f <folds>] [-k <kmax>] [-n <N>] [-d <mode>] [--knn-mode <impl>] <datasetname>
```
## Parameters

| Parameter           | Description                                                                       | Default  |
| ------------------- | --------------------------------------------------------------------------------- | -------- |
| `-h`                | Show help message and exit                                                        | —        |
| `-f <folds>`        | Number of folds *l* for cross-validation                                          | 5        |
| `-k <kmax>`         | Maximum value for *k*, defining K = {1,…,kmax}                                    | 200      |
| `-n <N>`            | Use only the first *N* data points (for quick tests and debugging)                | all data |
| `-d <mode>`         | Fold generation mode:<br>0 = random partition<br>1 = deterministic partition      | 0        |
| `--knn-mode <impl>` | kNN implementation mode:<br>`naive` = naive distance computation and full sorting | naive    |
| `datasetname`       | Path to the dataset CSV file                                                      | —        |

## Input Files
The program expects a CSV file where each row has the form: y, x1, x2, ..., xd with:
- y ∈ {−1, 1} (class label)
- xi ∈ [−1, 1] (feature values)

## Output
For each k ∈ {1,…,kmax}, the program prints the average classification error obtained via l-fold cross-validation.

Example output:
```bash
k=  1, avg error=0.6000
k=  2, avg error=0.4600
k=  3, avg error=0.4200
```

## Notes on Performance

The naive kNN implementation computes all pairwise distances and sorts them for each prediction. This results in high computational cost and is intended for:
- debugging
- validation
- small-scale experiments

For quick tests, it is recommended to use the -n option to limit the
dataset size. Example:
```bash
python classify.py -f 2 -k 5 -n 100 classification_data/bananas-1-2d.train.csv
```

## Content of classify.py

- Parse command-line arguments
- Load and optionally restrict the dataset
- Generate cross-validation folds
- Perform naive kNN classification
- Compute average cross-validation error for each k
- Print diagnostic output for reproducibility

## Project Run-Time 

| Data Set                      | Steinwart Run Time (in s)   | Run Time - main (error)   | Run Time Claude (error) | Run Time Gemini (error)
| ------------------------------|-----------------------------|---------------------------|-------------------------|-----------
| bananas-1-2d                  | 15.63                       | 10.6 (0.223)              | 9.30 (0.223)            |
| bananas-2-2d                  | 15.35                       | 12.1 (0.224)              | 13.9 (0.224)            |
| bananas-5-2d                  | 18.15                       | 13.7 (0.253)              | 13.5 (0.252)            |
| bananas-1-4d                  | 16.26                       | 14.2 (0.223)              | 10.3 (0.224)            |
| bananas-2-4d                  | 19.39                       | 16.2 (0.220)              | 17.4 (0.223)            |
| bananas-5-4d                  | 25.24                       | 18.8 (0.254)              | 19.5 (0.253)            | 15.9 (0.253)
| crosses-2d                    | 21.06                       | 14.9 (0.274)              | 13.7 (0.273)            |
| toy-2d                        | 20.46                       | 13.7 (0.220)              | 13.9 (0.220)            |
| toy-3d                        | 21.16                       | 16.3 (0.219)              | 15.8 (0.220)            |
| toy-4d                        | 21.20                       | 17.0 (0.219)              | 16.1 (0.216)            |
| toy-10d                       | 31.48                       | 24.0 (0.222)              | 23.0 (0.223)            | 17.3 (0.222)
| phishing.small                | 484,22?                     | 370                       |                         |
| magic_gamma_telescope.small   | 131.88                      | 118.5 (0.163)             | 122.2 (0.163)           | 90.0 (0.162)
| cod-rna.small                 | 238.27                      | 189 (0.073)               | 160.2 (0.071)           | 137.4 (0.076)
| covtype.small                 | 3600 (exit)                 | 1145                      | 1061                    | 628.8 (0.167)
