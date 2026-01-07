#!/usr/bin/env python3
import argparse
import sys
import random
import math

def parse_args():
    parser = argparse.ArgumentParser(
        description="KNN classification with optional folding parameters",
        add_help=False
    )

    parser.add_argument("datasetname", nargs="?", help="Name of the dataset to load")
    parser.add_argument("-h", action="store_true", help="Show help and exit")
    parser.add_argument("-f", type=int, default=5, help="Number of folds (default 5)")
    parser.add_argument("-k", type=int, default=200, help="Maximum k value (default 200)")
    parser.add_argument("-d", type=int, default=0, help="Mode: 0=random, 1=deterministic")
    parser.add_argument("--knn-mode", choices=["naive"], default="naive", help="KNN implementation mode (only 'naive' supported)")
    parser.add_argument("-n", type=int, default=None, help="Use only the first N data points (for quick tests)")

    args = parser.parse_args()
    return args

def generate_folds(data, l, mode):
    """Generate l folds either randomly (mode 0) or deterministically (mode 1)."""
    n = len(data)
    folds = [[] for _ in range(l)]

    if mode == 0:
        random.shuffle(data) # mix data pairs "in place"
        for i, item in enumerate(data):
            folds[i % l].append(item)
    else:
        # deterministic: D1 = (1, l+1, 2l+1, ...), D2 = (2, l+2, ...)
        for idx, item in enumerate(data):
            folds[idx % l].append(item)

    return folds

def knn_predict_naive(x, D, k):
    distances = []

    for y_i, x_i in D:
        dist = 0.0
        for a, b in zip(x, x_i):
            dist += (a - b) ** 2
        dist = math.sqrt(dist)
        distances.append((dist, y_i))

    distances.sort(key=lambda t: t[0])
    neighbors = distances[:k]

    s = sum(y for _, y in neighbors)
    return 1 if s >= 0 else -1

def print_help():
    help_text = """
Usage:
  classify.py [-h] [-f <folds>] [-k <kmax>] [-n <N>] [-d <mode>] [--knn-mode <impl>] <datasetname>

Options:
  -h                  Show this help message and exit
  -f <folds>          Number of folds for cross-validation (default: 5)
  -k <kmax>           Maximum k value; K = {1, ..., <kmax>} (default: 200)
  -n <N>              Use only the first N data points from the dataset
                      (intended for quick tests and debugging; default: use all data)
  -d <mode>           Fold generation mode:
                        0 = random partition of the dataset
                        1 = deterministic partition
                      (default: 0)
  --knn-mode <impl>   KNN implementation mode.
                        naive = naive KNN implementation (distance computation
                                followed by full sorting; intended for debugging)
                      (default: naive)

Description:
  This program performs binary classification using the k-nearest neighbors (kNN)
  method. For a given dataset, the data is split into l folds and cross-validation
  is used to select the optimal k from K = {1, ..., kmax}.

  Optionally, the dataset size can be limited to the first N samples using the
  -n parameter. This is useful for quick tests and debugging, as the naive kNN
  implementation has high computational cost.

  The naive KNN implementation computes all distances explicitly and determines
  the k nearest neighbors by full sorting. This mode is intended for debugging
  and validation purposes rather than large-scale experiments.

Dataset format:
  CSV file where each row has the form:
    y, x1, x2, ..., xd
  with y in {-1, 1} and xi in [-1, 1].

Example:
  classify.py data.csv -f 5 -k 50 -n 200 -d 0 --knn-mode naive
"""
    print(help_text)


def main():
    args = parse_args()

    if args.knn_mode == "naive":
        predict = knn_predict_naive
    else:
        print("Error: unsupported KNN mode")
        sys.exit(1)


    if args.h:
        print_help()
        sys.exit(0)

    if not args.datasetname:
        print("Error: No dataset name provided. Use -h for help.")
        sys.exit(1)

    # Read parameters
    l = args.f
    kmax = args.k
    mode = args.d

    print(f"Parameters: folds={l}, kmax={kmax}, fold_mode={mode}, n={args.n}, knn_mode={args.knn_mode}, dataset={args.datasetname}")
    K = list(range(1, kmax + 1))

    # Load dataset
    try:
        import csv
        data = []
        with open(args.datasetname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    label = int(row[0])
                    features = [float(v) for v in row[1:]]
                    data.append((label, features))
                except ValueError:
                    print("Warning: skipping malformed row", row)
        print(f"Dataset loaded successfully. Entries: {len(data)}") # Test    
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    if args.n is not None:
        data = data[:args.n]
        print(f"Using only first {len(data)} data points")

    folds = generate_folds(data, l, mode)
    


    print("Starting naive cross-validation...")
    print(f"Folds generated: {l}")
    print(f"First K values: {K[:10]}")
    print(f"Last K value: {K[-1]}")  # {K[:10]} ... {K[-1]}
    print(f"Fold generation mode: {mode}")
    print(f"KNN mode: {args.knn_mode}")
    print(f"Dataset: {args.datasetname}")
    print(f"n={args.n}")


    for i, f in enumerate(folds):
        print(f"Fold {i+1} length: {len(f)}")

    print("\nRunning naive kNN cross-validation...")

    results = {}  # k -> average error

    for k in K:
        fold_errors = []

        for i in range(l):
            test_fold = folds[i]
            train_folds = [item for j, f in enumerate(folds) if j != i for item in f]

            errors = 0
            for y_true, x in test_fold:
                y_pred = predict(x, train_folds, k)
                if y_pred != y_true:
                    errors += 1

            fold_error = errors / len(test_fold)
            fold_errors.append(fold_error)

        avg_error = sum(fold_errors) / l
        results[k] = avg_error

        print(f"k={k:3d}, avg error={avg_error:.4f}")


if __name__ == "__main__":
    main()
