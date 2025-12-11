#!/usr/bin/env python3
import argparse
import sys
import random

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

def print_help():
    help_text = """
Usage: classify.py [-h] [-f <folds>] [-k <kmax>] [-d <mode>] <datasetname>

Options:
  -h              Show this help message and exit
  -f <folds>      Number of folds (default: 5)
  -k <kmax>       Maximum k value (default: 200)
  -d <mode>       Mode for fold generation: 0=random, 1=deterministic (default: 0)

Description:
  This program reads optional parameters, sets l=<folds> and K={1,...,<kmax>}, and
  generates dataset folds according to the chosen mode.
    """
    print(help_text)


def main():
    args = parse_args()

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

    print(f"Parameters: folds={l}, kmax={kmax}, mode={mode}, dataset={args.datasetname}") # Test

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

    folds = generate_folds(data, l, mode)

    print(f"Folds generated: {l}")
    print(f"First K values: {K[:10]}")
    print(f"Last K value: {K[-1]}") # {K[:10]} ... {K[-1]}")
    print(f"Mode: {mode}")
    print(f"Dataset: {args.datasetname}")

    for i, f in enumerate(folds):
        print(f"Fold {i+1} length: {len(f)}")


if __name__ == "__main__":
    main()
