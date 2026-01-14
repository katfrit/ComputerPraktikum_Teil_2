#!/usr/bin/env python3
import sys
import argparse
import random
from ball_tree import BallTree, distance
import time



def load_data(filename):
    # Daten aus CSV laden: Label y ist erster Wert, danach die Features x
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.replace(',', ' ').split()
                if not parts: continue
                # y_i ist erstes Element
                y = float(parts[0])
                # Alle weiteren Elemente sind Features/Merkmale
                x = [float(val) for val in parts[1:]]
                data.append((y, x))
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        sys.exit(1)
    return data


def predict_naive(target_x, train_data, k):
    # Naive KNN-Implementierung zum Vgl oder Validierung
    dists = [(distance(target_x, x), y) for y, x in train_data]
    dists.sort()
    k_nearest = dists[:k]
    # Signum der Summe: >=0 wird zu 1.0, <0 zu -1.0
    avg_y = sum(y for d, y in k_nearest)
    return 1.0 if avg_y >= 0 else -1.0


def run_cross_validation(data, l_folds, K_max, mode):   # führt l-fache Kreuzvalidierung zur Besitmmung von k* durch
    n = len(data)
    if mode == 1:  # Deterministisch (kein Shuffle)
        pass
    else:  # Zufällig
        random.shuffle(data)

    # Erstellt Folds mittels Slicing: D_j = {x_i : i mod L = j}
    folds = [data[i::l_folds] for i in range(l_folds)]
    # Dictionary zum Zählen der Klassifikatinosfehler für jedes k
    errors = {k: 0 for k in range(1, K_max + 1)}

    for i in range(l_folds):
        test_set = folds[i]
        # Trainingsmenge ist Vereiningung aller anderen Folds
        train_set = []
        for j in range(l_folds):
            if i != j: train_set.extend(folds[j])

        # Baumaufbau: nur einmal pro Fold
        tree = BallTree(train_set)

        for y_true, x_test in test_set:
            # Query liefert K_max nächsten Nachbarn vorsortiert
            neighbors = tree.query(x_test, K_max)

            # Evaluiere alle k von 1 bis K_mx gleichzeitig für diesen Testpunk
            for k in range(1, K_max + 1):
                k_labels = neighbors[:k]
                y_pred = 1.0 if sum(k_labels) >= 0 else -1.0
                if y_pred != y_true:
                    errors[k] += 1

    # Bestimme k mit geringster Fehlerrate
    best_k = min(errors, key=errors.get)
    min_error_rate = errors[best_k] / n
    return best_k, min_error_rate




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Klassifikation")
    parser.add_argument("datasetname", help="Name des Datensatzes")
    parser.add_argument("-f", type=int, default=5, help="Anzahl Folds")
    parser.add_argument("-k", type=int, default=200, help="Maximales k")
    parser.add_argument("-d", type=int, choices=[0, 1], default=1, help="Modus (0=Zufall, 1=Det)")
    parser.add_argument("-n", type=int, default=None, help="Nur die ersten N Punkte verwenden")

    args = parser.parse_args()

    # Automatische PFadkonstruktion
    train_path = f"../classification-data/{args.datasetname}.train.csv"
    test_path = f"../classification-data/{args.datasetname}.test.csv"

    # Daten laden und optional auf N Punkte kürzen
    dataset_train = load_data(train_path)
    if args.n is not None:
        dataset_train = dataset_train[:args.n]
    dataset_test = load_data(test_path)
    print(f"Trainingsdatensatz {train_path} geladen. {len(dataset_train)} Punkte.")
    print(f"Testdatensatz {test_path} geladen. {len(dataset_test)} Punkte.")

    # --- Trainingsphase ---
    start_train = time.time()
    best_k, best_error = run_cross_validation(dataset_train, args.f, args.k, args.d)
    end_train = time.time()
    elapsed_train = end_train - start_train

    # --- Testphase ---
    start_test = time.time()

    # Finales Modell f_D wird auf vollständigen Trainingsset trainiert
    final_tree = BallTree(dataset_train)

    test_errors = 0
    m = len(dataset_test)
    predictions = []

    for y_true, x_test in dataset_test:
        # Suche die k* Nachbarn
        neighbors = final_tree.query(x_test, best_k)
        y_pred = 1.0 if sum(neighbors) >= 0 else -1.0
        predictions.append(y_pred)

        if y_pred != y_true:
            test_errors += 1

    end_test = time.time()
    elapsed_test = end_test - start_test
    test_error_rate = test_errors / m if m>0 else 0.0


    # Ausggabe
    print(f"Benötigte Trainingszeit: {elapsed_train:.1f} Sekunden")
    print(f"Benötigte Testzeit: {elapsed_test:.1f} Sekunden")
    print(f"Bestes k*: {best_k} mit Fehlerrate R_D(k*): {best_error:.3f}")
    print(f"R_D'(f_D): {test_error_rate:.3f}")
