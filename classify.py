#!/usr/bin/env python3
import sys
import os
import argparse
import random
from ball_tree import BallTree
import time
import matplotlib.pyplot as plt


def load_data(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip(): continue
                # Direkter Split am Komma schneller
                parts = line.split(',')
                # y (Label) ist das erste Element, der Rest sind x (Koordinaten)
                # map(float, ...) ist eine C-Funktion und schneller als List Comp
                data.append((float(parts[0]), list(map(float, parts[1:]))))
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        sys.exit(1)
    return data


def run_cross_validation(data, l_folds, K_max, mode):
    n = len(data)
    # Fold-Erzeugung wie gehabt
    if mode != 1: random.shuffle(data)
    folds = [data[i::l_folds] for i in range(l_folds)]

    # Speichere Fehlerraten pro Fold und k - Struktur: fold_errors[k] = [rate_fold1, rate_fold2, ...]
    fold_errors = {k: [] for k in range(1, K_max + 1)}

    for i in range(l_folds):
        test_set = folds[i]
        train_set = []
        for j in range(l_folds):
            if i != j: train_set.extend(folds[j])

        tree = BallTree(train_set)
        current_fold_counts = {k: 0 for k in range(1, K_max + 1)}

        for y_true, x_test in test_set:
            neighbors = tree.query(x_test, K_max)
            current_sum = 0
            for idx, label in enumerate(neighbors):
                k = idx + 1
                current_sum += label
                y_pred = 1.0 if current_sum >= 0 else -1.0
                if y_pred != y_true:
                    current_fold_counts[k] += 1

        # Fehlerrate für diesen Fold speichern
        for k in range(1, K_max + 1):
            fold_errors[k].append(current_fold_counts[k] / len(test_set))

    # Berechne Mittelwert R_D(k)
    avg_errors = {k: sum(fold_errors[k]) / l_folds for k in range(1, K_max + 1)}
    best_k = min(avg_errors, key=avg_errors.get)

    return best_k, avg_errors[best_k], avg_errors, fold_errors


if __name__ == "__main__":
    team_number = "2"
    parser = argparse.ArgumentParser(description="KNN Klassifikation")
    parser.add_argument("datasetname", help="Name des Datensatzes")
    parser.add_argument("-f", type=int, default=5, help="Anzahl Folds")
    parser.add_argument("-k", type=int, default=200, help="Maximales k")
    parser.add_argument("-d", type=int, choices=[0, 1], default=0, help="Modus (0=Zufall, 1=Det)")
    parser.add_argument("-n", type=int, default=None, help="Nur die ersten N Punkte verwenden")

    args = parser.parse_args()

    train_path = f"../classification-data/{args.datasetname}.train.csv"
    test_path = f"../classification-data/{args.datasetname}.test.csv"

    dataset_train = load_data(train_path)
    if args.n is not None:
        dataset_train = dataset_train[:args.n]
    dataset_test = load_data(test_path)
    #print(f"Trainingsdatensatz {args.datasetname} mit {len(dataset_train)} Punkten geladen.")
    #print(f"Testdatensatz {args.datasetname} mit {len(dataset_test)} Punkten geladen")

    # --- Trainingsphase ---
    start_train = time.perf_counter()
    best_k, best_error, cv_errors, fold_errors = run_cross_validation(dataset_train, args.f, args.k, args.d)
    elapsed_train = time.perf_counter() - start_train

    # --- Testphase ---
    start_test = time.perf_counter()
    final_tree = BallTree(dataset_train)
    test_errors = 0
    #m = len(dataset_test)
    predictions = []

    for y_true, x_test in dataset_test:
        neighbors = final_tree.query(x_test, best_k)
        y_pred = 1.0 if sum(neighbors) >= 0 else -1.0
        predictions.append(y_pred)
        if y_pred != y_true: test_errors += 1

    elapsed_test = time.perf_counter() - start_test
    test_error_rate = test_errors / len(dataset_test) if dataset_test else 0.0

    # --- Ausgabe ---
    print(f"Benötigte Trainingszeit: {elapsed_train:.1f} Sekunden")
    #print(f"Benötigte Testzeit: {elapsed_test:.1f} Sekunden")
    print(f"Bestes k*: {best_k} mit Fehlerrate R_D(k*): {best_error:.3f}")
    #print(f"R_D'(f_D): {test_error_rate:.3f}")

    output_dir = "../classification-results"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"team-{team_number}-{args.datasetname}"

    # 1. .result.csv
    with open(os.path.join(output_dir, f"{base_filename}.result.csv"), 'w') as f:
        for i in range(len(dataset_test)):
            y_p = int(predictions[i])
            coords = ", ".join(map(str, dataset_test[i][1]))
            f.write(f"{y_p}, {coords}\n")

    # 2. .log
    with open(os.path.join(output_dir, f"{base_filename}.log"), 'w') as f:
        for kv in range(1, args.k + 1):
            row = [str(kv)] + [f"{e:.6f}" for e in fold_errors[kv]] + [f"{cv_errors[kv]:.6f}"]
            f.write(", ".join(row) + "\n")

    # 3. .result.log
    with open(os.path.join(output_dir, f"{base_filename}.result.log"), 'w') as f:
        f.write(f"{elapsed_train:.2f}\n{elapsed_test:.2f}\n{best_k}\n{test_error_rate:.6f}\n")

    # 4. .log.png
    plt.figure(figsize=(10, 6))
    ks = sorted(cv_errors.keys())
    err_rates = [cv_errors[k] for k in ks]

    plt.plot(ks, err_rates, color='blue', linewidth=2, label='$\\tilde{R}_D(k)$ (Kreuzvalidierung)')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'$k^* = {best_k}$')
    plt.hlines(test_error_rate, xmin=1, xmax=args.k, color='green',
                   label=f"$R_{{D'}}(f_D) = {test_error_rate:.4f}$")

    plt.xlabel('K = {1, 2, ..., ' + str(args.k) + '}')
    plt.ylabel('Fehlerrate')
    plt.title(f'Fehlerkurve: {args.datasetname}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{base_filename}.log.png"))
    plt.close()

    # 5. Visualisierung für d = 2
    current_dimension = len(dataset_train[0][1])
    if current_dimension == 2:
        #print(f"Dimension d=2 erkannt. Erstelle Visualisierungen für {args.datasetname}...")


        # --- Hilfsfunktion für zufälliges Plotten ---
        def plot_shuffled(data, labels, title, filename):
            plt.figure(figsize=(8, 6))

            # 1. Daten und Labels zusammenführen
            # Wir bauen eine Liste aus Tupeln: (x1, x2, label)
            plot_data = []
            for i, (y, x) in enumerate(data):
                # Falls labels separat übergeben wurden (bei Vorhersagen), nutze diese
                current_label = labels[i] if labels is not None else y
                plot_data.append((x[0], x[1], current_label))

            # 2. Zufällig mischen (gegen Overplotting)
            random.shuffle(plot_data)

            # 3. Entpacken für Matplotlib
            x_vals = [p[0] for p in plot_data]
            y_vals = [p[1] for p in plot_data]
            l_vals = [p[2] for p in plot_data]

            # 4. Farben zuweisen
            # Blau für +1, Rot für -1
            colors = ['dodgerblue' if l == 1.0 else 'tomato' for l in l_vals]

            # 5. Plotten
            # alpha=0.6 sorgt zusätzlich für Transparenz
            plt.scatter(x_vals, y_vals, c=colors, marker='o', s=20, alpha=0.6, edgecolors='none')

            # Fake-Legende erstellen (da wir nur einen scatter-Call haben)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', label='Klasse +1', markersize=8),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='Klasse -1', markersize=8)
            ]

            plt.title(title)
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            plt.legend(handles=legend_elements)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()


        # --- A. Train Plot (Wahre Labels) ---
        # Wir übergeben None als Labels, da sie im dataset_train schon drin sind
        plot_shuffled(dataset_train, None,
                      f"Trainingsdaten D: {args.datasetname}",
                      f"{base_filename}.train.png")

        # --- B. Result Plot (Vorhersagen) ---
        # Predictions übergeben
        plot_shuffled(dataset_test, predictions,
                      f"Klassifikationsergebnis $f_D$ (k={best_k})",
                      f"{base_filename}.result.png")

    #else:
       # print(f"Dimension ist d={current_dimension}. Überspringe 2D-Visualisierung.")
