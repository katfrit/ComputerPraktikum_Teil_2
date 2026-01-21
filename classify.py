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
                parts = line.replace(',', ' ').split()
                if not parts: continue
                y = float(parts[0])
                x = [float(val) for val in parts[1:]]
                data.append((y, x))
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        sys.exit(1)
    return data


def run_cross_validation(data, l_folds, K_max, mode):
    n = len(data)
    if mode != 1: random.shuffle(data)
    folds = [data[i::l_folds] for i in range(l_folds)]

    # Adaptive Optimierung für gr0ße Datensätze (Sampling der k-Werte)
    use_step = n > 30000
    k_to_check = set(range(1, K_max + 1))
    if use_step:
        # feingranular bei kleinen k, grober bei großen k (da stabiler)
        k_to_check = set(list(range(1, 21)) + list(range(25, K_max + 1, 5)))
        if K_max not in k_to_check: k_to_check.add(K_max)

    errors = {k: 0 for k in range(1, K_max + 1)}

    for i in range(l_folds):
        test_set = folds[i]
        train_set = []
        for j in range(l_folds):
            if i != j: train_set.extend(folds[j])

        tree = BallTree(train_set)

        for y_true, x_test in test_set:
            # alle k_max Nachbarn einmalig abholen
            neighbors = tree.query(x_test, K_max)
            # Running Sum Optimierung: Berechnet alle k in O(k_max) statt O(K_max^2)
            current_sum = 0
            for idx, label in enumerate(neighbors):
                k = idx + 1
                current_sum += label
                if k in k_to_check:
                    # Klassifikationsunterscheidung
                    y_pred = 1.0 if current_sum >= 0 else -1.0
                    if y_pred != y_true:
                        errors[k] += 1

    if use_step:
        # Lineare Interpolation der Fehlerraten für log-Datei
        s_ks = sorted(list(k_to_check))
        for idx in range(len(s_ks) - 1):
            for fill in range(s_ks[idx] + 1, s_ks[idx + 1]):
                errors[fill] = errors[s_ks[idx]]

    best_k = min(errors, key=errors.get)
    min_error_rate = errors[best_k] / n
    return best_k, min_error_rate, errors


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
    print(f"Trainingsdatensatz {args.datasetname} mit {len(dataset_train)} Punkten geladen.")
    print(f"Testdatensatz {args.datasetname} mit {len(dataset_test)} Punkten geladen")

    # --- Trainingsphase ---
    start_train = time.perf_counter()
    best_k, best_error, cv_errors = run_cross_validation(dataset_train, args.f, args.k, args.d)
    elapsed_train = time.perf_counter() - start_train

    # --- Testphase ---
    start_test = time.perf_counter()
    final_tree = BallTree(dataset_train)
    test_errors = 0
    m = len(dataset_test)
    predictions = []

    for y_true, x_test in dataset_test:
        neighbors = final_tree.query(x_test, best_k)
        y_pred = 1.0 if sum(neighbors) >= 0 else -1.0
        predictions.append(y_pred)
        if y_pred != y_true:
            test_errors += 1

    elapsed_test = time.perf_counter() - start_test
    test_error_rate = test_errors / m if m > 0 else 0.0

    # --- Ausgabe ---
    print(f"Benötigte Trainingszeit: {elapsed_train:.1f} Sekunden")
    print(f"Benötigte Testzeit: {elapsed_test:.1f} Sekunden")
    print(f"Bestes k*: {best_k} mit Fehlerrate R_D(k*): {best_error:.3f}")
    print(f"R_D'(f_D): {test_error_rate:.3f}")

    output_dir = "../classification-results"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"team-{team_number}-{args.datasetname}"

    # 1. Result-Datei
    res_file = os.path.join(output_dir, f"{base_filename}.result.csv")
    with open(res_file, 'w') as f:
        for p in predictions: f.write(f"{int(p)}\n")
    print(f"Vorhersagen gespeichert in: {res_file}")

    # 2. Klassifikations-Logdatei (.log)
    log_file = os.path.join(output_dir, f"{base_filename}.log")
    with open(log_file, 'w') as f:
        n_train = len(dataset_train)
        for k_val in range(1, args.k + 1):
            rate = cv_errors[k_val] / n_train
            f.write(f"{k_val}\t{rate:.6f}\n")
    print(f"Fehlertabelle gespeichert in: {log_file}")

    # 3. Zusammenfassungs-Logdatei (.result.log)
    summary_log = os.path.join(output_dir, f"{base_filename}.result.log")
    with open(summary_log, 'w') as f:
        f.write(f"{elapsed_train:.2f}\n")
        f.write(f"{elapsed_test:.2f}\n")
        f.write(f"{best_k}\n")
        f.write(f"{test_error_rate:.6f}\n")

    # 4. Grafik der Fehlerkurve (log.png)
    plt.figure(figsize=(10, 6))
    ks = sorted(list(cv_errors.keys()))
    errs = [cv_errors[k] / len(dataset_train) for k in ks]
    plt.plot(ks, errs, label='Risiko $\\tilde{R}_D(k)$')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Bestes k* = {best_k}')
    plt.hlines(test_error_rate, xmin=1, xmax=args.k, color='g',
               label=f"Testfehler $R_{{D'}}(f_D)$ = {test_error_rate:.3f}")
    plt.xlabel('k')
    plt.ylabel('Fehlerrate')
    plt.title(f'Kreuzvalidierung für {args.datasetname}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{base_filename}.log.png"))
    plt.close()

    # 5. Visualisierung für d = 2
    current_dimension = len(dataset_train[0][1])
    if current_dimension == 2:
        print(f"Dimension d=2 erkannt. Erstelle Visualisierungen für {args.datasetname}...")
        # Train Plot
        plt.figure(figsize=(8, 6))
        x_pos_1 = [p[1][0] for p in dataset_train if p[0] == 1.0]
        x_pos_2 = [p[1][1] for p in dataset_train if p[0] == 1.0]
        x_neg_1 = [p[1][0] for p in dataset_train if p[0] == -1.0]
        x_neg_2 = [p[1][1] for p in dataset_train if p[0] == -1.0]
        plt.scatter(x_pos_1, x_pos_2, c='blue', marker='o', label='Klasse +1 (wahr)', s=20, alpha=0.7)
        plt.scatter(x_neg_1, x_neg_2, c='red', marker='x', label='Klasse -1 (wahr)', s=20, alpha=0.7)
        plt.title(f"Trainingsdaten D: {args.datasetname}")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_dir, f"{base_filename}.train.png"))
        plt.close()

        # Result Plot
        plt.figure(figsize=(8, 6))
        x_pred_pos_1 = [dataset_test[i][1][0] for i in range(len(dataset_test)) if predictions[i] == 1.0]
        x_pred_pos_2 = [dataset_test[i][1][1] for i in range(len(dataset_test)) if predictions[i] == 1.0]
        x_pred_neg_1 = [dataset_test[i][1][0] for i in range(len(dataset_test)) if predictions[i] == -1.0]
        x_pred_neg_2 = [dataset_test[i][1][1] for i in range(len(dataset_test)) if predictions[i] == -1.0]
        plt.scatter(x_pred_pos_1, x_pred_pos_2, c='dodgerblue', marker='o', label='Vorhersage +1', s=20, alpha=0.6)
        plt.scatter(x_pred_neg_1, x_pred_neg_2, c='tomato', marker='x', label='Vorhersage -1', s=20, alpha=0.6)
        plt.title(f"Klassifikationsergebnis $f_D$ (k={best_k})")
        plt.xlabel("$x_1'$")
        plt.ylabel("$x_2'$")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_dir, f"{base_filename}.result.png"))
        plt.close()
    else:
        print(f"Dimension ist d={current_dimension}. Überspringe 2D-Visualisierung.")
