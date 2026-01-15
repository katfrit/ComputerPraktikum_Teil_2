#!/usr/bin/env python3
import sys
import os
import argparse
import random
from ball_tree import BallTree, distance
import time
import matplotlib.pyplot as plt



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

    # Automatische Pfadkonstruktion
    train_path = f"../classification-data/{args.datasetname}.train.csv"
    test_path = f"../classification-data/{args.datasetname}.test.csv"

    # Daten laden und optional auf N Punkte kürzen
    dataset_train = load_data(train_path)
    if args.n is not None:
        dataset_train = dataset_train[:args.n]
    dataset_test = load_data(test_path)
    print(f"Trainingsdatensatz {args.datasetname} mit {len(dataset_train)} Punkten geladen.")
    print(f"Testdatensatz {args.datasetname} mit {len(dataset_test)} Punkten geladen")

    # --- Trainingsphase ---
    start_train = time.time()
    best_k, best_error, cv_errors = run_cross_validation(dataset_train, args.f, args.k, args.d)
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

    output_dir = "../classification-results"
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"team-{team_number}-{args.datasetname}"

    # 1. Result-Datei
    res_file = os.path.join(output_dir, f"{base_filename}.result.csv")
    with open(res_file, 'w') as f:
        for p in predictions:
            f.write(f"{int(p)}\n")
    print(f"Vorhersagen gespeichert in: {res_file}")

    # 2. Klassifikations-Logdatei (.log)
    log_file = os.path.join(output_dir, f"{base_filename}.log")
    with open(log_file, 'w') as f:
        # Header (optional, aber gut für Lesbarkeit, falls erlaubt)
        # Falls striktes Format ohne Header gefordert ist, die nächste Zeile löschen:
        f.write("# k\terror_rate\n")

        n_train = len(dataset_train)
        for k_val in range(1, args.k + 1):
            # Fehlerquote = Anzahl Fehler / Anzahl Trainingsdaten
            rate = cv_errors[k_val] / n_train
            f.write(f"{k_val}\t{rate:.6f}\n")

    print(f"Fehlertabelle gespeichert in: {log_file}")

    # 3. Zusammenfassungs-Logdatei (.result.log)
    summary_log = os.path.join(output_dir, f"{base_filename}.result.log")
    with open(summary_log, 'w') as f:
        f.write(f"{elapsed_train:.2f}\n")  # Zeile 1: Trainingszeit
        f.write(f"{elapsed_test:.2f}\n")  # Zeile 2: Testzeit
        f.write(f"{best_k}\n")  # Zeile 3: k*
        f.write(f"{test_error_rate:.6f}\n")  # Zeile 4: Test-Fehlerrate

    # 4. Grafik der Fehlerkurve (log.png)
    plt.figure(figsize=(10, 6))
    ks = list(cv_errors.keys())
    errs = [cv_errors[k]/len(dataset_train) for k in ks]

    plt.plot(ks, errs, label='Risiko $\\tilde{R}_D(k)$')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Bestes k* = {best_k}')
    # Markierung für den Testfehler (als horizontaler Strich oder Punkt)
    plt.hlines(test_error_rate, xmin=1, xmax=args.k, color='g',
               label=f"Testfehler $R_{{D'}}(f_D)$ = {test_error_rate:.3f}")

    plt.xlabel('k')
    plt.ylabel('Fehlerrate')
    plt.title(f'Kreuzvalidierung für {args.datasetname}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{base_filename}.log.png"))
    plt.close()
