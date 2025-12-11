import sys 
import csv

data = []
filename = sys.argv[1]

with open(filename, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)

    for row in reader:
        data.append(row)


print(data[50], len(data))

