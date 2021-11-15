import numpy as np
import csv

file = open("glass.data")
names = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
type_labels = ["building_windows_float_processed", "building_windows_non_float_processed",
               "vehicle_windows_float_processed", "vehicle_windows_non_float_processed", "containers", "tableware",
               "headlamps"]
type_labels = ["build_window_float", "build_window_non_float",
               "vehicle_window_float", "vehicle_window_non_float", "containers", "tableware",
               "headlamps"]

data = [names]
for line in file:
    data_raw = line.split(",")
    data_raw[-1] = type_labels[int(data_raw[-1])-1]
    data.append(data_raw)

with open('glass.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows(data)

print("")