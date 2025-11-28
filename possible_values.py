import csv

# categorical_indices = {i for i in range(37)}
# numerical_indices = {6, 12, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
# categorical_indices = categorical_indices - numerical_indices

categorical_indices = {i for i in range(21)}
numerical_indices = {13, 15, 16, 17, 18, 19}
categorical_indices = categorical_indices - numerical_indices

possible_values = {i: set() for i in categorical_indices}

with open("clean_data.csv", 'r') as f:
    reader = csv.reader(f, delimiter=",")
    next(reader)
    for row in reader:
        for i in categorical_indices:
            if row[i] not in possible_values[i]:
                possible_values[i].add(int(row[i]))

print("{")
for k, v in possible_values.items():
    print(f"{k}: {sorted(v)},")
print("}")