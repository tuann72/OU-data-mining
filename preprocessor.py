import csv
import re

# numRows, drops, nonDrops = 0, 0, 0
remove_indices = {1, 2, 3, 5, 6, 12, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32}
keep_cols = {i for i in range(37)}
keep_cols = keep_cols - remove_indices

with open("data.csv", "r", newline="") as infile, open("clean_data.csv", "w", newline="") as outfile:
    reader = csv.reader(infile, delimiter=",")
    writer = csv.writer(outfile)
    header = next(reader)
    header = [re.sub(r'[^A-Za-z ,]', '', col) for col in header]
    new_header = [header[i] for i in keep_cols]
    for i, col in enumerate(new_header):
        print(f"{i}: {col}")
    writer.writerow(new_header)
    for row in reader:
        new_row = [row[i] for i in keep_cols]
        # numRows += 1
        if new_row[-1] == "Dropout":
            new_row[-1] = 1
            # drops += 1
        else:
            new_row[-1] = 0
            # nonDrops += 1
        writer.writerow(new_row)

# print(f"Dropout Rate: {drops/numRows}")
# print(f"Non-Dropout Rate: {nonDrops/numRows}")
# print(f"Total Rows: {numRows}")