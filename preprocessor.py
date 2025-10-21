import csv
import re

numRows, drops, nonDrops = 0, 0, 0
with open("data.csv", "r", newline="") as infile, open("clean_data.csv", "w", newline="") as outfile:
    reader = csv.reader(infile, delimiter=",")
    writer = csv.writer(outfile)
    header = next(reader)
    header = [re.sub(r'[^A-Za-z ,]', '', col) for col in header]
    writer.writerow(header)
    for row in reader:
        numRows += 1
        if row[-1] == "Dropout":
            row[-1] = 0
            drops += 1
        else:
            row[-1] = 1
            nonDrops += 1
        writer.writerow(row)

# print(f"Dropout Rate: {drops/numRows}")
# print(f"Non-Dropout Rate: {nonDrops/numRows}")
# print(f"Total Rows: {numRows}")