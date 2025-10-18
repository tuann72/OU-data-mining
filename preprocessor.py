import csv

with open("data.csv", "r", newline="") as infile, open("clean_data.csv", "w", newline="") as outfile:
    reader = csv.reader(infile, delimiter=";")
    writer = csv.writer(outfile)
    for row in reader:
        writer.writerow(row)