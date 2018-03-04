import csv

with open('../data/ratings.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        print(row[0])
        print(row[0],row[1])
        break
print readCSV

