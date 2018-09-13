import csv
import numpy as np
from pathlib import Path


def convert(filename: Path):
    print('begin to convert CSV file...')
    reader = csv.reader(open(filename/'results.csv', 'rU'))
    results = [['ImageNumber','ObjectID','ParentObjectID']]

    for index, row in enumerate(reader):
        if index != 0:
            result = []
            result.append(row[0][11:14])
            result.append(row[1])
            parent = row[1][:-1]
            if parent == '':
                parent = '0'
            result.append(parent)
            results.append(result)
    np.savetxt(filename/'converted_results.csv', results, fmt='%s', delimiter=',')
    print('Finish conversion!')
