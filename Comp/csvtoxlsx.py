# transforms the csv files in the current directory to xlsx files

import os
import csv
import openpyxl

for csvFilename in os.listdir('.'):
    if not csvFilename.endswith('.csv'):
        continue
    print('Transforming ' + csvFilename + ' to xlsx...')
    wb = openpyxl.Workbook()
    sheet = wb.active
    with open(csvFilename) as f:
        reader = csv.reader(f)
        for row in reader:
            sheet.append(row)
    wb.save(csvFilename[:-4] + '.xlsx')
    print('Done')
print('All files transformed.') 

