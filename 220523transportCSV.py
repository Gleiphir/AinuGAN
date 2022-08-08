import torch
import pandas as pd
import csv
import glob

f = open('220523FixedCSV.csv', 'w',encoding='UTF8')

# create the csv writer
writer = csv.writer(f)




Dir = "/**/{}*"

print( )

train = pd.read_csv('Databases_Combined2.csv',encoding='utf8',usecols= ['CategoryInJapanese','CategoryInAinu','ItemNumber'])
#train_tensor = torch.tensor(train.to_numpy())

#print(train.to_numpy())
i = 0

for JCate, ECate, Item in train.to_numpy():
    for filename in glob.glob(Dir.format(Item),recursive = True):
        writer.writerow( [filename, JCate, ECate ] )
        i = i+ 1
        print(i, filename)

f.close()