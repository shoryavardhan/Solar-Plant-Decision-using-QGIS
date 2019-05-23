import numpy as np
from fractions import Fraction

def load_units(file):
    units = []
    with open(file) as fp:
        for line in fp:
            units.append(line.split()[0]) # Used to deal with '\n'
        return units

def cross_compare(units):
    n = len(units)
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            if i == j:
                scale = 1
            else:
                scale = float(Fraction(input(units[i]+' to '+units[j]+':')))
            A[i][j] = scale
            A[j][i] = float(1/scale)
    return A

def get_weight(A):
    n = A.shape[0]
    d=[0,0,0,0,0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = max(e_vals)
    w = e_vecs[:, e_vals.argmax()]
    w = w / np.sum(w) # Normalization
    # Consistency Checking
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri[n]
    for i in range (0,5):
        d[i]=float(w[i])
    print("The normalized eigen vector:")
    print(d)
    print('CR = %f'%cr)
    if cr >= 0.1:
        print("Failed in Consistency check.")
        exit = input("Enter 'q' to quit.")
        raise
    return d, float(cr)

def ahp():
    criterions = load_units('criterions.txt')
    n2 = len(criterions)
    A = cross_compare(criterions)
    print("The matrix A")
    print(A)
    print()
    W2, cr2 = get_weight(A)
    print("######################")
    return(W2)
    
import csv

csv.register_dialect('myDialect',
delimiter = ';',
skipinitialspace=True)
p=[0,0,0,0,0]
weight=[0,0,0,0,0]
q=[100000,100000,100000,100000,100000]
fields = ['name', 'branch', 'year', 'cgpa']
write=["iwier",0,0,0,0,0]
data=['name', 'branch', 'year', 'cgpa','af']
with open('data.csv', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            if(int(row[i])>p[i-1]):
                p[i-1]=int(row[i])
            if(int(row[i])<q[i-1]):
                q[i-1]=int(row[i])
                
with open('fizzy.csv', 'w') as ccvFile:
    reader = csv.reader(open('data.csv', 'r'), dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            write[0]=row[0]
            write[i]=(int(row[i])-q[i-1])/(p[i-1]-q[i-1])
            ccvFile.write(str(write[i-1]))
            ccvFile.write(';')
        ccvFile.write('\n')          
csvFile.close()
weight=ahp()
print("The final Weight:")
print(weight)
i=0
r=0
with open('final.csv', 'w') as ccfFile:
    reader = csv.reader(open('fizzy.csv', 'r'), dialect='myDialect')
    for row in reader:
        si=0.0
        for i in range (1,5):
            #print(row[i])
            floatRow = float(row[i])
            addition = floatRow * weight[i]
            si=si+addition
        stt=str(si)
        ccfFile.write(row[0])
        ccfFile.write(';')
        ccfFile.write(stt)
        ccfFile.write('\n')          
ccfFile.close()
'''with open('final.csv',newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('final.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Names','SI'])
    w.writerows(data)
'''
import pandas as pd 

#making data frame from csv file 
data=pd.read_csv("final.csv",sep=';',names=["Names","SI"] )
#sorting data frame by Team and then By names 
data.sort_values(by=["SI"], axis=0,ascending=True, inplace=True)
with open('finalish.csv', 'w') as ccfFile:
    ccfFile.write(str(data))
