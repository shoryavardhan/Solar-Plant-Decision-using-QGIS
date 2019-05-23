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
    B=([[1,0.5,0.5,0.5,0.5],
        [2,1,2,2,2],
        [2,0.5,1,1,1],
        [2,0.5,1,1,1],
        [2,0.5,1,1,1]])
    for i in range(0, n):
        for j in range(i, n):
            if i == j:
                scale = 1
            else:
                scale = float(B[i][j])
            A[i][j] = scale
            A[j][i] = float(1/scale)
    #A=np.zeros((5,5))
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
    criterions =["population","PVOUT","GHI","DNI","GTI"]
    #load_units('/home/mannu/Desktop/myproject/criterions.txt')
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
with open('/home/mannu/Desktop/myproject/minordata.csv', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            if(int(row[i])>p[i-1]):
                p[i-1]=int(row[i])
            if(int(row[i])<q[i-1]):
                q[i-1]=int(row[i])
                
with open('/home/mannu/Desktop/myproject/fizzy.csv', 'w') as ccvFile:
    reader = csv.reader(open('/home/mannu/Desktop/myproject/minordata.csv', 'r'), dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            write[0]=row[0]
            write[i]=(int(row[i])-q[i-1])/(p[i-1]-q[i-1])
            ccvFile.write(str(write[i-1]))
            ccvFile.write(';')
        ccvFile.write(row[6])
        ccvFile.write(';')
        ccvFile.write(row[7])
        ccvFile.write('\n')          
csvFile.close()
weight=ahp()
#[0.25281222248342056, 0.2528122224834205, 0.1765911401326551, 0.14313445420535845, 0.17464996069514543]
print("The final Weight:")
print(weight)
i=0
r=0
with open('/home/mannu/Desktop/myproject/final.csv', 'w') as ccfFile:
    reader = csv.reader(open('/home/mannu/Desktop/myproject/fizzy.csv', 'r'), dialect='myDialect')
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
        ccfFile.write(';')
        ccfFile.write(row[5])
        ccfFile.write(';')
        ccfFile.write(row[6])
        ccfFile.write('\n')
ccfFile.close()
import pandas as pd 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#making data frame from csv file 
data=pd.read_csv("/home/mannu/Desktop/myproject/final.csv",sep=';',names=["Names","SI","Latt","Long"] )
#sorting data frame by Team and then By names 
reader = csv.reader(open('/home/mannu/Desktop/myproject/fizzy.csv', 'r'), dialect='myDialect')
data.sort_values(by=["SI"], axis=0,ascending=False, inplace=True)
with open('/home/mannu/Desktop/myproject/finalish.csv', 'w') as ccfFile:
    #ccfFile.write(str(data))
    data.to_csv('/home/mannu/Desktop/myproject/finalish.csv', sep=';', encoding='utf-8')
    print(data)
k=0
with open('/home/mannu/Desktop/myproject/topten.csv', 'w') as ccvfile1:
    with open('/home/mannu/Desktop/myproject/tentotwo.csv', 'w') as ccvfile2:
        with open('/home/mannu/Desktop/myproject/twotoend.csv', 'w') as ccvfile3:
            reader = csv.reader(open('/home/mannu/Desktop/myproject/finalish.csv', 'r'), dialect='myDialect')
            ccvfile2.write("Names;SI;Latt;Long")
            ccvfile2.write('\n')
            ccvfile3.write("Names;SI;Latt;Long")
            ccvfile3.write('\n')
            for row in reader:
                if k<30:
                    #print("0",k)
                    ccvfile1.write(row[1])
                    ccvfile1.write(';')
                    ccvfile1.write(row[2])
                    ccvfile1.write(';')
                    ccvfile1.write(row[3])
                    ccvfile1.write(';')
                    ccvfile1.write(row[4])
                    ccvfile1.write('\n')
                elif k>=30:
                    if k<70:
                        #print("10",k)
                        ccvfile2.write(row[1])
                        ccvfile2.write(';')
                        ccvfile2.write(row[2])
                        ccvfile2.write(';')
                        ccvfile2.write(row[3])
                        ccvfile2.write(';')
                        ccvfile2.write(row[4])
                        ccvfile2.write('\n')
                    elif k>=70:
                        #print("20",k)
                        ccvfile3.write(row[1])
                        ccvfile3.write(';')
                        ccvfile3.write(row[2])
                        ccvfile3.write(';')
                        ccvfile3.write(row[3])
                        ccvfile3.write(';')
                        ccvfile3.write(row[4])
                        ccvfile3.write('\n')
                k=k+1        
from PyQt5 import *
uri1 = "file:///home/mannu/Desktop/myproject/topten.csv?type=csv&delimiter=;&quote=&escape=&detectTypes=yes&xField=Long&yField=Latt&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no&yField=3"
vlayer= QgsVectorLayer(uri1,'Most_Recommended','delimitedtext')
pal_layer = QgsPalLayerSettings()
pal_layer.fieldName = 'Names'
pal_layer.enabled = True
pal_layer.placement = QgsPalLayerSettings.Line
labels = QgsVectorLayerSimpleLabeling(pal_layer)
vlayer.setLabeling(labels)
vlayer.setLabelsEnabled(True)
vlayer.triggerRepaint()
QgsProject.instance().addMapLayer(vlayer)
uri2 = "file:///home/mannu/Desktop/myproject/tentotwo.csv?type=csv&delimiter=;&quote=&escape=&detectTypes=yes&xField=Long&yField=Latt&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no&yField=3"
vlayer2 = QgsVectorLayer(uri2,'Recommended','delimitedtext')
pal_layer = QgsPalLayerSettings()
pal_layer.fieldName = 'Names'
pal_layer.enabled = True
pal_layer.placement = QgsPalLayerSettings.Line
labels = QgsVectorLayerSimpleLabeling(pal_layer)
vlayer2.setLabeling(labels)
vlayer2.setLabelsEnabled(True)
vlayer2.triggerRepaint()
QgsProject.instance().addMapLayer(vlayer2)
uri3 = "file:///home/mannu/Desktop/myproject/twotoend.csv?type=csv&delimiter=;&quote=&escape=&detectTypes=yes&xField=Long&yField=Latt&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no&yField=3"
vlayer3 = QgsVectorLayer(uri3,'Less_Recommended','delimitedtext')
pal_layer = QgsPalLayerSettings()
pal_layer.fieldName = 'Names'
pal_layer.enabled = True
pal_layer.placement = QgsPalLayerSettings.Line
labels = QgsVectorLayerSimpleLabeling(pal_layer)
vlayer3.setLabeling(labels)
vlayer3.setLabelsEnabled(True)
vlayer3.triggerRepaint()
QgsProject.instance().addMapLayer(vlayer3)
'''layer=None
for lyr in QgsProject.instance().mapLayers().values():
    if lyr.name()=="Names":
        layer=lyr[0.25281222248342056, 0.2528122224834205, 0.1765911401326551, 0.14313445420535845, 0.17464996069514543]
        break
iface.mapCanvas().setSelectionColor(QColor("red"))
expr=QgsProject.instance().addMapLayer(vlayer)
''''''
uri1 = "file:///home/mannu/Desktop/myproject/topten.csv?type=csv&delimiter=;&quote=&escape=&detectTypes=yes&xField=Long&yField=Latt&crs=EPSG:4326&spatialIndex=no&subsetIndex=no&watchFile=no&yField=3"
vlayer = QgsVectorLayer(uri1,'MultiPoints','delimitedtext')
QgsProject.instance().addMapLayer(vlayer)
'''
