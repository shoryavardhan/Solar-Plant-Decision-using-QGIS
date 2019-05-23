import csv

csv.register_dialect('myDialect',
delimiter = ';',
skipinitialspace=True)
p=[0,0,0,0,0]
q=[100000,100000,100000,100000,100000]
fields = ['name', 'branch', 'year', 'cgpa']
write=[0,0,0,0,0]
data=['name', 'branch', 'year', 'cgpa','af']
with open('/home/mannu/Documents/data.csv', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            if(int(row[i])>p[i-1]):
                p[i-1]=int(row[i])
            if(int(row[i])<q[i-1]):
                q[i-1]=int(row[i])
with open('/home/mannu/Documents/fizzy.csv', 'w') as ccvFile:
    reader = csv.reader(open('/home/mannu/Documents/data.csv', 'r'), dialect='myDialect')
    for row in reader:
        for i in range (1,6):
            write[i-1]=(int(row[i])-q[i-1])/(p[i-1]-q[i-1])
            ccvFile.write(str(write[i-1]))
            ccvFile.write(';')
        ccvFile.write('\n')
      
    
csvFile.close()
