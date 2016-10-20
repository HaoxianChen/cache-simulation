import argparse
import matplotlib.pyplot as plt
import numpy as np

FILENAME1 = 'logs/curve-1-80-1-d1-0.log'
label1 = 'd1-0'
FILENAME2 = 'logs/curve-1-80-1-d2-d1-0.log'
label2 = 'd2-d1-0'
FILENAME3 = 'logs/curve-1-80-1-0.log'
label3 = '0'
FILENAME4 = 'logs/curve-1-80-1-d2-0.log'
label4 = 'd2-0'
FILENAME5 = 'logs/curve-1-80-1-d1-d2-0.log'
label5 = 'd1-d2-0'

f = open(FILENAME1)
data_len = int(f.readline())

line = f.readline()
x = np.fromstring(line,dtype=float,sep=' ')
line = f.readline()
y = np.fromstring(line,dtype=float,sep=' ')
plt.figure()
plt.plot(x,y,label=label1)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

f = open(FILENAME2)
data_len = int(f.readline())

line = f.readline()
x = np.fromstring(line,dtype=float,sep=' ')
line = f.readline()
y = np.fromstring(line,dtype=float,sep=' ')
plt.plot(x,y,label=label2)

f = open(FILENAME3)
data_len = int(f.readline())

line = f.readline()
x = np.fromstring(line,dtype=float,sep=' ')
line = f.readline()
y = np.fromstring(line,dtype=float,sep=' ')
plt.plot(x,y,label=label3)

f = open(FILENAME4)
data_len = int(f.readline())

line = f.readline()
x = np.fromstring(line,dtype=float,sep=' ')
line = f.readline()
y = np.fromstring(line,dtype=float,sep=' ')
plt.plot(x,y,label=label4)

f = open(FILENAME5)
data_len = int(f.readline())

line = f.readline()
x = np.fromstring(line,dtype=float,sep=' ')
line = f.readline()
y = np.fromstring(line,dtype=float,sep=' ')
plt.plot(x,y,label=label5)

#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(loc='best', fontsize=12)
plt.xlabel('cache size')
plt.ylabel('miss rate')
plt.axis([0,64,0,1])
plt.show()
