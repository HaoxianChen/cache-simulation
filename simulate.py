import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse
from cache import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('s0', type=int, help='bgin cache size')
parser.add_argument('s1', type=int, help='end cache size')
parser.add_argument("-s","--step", type=int, help='simulation step in cache\
sizes', default = 4)
args = parser.parse_args()

# simulation parameters
x1 = 8 # size of array 1
x2 = 16 # size of array 2
x3 = 32 # size of array 3
p1 = 0.4 # probability of accessing array 1
p2 = 0.4 # probability of accessing array 2
p3 = 1 - p1 - p2 # probability of accessing array 3
ed = x1 + x2 + x3 # overall expected reuse distance: the working set size, also equals
assert p3 > 0
iterate_times = 1000

d1 = x1/p1
d2 = x2/p2
d3 = x3/p3
assert MAX_AGE > max([d1,d2,d3])

test_spec = '-'+str(x1) + '-' + str(p1) + '-' + str(x2) + '-' + str(p2) + '-' + str(x3)
# policy: give as values of each age
policy_name = '0-'
value = np.ones(MAX_AGE,dtype=float)
value[0] = 0
# value[d1] = -2
# value[d2] = -1
array1= range(0,x1)
array2 = range(x1,x1+x2)
array3 = range(x1+x2,x1+x2+x3)
cache_size = range(args.s0,args.s1,args.step)
miss_rate = np.zeros(len(cache_size),)

FILENAME = 'logs/' + policy_name + str(args.s0) + test_spec + '.log'
f = open(FILENAME,'w')
f.write(str(len(cache_size))+'\n')

def weighted_choice(weights):
    totals = []
    running_total = 0
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total

    for i,total in enumerate(totals):
        if rnd < total:
            return i

for j,s in enumerate(cache_size):
    for pid,p in enumerate(policies):
        if pid !=0: continue
        cache = Cache(s,p)
        a_counter1 = 0
        a_counter2 = 0
        a_counter3 = 0
        for i in range(iterate_times):
            # simulate data access
            if i % 10 < 10*p1:
                k = 0
            else:
                k = 1
#            elif i % 10 >= 10*p1 and i % 10 < 10*(p1+p2):
#                k = 1
#            elif i % 10 >= 10*(p1+p2): 
#                k = 2
                
            if k == 0:
                addr = array1[a_counter1 % len(array1)]
                a_counter1 += 1
            elif k == 1:
                addr = array2[a_counter2 % len(array2)]
                a_counter2 += 1
            elif k == 2:
                addr = array3[a_counter3 % len(array3)]
                a_counter3 += 1
                
            cache.lookup(addr)
        miss_rate[j] = 1-cache.get_hit_rate()
        print 'policy ' + str(pid) + ' miss rate:' + str(miss_rate[j])

        # log hit age and eviction age distribution
        event_sum = sum(cache.get_hit_ages()) + sum(cache.get_evict_ages())
        hit_distribution = [float(a)/event_sum for a in cache.get_hit_ages()]
        evict_distribution = [float(a)/event_sum for a in cache.get_evict_ages()]
        f.write(str(s)+'\n')
        for a,h in enumerate(hit_distribution):
            #if h > 0: print 'hit rate at age ' + str(a) + '= ' + str(h)
            f.write(str(a)+' ')
        f.write('\n')
        for a,e in enumerate(evict_distribution):
            #if e > 0: print 'evict rate at age ' + str(a) + '= ' + str(e)
            f.write(str(a)+' ')
        f.write('\n')
# 
#         # log age values
#         for v in value:
#             f.write(str(v)+' ')
#         f.write('\n')
# 
# 
# # log miss rate curve
# for s in cache_size:
#     f.write(str(s)+' ')
# f.write('\n')
# for r in miss_rate.tolist():
#     f.write(str(r)+' ')
# f.write('\n')
# 
# # log test spec
# f.write( "x1: " + str(x1))
# f.write( "x2: " + str(x2))
# f.write( "x3: " + str(x3))
# f.write( "p1: " + str(p1))
# f.write( "p2: " + str(p2))
# f.write( "p3: " + str(p3))

<<<<<<< HEAD
    cache = Cache(s,value)
    a_counter1 = 0
    a_counter2 = 0
    a_counter3 = 0
    for i in range(iterate_times):
        # simulate data access
        if i % 10 < 10*p1:
            k = 0
        elif i % 10 >= 10*p1 and i % 10 < 10*(p1+p2):
            k = 1
        elif i % 10 >= 10*(p1+p2): 
            k = 2
            
        if k == 0:
            addr = array1[a_counter1 % len(array1)]
            a_counter1 += 1
        elif k == 1:
            addr = array2[a_counter2 % len(array2)]
            a_counter2 += 1
        elif k == 2:
            addr = array3[a_counter3 % len(array3)]
            a_counter3 += 1
            
        cache.lookup(addr)
    miss_rate[j] = 1-cache.get_hit_rate()
    print 'cache size:' + str(s) + ' miss rate:' + str(miss_rate[j])

    # log hit age and eviction age distribution
    f.write(str(s)+'\n')
    for a in cache.get_hit_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')
    for a in cache.get_evict_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')

    # log age values
    for v in value:
        f.write(str(v)+' ')
    f.write('\n')


# log miss rate curve
for s in cache_size:
    f.write(str(s)+' ')
f.write('\n')
for r in miss_rate.tolist():
    f.write(str(r)+' ')
f.write('\n')

# log test spec
f.write( "x1: " + str(x1))
f.write( "x2: " + str(x2))
f.write( "x3: " + str(x3))
f.write( "p1: " + str(p1))
f.write( "p2: " + str(p2))
f.write( "p3: " + str(p3))

f.close()
