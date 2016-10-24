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
is_iter_policy = False
d1 = x1 * 3
d2 = x2 * 3
d3 = x3 * 3
ed = x1 + x2 + x3
print 'd1= ' + str(d1)
print 'd2= ' + str(d2)
print 'd3= ' + str(d3)
print 'ed= ' + str(ed)

iterate_times = 1000

# policy: give as values of each age
policies = []

# policy 0: only evict at 0
value1 = np.ones(MAX_AGE,dtype=float)
value1[0] = 0
policies.append(value1)

# policy 1: evict at d1, if no, then evict at 0
value2 = np.ones(MAX_AGE,dtype=float)
value2[0] = 0
value2[d1] = -1
policies.append(value2)

# policy 2: evict at d2, if no, then evict at 0
value3 = np.ones(MAX_AGE,dtype=float)
value3[0] = 0
value3[d2] = -1
policies.append(value3)

# policy 3: evict at d2, if no, then d1, then 0 
value4 = np.ones(MAX_AGE,dtype=float)
value4[0] = 0
value4[d1] = -1
value4[d2] = -2
policies.append(value4)

array1= range(0,x1)
array2 = range(x1,x1+x2)
array3 = range(x1+x2,x1+x2+x3)
cache_size = range(args.s0,args.s1,args.step)
miss_rate = np.zeros(len(cache_size),)

FILENAME = 'logs/' + str(args.s0) + '.log'
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
        if pid!=0 and not is_iter_policy:   continue
        cache = Cache(s,p)
        a_counter1 = 0
        a_counter2 = 0
        a_counter3 = 0
        for i in range(iterate_times):
            # simulate data access
            if i % 3 == 0:
                k = weighted_choice([200,1,1])
            elif i % 3 == 1:
                k = weighted_choice([1,200,1])
            elif i % 3 == 2:
                k = weighted_choice([1,1,200])

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
#        hit_distribution = [float(a)/event_sum for a in cache.get_hit_ages()]
        hit_distribution = [float(a)/sum(cache.get_hit_ages()) for a in cache.get_hit_ages()]

        evict_distribution = [float(a)/event_sum for a in cache.get_evict_ages()]
        f.write(str(s)+'\n')
        for a,h in enumerate(hit_distribution):
            # if h > 0: print 'hit rate at age ' + str(a) + '= ' + str(h)
            f.write(str(h)+' ')
        f.write('\n')
        for a,e in enumerate(evict_distribution):
            # if e > 0: print 'evict rate at age ' + str(a) + '= ' + str(e)
            f.write(str(e)+' ')
        f.write('\n')
#         plt.subplot(4,2,pid*2+1)
#         plt.plot(hit_distribution)
#         plt.title('policy ' + str(pid))
#         plt.subplot(4,2,pid*2+2)
#         plt.plot(evict_distribution)
#         plt.title('policy ' + str(pid))

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

plt.show()
f.close()
