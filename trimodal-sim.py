import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse
import mru
from cache import *

if __name__ == '__main__':

    d1 = 24
    d2 = 96
    d3 = 240
    p1 = 0.25
    p2 = 0.25
    p3 = 0.5
    ed = d1*p1 + d2*p2 + d3*p3
    idealRdDist = [(p1,d1),(p2,d2),(p3,d3)]
    cache_size = np.arange(40,50,3)

    # trace = mru.TraceScan([p for p,d in idealRdDist], [int(p*d+0.5) for p,d in idealRdDist])
    trace = mru.TraceDistribution(idealRdDist)
    trace.generate(10000)

    plt.figure()
    def idealRdDistNonSparse():
        cump = 0.
        i = 0
        rdd = np.zeros_like(trace.rdDist)
        for d in range(len(rdd)):
            while i < len(idealRdDist) and idealRdDist[i][1] <= d:
                cump += idealRdDist[i][0]
                i += 1
            rdd[d] = cump * len(trace.trace)
        return rdd

    plt.plot(idealRdDistNonSparse(), label='Ideal RD dist')
    plt.plot(np.cumsum(trace.rdDist), label='Actual RD dist')
    plt.legend(loc='best')
    plt.show()

    policy_value = np.arange(MAX_AGE)
    policy_value[d2:MAX_AGE] -= MAX_AGE
    policy_value[d1:d2] -= policy_value[d2-1]

    plt.figure()
    plt.plot(policy_value)
    plt.xlabel('age')
    plt.ylabel('value')
    plt.show()
    
    miss_rate = np.zeros_like(cache_size,dtype=np.float)

    for j,s in enumerate(cache_size):
        print 'simulating cache at size: ' + str(s) + '...'
        my_cache = Cache(s,policy_value)
        for i in range(len(trace.trace)):
#            if i % 1000 == 0: print 'accesses: ' + str(i) + ' hit rate: ' + str(my_cache.get_hit_rate())
            my_cache.lookup(trace.trace[i])
        miss_rate[j] = 1.0 - float(my_cache.get_hit_rate())
        print 'miss rate: ' + str(miss_rate[j])

        events = float(sum(my_cache.get_hit_ages()) + sum(my_cache.get_evict_ages()))

        if len(cache_size) <6:
            plt.figure()
            plt.subplot(2,1,1,title='hit age distribution')
            plt.plot(np.cumsum(my_cache.get_hit_ages()/events))
            plt.subplot(2,1,2,title='evict age distribution')
            plt.plot(np.cumsum(my_cache.get_evict_ages()/events))
            plt.show()

    if len(cache_size) >1:
        plt.figure()
        # plt.plot(cache_size, 1. - (cache_size) / (ed), label='Evict @ 0 only')
        # plt.plot(cache_size, 1. - (p1 * (cache_size)) / (d1), label='Evict @ d1, then 1')
        # when d1 < S < d2
        # plt.plot(cache_size, (ed-cache_size) / ((p2*d2+p3*d3)/(1-p1)-d1), label='Evict @ d1, then 1')
        # plt.plot(cache_size, 1-(1-p3)*cache_size/(p1*d1+p2*d2+p3*d2), label='Evict @ d2, then 0')
        plt.plot(cache_size, 1-p1-p2*(cache_size-d1)/((1-p1)*d2-d1), label='Evict @ d2, d1, then 0')
        # when d2 < S
        # plt.plot(cache_size, (ed-cache_size)/(d3-d2), label='Evict at d2, then 0')
        # plt.plot(cache_size, (1-p1)*(ed-cache_size)/(ed-d1), label='Evict at d1, then 0')
        plt.plot(cache_size,miss_rate, label='simulation')
        plt.legend(loc='best')
        plt.ylim(0,1)
        plt.show()

