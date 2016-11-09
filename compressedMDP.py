#!/usr/bin/python

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math

np.set_printoptions(precision=2)

# x1 - size of small array
# x2 - size of big array
# p - fraction of accesses to small array
# d1 = x1 / p
# d2 = x2 / (1-p)
#
# working set size = x1 + x2 = d1 * p + d2 * p = exp(rdd)

def vi_analysis(s):
    # s: cache size
    delta_d1 = float(d1*s/ed+1)/float((s+p2)*p1) - p1/p2
    return delta_d1

def miss_rate_mru(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    if s<ed and s>0:
        miss_rate = 1 - s/ed
    else:
        miss_rate = 0
    return miss_rate

def miss_rate_d1(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    if s>0 and s<=d1:
        miss_rate = 1 - p1*s/d1
    elif s>d1 and s<ed:
        miss_rate = (1-p1)*(ed-s)/(ed-d1)
    else:
        miss_rate = 0

    return miss_rate

def miss_rate_d2(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    if s>0 and s<=d2:
        miss_rate = 1 - (1-p3)*s/(p1*d1+p2*d2+p3*d2)
    elif s>d2 and s<ed:
        miss_rate = (ed-s)/(d3-d2)
    else:
        miss_rate = 0

    return miss_rate

def miss_rate_d2_d1(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    if s>0 and s<=d1:
        miss_rate = 1 - p1*s/d1
    elif s>d1 and s<=d2:
        miss_rate = 1-p1-p2*(s-d1) / ((1-p1)*(d2-d1))
    elif s>d2 and s<ed:
        miss_rate = (ed-s)/(d3-d2)
    else:
        miss_rate = 0

    return miss_rate

def hit_rate_mru(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    
    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<ed:
        h[d1] = p1*s/ed
        h[d2] = p2*s/ed
        h[d3] = p3*s/ed
        e[0] = 1-s/ed
    elif s>=ed:
        hit_rate = 1.
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3

    return h,e

def hit_rate_d1(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    
    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<= d1:
        hit_rate = float(p1*s/ed)
        h[d1] = hit_rate
        e[0] = 1. - hit_rate/p1
        e[d1] = (1. -p1) * hit_rate/p1
    elif s>d1 and s<ed:
        hit_rate = 1. - (1. - p1)*(ed-s)/(ed-d1)
        h[d1] = p1
        h[d2] = p2*hit_rate/(1. -p1) 
        h[d3] = p3*hit_rate/(1. -p1) 
        e[d1] = 1. - hit_rate
    elif s>=ed:
        hit_rate = 1.
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3

    return h,e

def hit_rate_d2(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3

    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<= d2:
        hit_rate = (1. - p3)*s/(p1*d1 + (1-p1)*d2)
        x = hit_rate / (1-p3)
        h[d1] = p1 * x
        h[d2] = p2 * x
        e[0] = 1. - x
        e[d2] = p3 * x
    elif s>d2 and s<ed:
        hit_rate = 1. - (ed-s)/(d3-d2)
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3 - (1. - hit_rate)
        e[d2] = 1. - hit_rate
    elif s>=ed:
        hit_rate = 1.
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3

    return h,e

def hit_rate_d2_d1(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3

    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<= d1:
        hit_rate = p1*s/d1
        x = hit_rate / p1
        h[d1] = p1 * x
        e[0] = 1. - x
        e[d1] = (1-p1) * x
    elif s>d1 and s<=d2:
        hit_rate = p1 + p2*(s-d1)/((1-p1)*(d2-d1))
        x = (hit_rate-p1)*(1-p1)/p2
        h[d1] = p1
        h[d2] = p2*x/(1-p1)
        e[d1] = 1-x-p1
        e[d2] = p3*x/(1-p1)
    elif s>d2 and s<ed:
        hit_rate = 1. - (ed-s)/(d3-d2)
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3 - (1. - hit_rate)
        e[d2] = 1. - hit_rate
    elif s>=ed:
        hit_rate = 1.
        h[d1] = p1
        h[d2] = p2
        h[d3] = p3

    return h,e

def analysis():
    _, (a1, a2) = plt.subplots(2,1)
    a1.set_title('RDD')
    a1.plot(rdd)

    s = np.arange(1,n)

    a2.set_title('Miss rate')
    # when 0<S<d1
    a2.plot([miss_rate_mru([p1,p2,p3],[d1,d2,d3],size) for size in s],label='MRU')
    a2.plot([miss_rate_d1([p1,p2,p3],[d1,d2,d3],size) for size in s],label='d1')
    a2.plot([miss_rate_d2([p1,p2,p3],[d1,d2,d3],size) for size in s],label='d2', marker='.')
    a2.plot([miss_rate_d2_d1([p1,p2,p3],[d1,d2,d3],size) for size in s],label='d2, d1', marker='*')
    a2.set_xlim(0,ed)
    a2.set_ylim(0,1)
    a2.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close('all')

def value_iteration(h,e,s,threshold,drag,allow_plot):
    v = np.zeros((2,n))
    delta = np.zeros(n)

    boundry = np.argmax(np.cumsum(h))

    cumevents = np.cumsum((h+e)[::-1])[::-1]
    cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    #slope = (1 + h[d1]) / d2
    #simple = slope * np.arange(len(rdd))
    #simple[d1:] += 1 - slope * d2
    #simple[d2:] = 0
    # ax.plot(simple)
    line, = ax.plot(v[0])
    plt.ion()

    for i in range(10000):
        this = v[i%2]
        that = v[(i+1)%2]

        for j in range(0, n):
            this[j] = 0

            # hit probability
            this[j] += h[j] / cumevents[j] * (1 + drag * that[0])

            # eviction probability
            this[j] += e[j] / cumevents[j] * (0 + drag * that[0])

            # otherwise
            if j+1 < n:
                leftover = (cumevents[j] - h[j] - e[j]) / cumevents[j]
                if leftover > 1e-5:
                    this[j] += leftover * (0 + drag * that[j+1])

            if j>=0 and j <= boundry:
                delta[j] = abs(this[j]-this[0]-(that[j]-that[0])) 

        if max(delta) < threshold: break

        if i%1 != 0 or not allow_plot: continue
        ax.set_title('After iteration %d' % i)
        yplot = this - this[0]
        line.set_ydata(yplot)
        ax.set_ylim(np.min(yplot[0:boundry+1]), np.max(yplot[0:boundry+1]))
        ax.relim()
        ax.autoscale_view()
        plt.draw() # tell pyplot data has changed
        plt.pause(0.000001) # it won't actually redraw until you call pause!!!

    print 'boundry=' + str(boundry)
    if boundry > d2:
        return np.argsort([this[0],this[d1+1],this[d2+1]])
    elif boundry > d1:
        return np.argsort([this[0],this[d1+1]])
    else:
        return 0
    
if __name__ == '__main__':
    n = 512
    d1 = 24
    d2 = 96
    d3 = 240
    p1 = 0.2
    p2 = 0.7
    p3 = 1 - p1 - p2

    rdd = np.zeros(n)
    rdd[d1] = p1
    rdd[d2] = p2
    rdd[d3] = p3
    ed = np.sum(np.arange(n) * rdd) # expected reuse distance = working set size

    analysis()
    s = ed-10
    h,e = hit_rate_d2([p1,p2,p3],[d1,d2,d3],s)

    print "size = " + str(s)
    plt.subplot(2,1,1)
    plt.plot(h)
    plt.title('hit')
    plt.subplot(2,1,2)
    plt.plot(e)
    plt.title('evict')
    plt.show()

    policy = value_iteration(h,e,s,1e-4,1,True)
    print policy
