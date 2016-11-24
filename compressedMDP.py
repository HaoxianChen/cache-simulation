#!/usr/bin/python

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import analytical_model as model

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
    critical_d = p1*d1 + (1-p1)*d2
    if s>0 and s<=critical_d:
        miss_rate = 1. - (1.-p3)*s/(p1*d1+p2*d2+p3*d2)
    elif s>critical_d and s<ed:
        miss_rate = (ed-s)/(d3-d2)
    else:
        miss_rate = 0

    return miss_rate

def miss_rate_d2_d1(p,d,s):
    [p1,p2,p3] = p
    [d1,d2,d3] = d
    ed = p1*d1 + p2*d2 + p3*d3
    critical_d = p1*d1 + (1-p1)*d2
    if s>0 and s<=d1:
        miss_rate = 1 - p1*s/d1
    elif s>d1 and s<=critical_d:
        miss_rate = 1-p1-p2*(s-d1) / ((1-p1)*(d2-d1))
    elif s>critical_d and s<ed:
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
        e[1] = 1-s/ed
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
        e[1] = 1. - hit_rate/p1
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
    critical_d = p1*d1 + (1-p1)*d2

    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<= critical_d:
        hit_rate = (1. - p3)*s/(p1*d1 + (1-p1)*d2)
        x = hit_rate / (1-p3)
        h[d1] = p1 * x
        h[d2] = p2 * x
        e[1] = 1. - x
        e[d2] = p3 * x
    elif s>critical_d and s<ed:
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
    critical_d = p1*d1 + (1-p1)*d2

    h = np.zeros(n)
    e = np.zeros(n)

    if s>0 and s<= d1:
        hit_rate = p1*s/d1
        x = hit_rate / p1
        h[d1] = p1 * x
        e[1] = 1. - x
        e[d1] = (1-p1) * x
    elif s>d1 and s<=critical_d:
        hit_rate = p1 + p2*(s-d1)/((1-p1)*(d2-d1))
        x = (hit_rate-p1)*(1-p1)/p2
        h[d1] = p1
        h[d2] = p2*x/(1-p1)
        e[d1] = 1-x-p1
        e[d2] = p3*x/(1-p1)
    elif s>critical_d and s<ed:
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

def analysis(p,d):
    _, (a1, a2) = plt.subplots(2,1)
    a1.set_title('RDD')
    a1.plot(rdd)

    s = np.arange(1,n)

    a2.set_title('Miss rate')
    # when 0<S<d1
    a2.plot([miss_rate_mru(p,d,size) for size in s],label='MRU')
    a2.plot([miss_rate_d1(p,d,size) for size in s],label='d1')
    a2.plot([miss_rate_d2(p,d,size) for size in s],label='d2', marker='.')
    a2.plot([miss_rate_d2_d1(p,d,size) for size in s],label='d2, d1', marker='*')
    a2.set_xlim(0,ed)
    a2.set_ylim(0,1)
    a2.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
def value_iteration(values,p,d,h,e,s,drag,allow_plot=False):
    
    allow_log = True
    threshold = 1e-4

    [d1,d2,d3] = d[0:3]
    v = np.zeros((2,n))
    v[1,:len(values)] = values

    delta = np.zeros(n)

    boundry = np.argmax(np.cumsum(h))

    cumevents = np.cumsum((h+e)[::-1])[::-1]
    cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

    if allow_plot:
        plt.close('all')
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        # slope = (1 + h[d1]) / d2
        # slope = 1 / ed
        # simple = slope * np.arange(len(rdd))
        # simple[d1+1:] += 1 - slope * d2
        # simple[d2+1:] = 0
        #simple = model.analysis_modal(p,d,h,e,False)
        #ax.plot(simple)
        line, = ax.plot(v[0])
        plt.ion()

    for i in range(20000):
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
                    # this[j] += leftover * (0 + that[j+1])

            if j>=0 and j <= boundry:
                delta[j] = abs(this[j]-this[0]-(that[j]-that[0])) 

        if max(delta) < threshold: 
            if allow_plot:
                plt.ioff()
            if allow_log:
                log_values(this-this[0])
            break

        if i%50 != 0 or not allow_plot: continue
        ax.set_title('After iteration %d' % i)
        yplot = this - this[0]
        line.set_ydata(yplot)
        ax.set_ylim(np.min(yplot[0:boundry+1]), np.max(yplot[0:boundry+1]))
        ax.set_xlim(0,boundry+1)
        ax.relim()
        ax.autoscale_view()
        plt.draw() # tell pyplot data has changed
        plt.pause(0.000001) # it won't actually redraw until you call pause!!!

    return this - this[0]

def opt_policy(p,d,s):
    miss_rate = np.zeros(4)
    miss_rate[0] = miss_rate_mru(p,d,s)
    miss_rate[1] = miss_rate_d1(p,d,s)
    miss_rate[2] = miss_rate_d2(p,d,s)
    miss_rate[3] = miss_rate_d2_d1(p,d,s)
    if np.argmin(miss_rate) == 0:
        return [0,],miss_rate[0]
    elif np.argmin(miss_rate) == 1:
        return [1,],miss_rate[1]
    elif np.argmin(miss_rate) == 2:
        return [2,0,1],miss_rate[2]
    elif np.argmin(miss_rate) == 3:
        return [2,1,0],miss_rate[3]

def policy_iteration(p,d,s,drag=1):
    ed = round(np.dot(p,d))
    step = - int(ed/8.)
    assert abs(step) >= 1
    # cache_size = range(int(ed),s,step) + [s,]
    cache_size = [s] * 10
    # values = np.zeros(max(d)+10)
    values = np.arange(max(d)+10)
    for s in cache_size:

        [opt,opt_miss_rate] = opt_policy(p,d,s)

        [h,e] = parse_policy(values,p,d,s)[1:3]
        
        if h[d[-1]] < 1e-4:
            print 'fill in rdd'
            h = fill_rdd(p,d,h)

        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(h)
        plt.xlabel('age')
        plt.ylabel('hit probability')
        plt.title('hit age distribution')
        plt.subplot(1,2,2)
        plt.plot(e)
        plt.xlabel('age')
        plt.ylabel('evict probability')
        plt.title('evict age distribution')
        plt.show()

        values = value_iteration(values,p,d,h,e,s,drag,True)
        miss_rate = parse_policy(values,p,d,s)[0]
        print 'size= ' + str(s) 
        print 'optimal policy: ' + str(opt) + ' miss rate: ' + str(opt_miss_rate)
        print "v[1] = %.3f" %values[1]
        for i in range(len(d)):
            print "v[%d] = %.3f" %(d[i]+1,values[d[i]+1])
    return values

def fill_rdd(p,d,h):
    w = 0.05
    new_h = (1-w) * h + w * rdd

    return new_h

def parse_policy(values,p,d,s):
    critical_point = [0,] + d[0:-1]
    policy = np.argsort([values[critical_point[i]+1] for i in range(len(critical_point))])
    
    if policy[0] == 0:
        miss_rate = miss_rate_mru(p,d,s)
        h,e = hit_rate_mru(p,d,s)
    elif policy[0] == 1:
        miss_rate = miss_rate_d1(p,d,s)
        h,e = hit_rate_d1(p,d,s)
    elif policy[0] == 2:
        if policy[1] == 0:
            miss_rate = miss_rate_d2(p,d,s)
            h,e = hit_rate_d2(p,d,s)
        elif policy[1] == 1:
            miss_rate = miss_rate_d2_d1(p,d,s)
            h,e = hit_rate_d2_d1(p,d,s)

#     testing convergence
#     for a in range(1,len(e)):
#         e[a] += e[a-1] * 0.9
# 
#     if np.sum(h) < 1:
#         e *= (1. - np.sum(h)) / np.sum(e)

    return miss_rate, h, e

def log_values(values):
    f = open('policy.log','w+')
    f.write(str(len(values))+'\n')
    for v in values:
        f.write("%.2f " %v )
    f.close()
    
n = 512
if __name__ == '__main__':
    p = [0.25, 0.25, 0.5]
    d = [40,80,320]

    rdd = np.zeros(n)
    for i in range(len(d)):
        rdd[d[i]] = p[i]
    ed = np.sum(np.arange(n) * rdd) # expected reuse distance = working set size
    s = 80
    #assert ed > d2

    analysis(p,d)
    # values = fill_rdd(p,d,s)
    drag = 0.9999
    values = policy_iteration(p,d,s,drag)

    #print opt_policy(p,d,s)
    # values = np.arange(max(d)+10)
    # values[d[0]+1:] -= values[d[1]]
    # values[d[1]+1:] -= values[d[2]]
    # plt.plot(values)
    # plt.show()
    # [h,e] = parse_policy(values,p,d,s)[1:3]

    # if h[max(d)] < 10e-4:
    #     print 'fill rdd'
    #     h = fill_rdd(p,d,h)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(h)
    # plt.xlim(0,ed)
    # plt.subplot(1,2,2)
    # plt.plot(e)
    # plt.xlim(0,ed)
    # plt.show()
    # values = value_iteration(p,d,h,e,s,drag,True)
    # plt.close("all")
    print "v[1] = %.3f" %values[1]
    for i in range(len(d)):
        print "v[%d] = %.3f" %(d[i]+1,values[d[i]+1])
    plt.close('all')
    plt.figure()
    plt.plot(values)
    plt.xlim(0,max(d))
    plt.ylim(min(values[0:d[2]]),max(values[0:d[2]]))
    plt.show()
    
