import matplotlib.pyplot as plt
import numpy as np

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
    n = 512
    rdd = np.zeros(n)
    for i in range(len(d)):
        rdd[d[i]] = p[i]
    ed = np.sum(np.arange(n) * rdd) # expected reuse distance = working set size

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
