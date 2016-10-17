#!/usr/bin/python

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math

np.set_printoptions(precision=2)

n = 256
x1 = 32
x2 = 48
p = 0.5
d1 = x1 / p 
d2 = x2 / (1-p)

rdd = np.zeros(n)
rdd[d1] = p
rdd[d2] = 1-p
ed = np.sum(np.arange(n) * rdd) # expected reuse distance = working set size

print 'ed = ' + str(ed)

# x1 - size of small array
# x2 - size of big array
# p - fraction of accesses to small array
# d1 = x1 / p
# d2 = x2 / (1-p)
#
# working set size = x1 + x2 = d1 * p + d2 * p = exp(rdd)

def analysis():
    _, (a1, a2) = plt.subplots(2,1)
    a1.set_title('RDD')
    a1.plot(rdd)

    s = np.arange(n)

    a2.set_title('Miss rate')
    a2.plot(1. - s / ed, label='Evict @ 0 only')
    a2.plot(1. - p * s / d1, label='Evict @ 0, then d1')
    a2.plot((ed - s) / (d2 - d1), label='Evict @ d1')
    a2.plot(d1, (ed - d1) / (d2 - d1), ls='none', marker='o', label='Can evict @ d1')
    a2.set_xlim(0,ed)
    a2.set_ylim(0,1)
    a2.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close('all')
analysis()

# we're going to model what happens on a cache of size 20, where the
# best policy is to evict at d1
s = 70
assert s > 0 and s < ed

v = np.zeros((2,n))

h = np.zeros_like(v[0])
e = np.zeros_like(v[0])

# evict at d1
assert s > d1
hitrate = 1. - (ed - s) / (d2 - d1)
h[d1] = rdd[d1]
h[d2] = rdd[d2] - (1 - hitrate)
e[d1] = 1 - hitrate

# # evict at zero
# hitrate = 1. - s / ed
# h[d1] = (1 - hitrate) * rdd[d1]
# h[d2] = (1 - hitrate) * rdd[d2]
# e[0] = 1 - hitrate

# Smoothing of evictions...
# for i in range(d1+1,n):
#     e[i] = e[i-1] * .8
# e *= (1 - np.sum(h)) / np.sum(e)

# totalevents = np.sum(h + e)
# h /= totalevents
# e /= totalevents

cumevents = np.cumsum((h+e)[::-1])[::-1]
cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

# plt.figure()
# plt.plot(np.cumsum(h))
# plt.plot(np.cumsum(e))
# plt.plot(np.cumsum(h+e))
# plt.show()

ages = np.arange(n)
hitprobability = np.sum(h)
cachesize = np.sum(ages * (h+e))
linegain = hitprobability / cachesize

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
line, = ax.plot(v[0])
plt.ion()

for i in range(100000):

    drag = 1

    this = v[i%2]
    that = v[(i+1)%2]

    # this[0] = 0

    for j in range(0, n):
        this[j] = 0

        # hit probability
        this[j] += h[j] / cumevents[j] * (1 + drag * that[0])
        # if j == 47: print 'H', h[j] / cumevents[j], this[j]

        # eviction probability
        this[j] += e[j] / cumevents[j] * (0 + drag * that[0])
        # if j == 47: print 'E', e[j] / cumevents[j], this[j]

        # otherwise
        if j+1 < n:
            leftover = (cumevents[j] - h[j] - e[j]) / cumevents[j]
            if leftover > 1e-5:
                this[j] += leftover * (0 + drag * that[j+1])
                # if j == 47: print 'L', leftover, this[j]

        if this[j] < 0:
            print i, j
            print h[j]
            print j, '%.2f = %.2f / %.2f * (1 + %.2f) + %.2f / %.2f * %.2f + %.2f * %.2f - %.2f / %.2f' % (this[j],
                                                                                                           h[j], cumevents[j], that[0],
                                                                                                           e[j], cumevents[j], that[0],
                                                                                                           leftover, that[j+1],
                                                                                                           hitprobability, cachesize)
        assert this[j] >= 0.

        # if cumevents[j] != 1.0: this[j] += 0.04
        # 0.01305 # 0.0435
        # 2.1425 / cachesize # 2.857 * hitprobability / cachesize # linegain

    if i % 50 != 0: continue

    # for j in range(n):
    #     if np.max(np.abs(this[j])) > 1e-3:
    #         print j, '%.2f = %.2f / %.2f * (1 + %.2f) + %.2f / %.2f * %.2f + %.2f * %.2f - %.2f / %.2f' % (this[j],
    #                                                                                                        h[j], cumevents[j], that[0],
    #                                                                                                        e[j], cumevents[j], that[0],
    #                                                                                                        leftover, that[j+1],
    #                                                                                                        hitprobability, cachesize)

    print 'After iteration %d' % i
    # raw_input("Press enter to continue...")
    ax.set_title('After iteration %d' % i)
    yplot = this - this[0]
    line.set_ydata(yplot)
    ax.set_ylim(np.min(yplot[0:d2]), np.max(yplot[0:d2]))
    ax.relim()
    ax.autoscale_view()
    plt.draw() # tell pyplot data has changed
    plt.pause(0.0001) # it won't actually redraw until you call pause!!!

plt.close('all')
