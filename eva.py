#!/usr/bin/python

import numpy as np
import math

# Order ages by EVA incrementally. Start from a large cache where
# everything fits and then:
# - Compute EVA for each age.
# - Among ages that haven't been ordered yet, set the age with
#   lowest EVA to the lowest available eviction order.
# - Repeat until all ages are ordered

class Cache:
    def __init__(self, rdd):
        assert abs(np.sum(rdd) - 1) < 1e-5
        self.hits = rdd
        self.evictions = np.zeros_like(rdd)

    def __init__(self, cache):
        self.hits = np.copy(cache.hits)
        self.evictions = np.copy(cache.evictions)

    # Distribution of lifetime length, ie age of hits or evictions
    def lifetimes(self):
        return self.hits + self.evictions

    # Implied cache size equals expected value of lifetime length
    def size(self):
        l = self.lifetimes()
        return np.sum(np.arange(l.shape) * l)

def computeEva(cache):

    life = cache.lifetimes()
    eventsabove = np.sum(life[::-1])[::-1]
    explifeunconditioned = np.sum(eventsabove[::-1])[::-1]
    explife = explifeunconditioned / eventsabove

    hitsabove = np.sum(cache.hits[::-1])[::-1]
    hitp = hitsabove / eventsabove

    size = explife[0]
    gain = np.sum(cache.hits) / size
    assert 0 <= gain and gain <= size

    eva = hitp - gain * explife

    return eva

def findNextVictim(ordering, eva):
    nextVictim = -1

    for a in range(ordering.shape):
        # -1 means the age hasn't been ordered yet
        if ordering[a] != -1:
            continue

        if nextVictim == -1:
            nextVictim = a
        elif eva[a] < eva[nextVictim]:
            nextVictim = a

    assert nextVictim != -1
    return nextVictim

# Verify that all ages that have been ordered (ie, selected for
# eviction) have lower EVA than the one that is going to be evicted
# next. This isn't trivial because the computed EVA values change as
# more ages are selected for eviction, so EVA inversions are maybe
# possible?
def checkStackPropertyHolds(ordering, eva, nextVictim):
    for a in range(ordering.shape):
        if ordering[a] != -1:
            if eva[a] > eva[nextVictim]:
                return False

    # This code WILL FAIL. The convergence problem says that this
    # check will fail BY DEFINITION. The convergence problem is
    # essentially that evicting an age RAISES its eva (because it now
    # spends less time in the cache) such that it is larger than other
    # candidates in the cache and hence "should not be evicted".

    return True

# Turn all the time spent at a given age in the cache into evictions
# and reduce the number of hits at later ages proportionally...
def evictAgeFromCache(inCache, nextVictim):
    cache = Cache(inCache)

    life = cache.lifetimes()
    lifeAboveVictim = np.sum(life[nextVictim:])
    ageAtVictim = lifeAboveVictim / cache.size()

    # This isn't right...if we assume full associativity, then MANY
    # evictions can happen at the lowest ranked age before we start
    # evicting from elsewhere. Think about how many evictions can
    # happen before we run out of victims at that age...

    return cache

def rankAgesByEvaFromRdd(rdd):
    cache = Cache(rdd)
    ordering = np.zeros_like(rdd) - 1

    for a in range(rdd.shape):
        eva = computeEva(cache)
        nextVictim = findNextVictim(ordering, eva)
        assert checkStackPropertyHolds(ordering, eva, nextVictim)
        cache = evictAgeFromCache(cache, nextVictim)

        # Set the victim age to be evicted going forward. In other
        # words, prevent the behavior that "This is going to be
        # evicted, so I don't need to evict it." that causes
        # convergence problems.
        ordering[nextVictim] = a

    return ordering
