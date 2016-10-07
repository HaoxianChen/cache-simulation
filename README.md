Toy simulation codes for cache behavior over different access patterns and
replacement policies.

#### To run simulation:
```
python mdp.py [minimum cache size] [maximum cache size] [-s step]
```

#### To plot simulation results:
Plot miss rate over different cache sizes
```
python plot.py [logs/log-file-name] m
```
Plot hit age and eviction age distribution
```
python plot.py [logs/log-file-name] h
```
Plot ranking functions 
```
python plot.py [logs/log-file-name] r
```
