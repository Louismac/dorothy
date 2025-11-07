from itertools import groupby
import random
import numpy as np
a = np.random.randint(0, 100, 10)
def get_key(v):
    key = "yes" if v< 50 else "no"
    return key

sorted_cmds = sorted(a, key=get_key)
print(sorted_cmds)
for k,g in groupby(sorted_cmds, get_key):
    listOfThings = k + " " + " and ".join([str(thing) for thing in g])
    print(listOfThings)