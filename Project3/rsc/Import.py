import os
import numpy as np

dir = os.path.dirname('__file__')
f = os.path.join(dir, '..', 'data','ratings.csv')
with open(f, 'r') as fid:
    data = np.loadtxt(fid, delimiter=",")

print data