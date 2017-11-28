import shelve
import numpy as np
from matplotlib import pyplot as plt

data = shelve.open('data')
fig, a = plt.subplots(2, 3)
a = a.ravel()
for idx, ax in enumerate(a):
    diff = np.diff(data['{0}. trace'.format(idx)]['CallStart'])
    ax.hist(diff, bins=np.arange(0, 15, 1))
plt.show()
data.close()