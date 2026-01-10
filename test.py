import numpy as np

vecA = np.array([(1,2,3), (4,5,6)])
vecB = np.array([7,8,9,10,11,11.5])

print(np.searchsorted(vecB, 8, side='right'))