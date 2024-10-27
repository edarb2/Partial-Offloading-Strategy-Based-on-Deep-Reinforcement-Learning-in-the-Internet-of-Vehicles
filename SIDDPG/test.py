import numpy as np
import math
import tensorflow as tf
BATCH_SIZE = 10
a0 = (1,2,3,4,5)
a1 = (11,22,33,44,55)
a2 = (111,222,333,444,555)
a3 = (1111,2222,3333,4444,5555)
a4 = (11111,22222,33333,44444,55555)
a = np.hstack((a0,a1,a2,a3,a4))
b = np.random.choice(5, size=BATCH_SIZE)
print(a)
print(b)
print(np.ones(10) * 209715)

