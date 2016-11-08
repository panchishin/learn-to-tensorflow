import numpy as np

### Creation of series data.
# The X is a series like [2,3,4] and the expected result is [5]
# Also makes decreasing and constant series
# of length 4,3,and 2
def makeXY() :
  x,y = makeXYfirst()
  if np.random.random() > 0.66 :
    x = x[1:]
  elif np.random.random() > 0.5 :
    x = x[2:]
  return x,y 

def makeXYfirst() :
  if np.random.random() > 0.66 :
    x = int( np.random.random() * 6. ) + np.array([0,1,2,3])
    y = x[-1] + 1
    return x,y
  if np.random.random() > 0.5 :
    x = int( np.random.random() * 6. ) + np.array([4,3,2,1])
    y = x[-1] - 1
    return x,y
  else : 
    x = int( np.random.random() * 10. ) + np.array([0,0,0,0])
    y = x[-1] 
    return x,y

