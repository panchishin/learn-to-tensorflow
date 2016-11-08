### imports
import tensorflow as tf
import numpy as np

labels = 10
neurons = 20
layers = 2
steps_per_epoch = 500

### induction

# input layer
dropout = # TODO
x0 = # TODO
y0 = # TODO

# one-hot encoding
x0_hot = # TODO
y0_hot = # TODO

# RNN layer cell definition
cell = # TODO Cell
cell = # TODO Dropout
cell = # TODO Multi

# RNN output
output, state = # TODO dynamic rnn
last_output = tf.reshape(output[-1,-1],[-1,neurons])

# fully connected output layer
m = # TODO neurons x labels
b = # TODO labels
z = # TODO
y_out = # TODO softmax
y_argmax = # TODO 


### loss
# use basic cross entropy
loss = # TODO


### training
# use adam or gradient decent optimizer with 0.01 
train = # TODO


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


### Execution
with tf.Session() as sess:
  # initialize session variables
  # TODO session execution command here

  for epoch in range(20):
    # initialize avgloss and num correct to 0 and 0.0 respectively
    avgloss, correct = 0 , 0.0

    for step in range(steps_per_epoch) :
      # retrieve x and y example data using makeXY()
      inx,outy = makeXY()

      # execute 'train' with dropout of 0.5 to train a resilient NN
      # TODO session execution command here

      # execute 'loss' and 'y_argmax' with dropout of 1.0 to gather stats
      thisloss,result = # TODO

      # add thisloss to avgloss
      avgloss += thisloss
      # increment correct if result is the same as outy
      correct += 1.0 if result == outy else 0.0

    print "Epoch =",epoch," , correct =",(correct/steps_per_epoch)," , avgloss =",(avgloss / steps_per_epoch)
    if correct == steps_per_epoch :
      print "Finished"
      break

