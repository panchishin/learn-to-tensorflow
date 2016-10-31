## Lesson 2 - solve XOR
# use random variables in the range of 0.1 to 0.9 and 32 bit floating precision

### imports
import tensorflow as tf

### constant data
x = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y = [[0.],[0.],[1.],[1.]]

### induction
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = # TODO
x0 = # TODO
y0 = # TODO

# Layer 1 = # TODO
m1 = # TODO
b1 = # TODO
h1 = # TODO

# Layer 2 = # TODO
m2 = # TODO
b2 = # TODO
y_ = # TODO


### loss

# loss : sum of the squares of y0 - y_
loss = # TODO

# training step : gradient decent (1.0) to minimize loss
train = # TODO


### training
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
  # TODO session execution command here
  print "\nloss"
  for epoc in range(5):
    for step in range(100) :
      # TODO session execution command here
    print # TODO get the loss from the session

  results = # TODO calculate and return m1,b1,m2,b2,y_,loss
  labels  = "m1,b1,m2,b2,y_,loss".split(",")
  for label,result in zip(*(labels,results)) :
    print ""
    print label
    print result

print ""
