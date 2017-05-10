### imports
import tensorflow as tf
import os

# file name
file_name = "./save_file"

### constant data
x  = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y_ = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]

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
y_out = # TODO


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

  if not os.path.exists(file_name) :
    print "No training file found.  Training..."
    print "\nloss"
    for step in range(500) :
      # TODO session execution command here
      if (step + 1) % 100 = # TODO
        print # TODO session execution command here

    print "Training complete.  Saving..."
    # TODO save to file 
    print "Model saved to file",file_name
    print "Run program again to use model."

  else :
    print "Training file",file_name,"found."
    # TODO restore from file
    results = # TODO calculate and return m1,b1,m2,b2,y_,loss
    labels  = "m1,b1,m2,b2,y_out,loss".split(",")
    for label,result in zip(*(labels,results)) :
      print ""
      print label
      print result

print ""
