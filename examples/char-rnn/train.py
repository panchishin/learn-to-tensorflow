### imports
import tensorflow as tf
import filehelper as helper
import time
import numpy as np
import os
import random

import model

text_length = 100
steps_per_report = 500
learning_rate_value = 0.01

print "text_length",text_length,", labels",model.labels, ", neurons",model.neurons, ", layers",model.layers, ", steps_per_report",steps_per_report


helper.set("text-input-file.txt",text_length)
file_name = "temp.save"



### Execution
print "Starting execution with learning rate",learning_rate_value
with tf.Session() as sess:

  saver = tf.train.Saver()
  if os.path.exists(file_name) :
    print "Restoring state from previous save..."
    saver.restore(sess,file_name)
  else :
    print "Initializing state..."
    # initialize session variables
    sess.run( tf.global_variables_initializer() )
    sess.run( model.set_init_state )
  print "...done"
  print " "

  for report in range(2500):
    start_time = time.time()
    # initialize avgloss and num correct to 0 and 0.0 respectively
    avgloss, epoch = 0, 0.0

    for step in range(steps_per_report) :
      # retrieve x and y example data using makeXY()
      inx,outy,epoch = helper.getXY( report*steps_per_report + step )

      # execute 'loss' and 'y_argmax' to gather stats
      junk_a,thisloss,result,junk_b = sess.run( 
        [model.train, model.loss, model.y_argmax, model.update_state] , 
        feed_dict={model.x0:inx, model.y0:outy, model.learning_rate:learning_rate_value} )

      # add thisloss to avgloss
      avgloss += thisloss

    avgloss = avgloss / steps_per_report
    learning_rate_value *= 0.99
    print "Epoch =",epoch,", avg loss =",avgloss,", learn rate =",learning_rate_value
    print "\telapse =",int(time.time() - start_time),"sec , speed =",int((steps_per_report*text_length)/(time.time() - start_time)),"chars/sec"
    start_time = time.time()
    saver.save(sess,file_name)
    print "\tsave done in",(time.time() - start_time),"seconds"
    
