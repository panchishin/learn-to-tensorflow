### imports
import tensorflow as tf
import file as helper
import time
import numpy as np
import os
import random

import model

labels = model.labels
neurons = model.neurons
layers = model.layers

text_length = 50
helper.set("text-input-file.txt",text_length)

steps_per_report = 50

for learning_rate_value in [ 0.1 , 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001 ] :
 print "\nStarting execution with learning rate",learning_rate_value
 with tf.Session() as sess:

  # initialize session variables
  sess.run( tf.initialize_all_variables() )
  sess.run( model.set_init_state )

  for report in range(3):
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
    print "Epoch =",epoch," , avgloss =",avgloss,", learning rate",learning_rate_value
  
print " " 
