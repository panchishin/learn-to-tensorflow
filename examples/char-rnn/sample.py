### imports
import tensorflow as tf
import read_file_helper as helper
import time
import numpy as np
import os
import random
import model


file_name = "temp.save"

### Execution
with tf.Session() as sess:

  if os.path.exists(file_name) :
    print "Restoring state from previous save..."
    saver = tf.train.Saver() 
    saver.restore(sess,file_name)
    print "...done"



  while True :
    data = raw_input("Start a sentence:")

    for temp in [0.2,0.3,0.5,0.7] :
      sentence = [ ord(c) for c in data ]

      sess.run( model.set_init_state )
      feeddata = sentence
      for _ in range(300) :
        lastProbs,junk = sess.run([model.y_out,model.update_state], feed_dict={model.x0:feeddata}) 
        lastProbs = np.array( lastProbs )[-1,:]

        index = model.randomLabel(lastProbs,temp)

        feeddata = [ index ]
        sentence.append( index )

      print "".join([ chr(c) for c in sentence ])
      print ""

    
