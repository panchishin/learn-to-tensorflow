import tensorflow as tf
import numpy as np
import layer

SIZE = 28

x0 = tf.placeholder(tf.float32, [None, SIZE*SIZE] , name="x0")
y0 = tf.placeholder(tf.float32, [None, 10] , name="y0")
x_reshape = tf.reshape( x0, [-1,SIZE,SIZE,1], name="x_in" )
x_in = x_reshape

#x_noisy = layer.high_low_noise( x_reshape , HIGH_LOW_NOISE)

stages = []
stages.append( layer.conv( x_in , 1 , 16 , width=5, stride=4, padding='SAME' ) )
embedding = tf.tanh( stages[-1] )
stages.append( embedding )
stages.append( layer.upscaleFlat( stages[-1] , scale=4 ) )
stages.append( layer.conv_relu( stages[-1] , 16 , 1 , width=5, padding='SAME' ) )
x_out = stages[-1]


loss = tf.reduce_mean( tf.square( x_in - x_out ) ) + 10. * tf.square( tf.reduce_mean(embedding) + .9 )
learning_rate = tf.placeholder( tf.float32 )
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()


sess = tf.Session()
sess.run( tf.global_variables_initializer() )

print """
==== Tensor Shapes ====
"""

results = sess.run(stages, feed_dict={x0:mnist.train.next_batch(4)[0]})
for result,stage in zip(results,stages) :
    print result.shape , stage.name


print """
=======================
"""

def report(index,result) :
    if index == 1 or ( index <= 50 and index % 10 == 0 ) or ( index <= 500 and index % 100 == 0 ) or index % 1000 == 0 :
        print "index : %5d, loss: %5.3f" % ( index, result)


def doAutoEncodeTraining( batches, batch_size=200, rate=1e-3 ) :
  print "doAutoEncodeTraining"
  for index in range(1,batches+1) :
    result,_ = sess.run( [loss,train], feed_dict={x0:mnist.train.next_batch(batch_size)[0],learning_rate:rate})
    report(index,result)


def showExample(count) :
  for _ in range(count) :
    sample_x,sample_y_in = mnist.train.next_batch(1)
    sample_in,sample_out = sess.run([x_in,x_out],feed_dict={x0:sample_x} )
    print "===== IN ====="
    print (sample_in.reshape([28,28]) * 5 ).round().astype(int)
    print "===== OUT ====="
    print (sample_out.reshape([28,28]) * 5 ).round().astype(int)

def reset() :
  sess.run( tf.global_variables_initializer() )

def embeddings() :
  sample_x, sample_y = mnist.train.next_batch(4)
  return sess.run([embedding,y0], feed_dict={x0:sample_x,y0:sample_y})


def help() :
  print """
showExample(2)
doAutoEncodeTraining(100)
"""

help()






















import falcon
import json
import random
import os.path
from scipy import spatial

def getImageWithIndex(index) :
    return mnist.test.images[index:index+1]

def getExampleIn(index) :
    return sess.run(x_in,feed_dict={x0:getImageWithIndex(index)} ).reshape([SIZE,SIZE])

def getExampleOut(index) :
    return sess.run(x_out,feed_dict={x0:getImageWithIndex(index)} ).reshape([SIZE,SIZE])

def arrayToImage(data) :
    import scipy.misc
    import tempfile
    with tempfile.TemporaryFile() as fp :
        scipy.misc.toimage( data ).save( fp=fp, format="PNG" )
        fp.seek(0)
        return fp.read()

def falconRespondArrayAsImage(data,resp) :
    resp.content_type = 'image/png'
    resp.body = arrayToImage(data)


print """
================================
Define the rest endpoints
================================
"""

class Ping:
    def on_get(self, req, resp):
        resp.body = json.dumps( { 'response': 'ping' } )


class Display:
    def on_get(self, req, resp, file_name):
        if not os.path.isfile("view/"+file_name) :
            return

        result = open("view/"+file_name,"r")
        if ( "html" in file_name) :
            resp.content_type = "text/html"
        else :
            resp.content_type = "text/plain"
        
        resp.body = result.read()
        result.close()


class DisplayImage:
    def on_get(self, req, resp, index, junk) :
      try :
        falconRespondArrayAsImage( getExampleIn(int(index)) , resp )
      except :
        pass


class DreamImage:
    def on_get(self, req, resp, index, junk) :
      try :
        falconRespondArrayAsImage( getExampleOut(int(index)) , resp )
      except :
        pass


class DoLearning:
    def on_get(self, req, resp) :
      doAutoEncodeTraining(1000)
      resp.body = json.dumps( { 'response': 'done'} )


print """
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/ping', Ping())
api.add_route('/view/{file_name}', Display())
api.add_route('/display/{index}/{junk}', DisplayImage())
api.add_route('/dream/{index}/{junk}', DreamImage())
api.add_route('/learn', DoLearning())






