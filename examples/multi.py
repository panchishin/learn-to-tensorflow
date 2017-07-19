import tensorflow as tf
import numpy as np
import layer

SIZE = 32
LEARNING_RATE = 1e-2
HIGH_LOW_NOISE = 0.02


x0 = tf.placeholder(tf.float32, [None, 28*28] , name="x0")
learning_rate = tf.placeholder( tf.float32 )

x_reshape = tf.reshape( x0, [-1,28,28,1], name="x_in" )
x_enlarge = tf.image.resize_images( x_reshape, [SIZE,SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False )
x_in = x_enlarge
x_noisy = layer.high_low_noise( x_in , HIGH_LOW_NOISE)


def encode(image, layers_in, layers_out, width=3) :
  result = image
  result = layer.conv_relu( result , layers_in , layers_out , stride=2, width=width )
  return result

def decode(image, layers_in, layers_out, width=3) :
  result = image
  result = layer.upscaleFlat( result , scale=2 )
  result = layer.conv_relu( result , layers_in , layers_out , width=width )
  return result


with tf.variable_scope("conv1") :
  conv5a = encode( x_noisy , 1 , 2 )
with tf.variable_scope("conv2") :
  conv5b = encode( conv5a , 2 , 4 )
with tf.variable_scope("conv3") :
  conv5c = encode( conv5b , 4 , 8 )
with tf.variable_scope("conv5") :
  conv5d = encode( conv5c , 8 , 16 )
with tf.variable_scope("conv5") :
  conv5e = encode( conv5d , 16 , 32 )
with tf.variable_scope("deconv3") :
  deconv5a = decode(conv5e , 32 , 16 )
with tf.variable_scope("deconv3") :
  deconv5b = decode(deconv5a , 16 , 8 )
with tf.variable_scope("deconv3") :
  deconv5c = decode(deconv5b , 8 , 4 )
with tf.variable_scope("deconv2") :
  deconv5d = decode(deconv5c , 4 , 2 )
with tf.variable_scope("deconv1") :
  deconv5e = decode(deconv5d , 2 , 1 )

x_out_5 = deconv5e

with tf.variable_scope("conv1", reuse=True) :
  conv4a = encode( x_noisy , 1 , 2 )
with tf.variable_scope("conv2", reuse=True) :
  conv4b = encode( conv4a , 2 , 4 )
with tf.variable_scope("conv3", reuse=True) :
  conv4c = encode( conv4b , 4 , 8 )
with tf.variable_scope("conv4", reuse=True) :
  conv4d = encode( conv4c , 8 , 16 )
with tf.variable_scope("deconv3", reuse=True) :
  deconv4a = decode(conv4d , 16 , 8 )
with tf.variable_scope("deconv3", reuse=True) :
  deconv4b = decode(deconv4a , 8 , 4 )
with tf.variable_scope("deconv2", reuse=True) :
  deconv4c = decode(deconv4b , 4 , 2 )
with tf.variable_scope("deconv1", reuse=True) :
  deconv4d = decode(deconv4c , 2 , 1 )

x_out_4 = deconv4d


with tf.variable_scope("conv1", reuse=True) :
  conv3a = encode( x_noisy , 1 , 2 )
with tf.variable_scope("conv2", reuse=True) :
  conv3b = encode( conv3a , 2 , 4 )
with tf.variable_scope("conv3", reuse=True) :
  conv3c = encode( conv3b , 4 , 8 )
with tf.variable_scope("deconv3", reuse=True) :
  deconv3a = decode(conv3c , 8 , 4 )
with tf.variable_scope("deconv2", reuse=True) :
  deconv3b = decode(deconv3a , 4 , 2 )
with tf.variable_scope("deconv1", reuse=True) :
  deconv3c = decode(deconv3b , 2 , 1 )

x_out_3 = deconv3c

with tf.variable_scope("conv1", reuse=True) :
  conv2a = encode( x_noisy , 1 , 2 )
with tf.variable_scope("conv2", reuse=True) :
  conv2b = encode( conv2a , 2 , 4 )
with tf.variable_scope("deconv2", reuse=True) :
  deconv2a = decode(conv2b , 4 , 2 )
with tf.variable_scope("deconv1", reuse=True) :
  deconv2b = decode(deconv2a , 2 , 1 )

x_out_2 = deconv2b

with tf.variable_scope("conv1", reuse=True) :
  conv1a = encode( x_noisy , 1 , 2 )

with tf.variable_scope("deconv1", reuse=True) :
  deconv1a = decode(conv1a , 2 , 1 )

x_out_1 = deconv1a


loss_1_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_1 ) ) )
loss_2_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_2 ) ) )
loss_3_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_3 ) ) )
loss_4_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_4 ) ) )
loss_5_raw = tf.log( tf.reduce_mean( tf.square( x_in - x_out_5 ) ) )

loss_1 = loss_1_raw
loss_2 = loss_2_raw
loss_3 = loss_3_raw
loss_4 = loss_4_raw
loss_5 = loss_5_raw
loss_6 = loss_5_raw * 4 + loss_4_raw * 3 + loss_3_raw * 2 + loss_2_raw + loss_1_raw

update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
  train_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
  train_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)
  train_4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_4)
  train_5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_5)
  train_6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_6)


sess = tf.Session()

def resetSession() :
  sess.run( tf.global_variables_initializer() )

resetSession()





def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)

mnist = get_mnist_data()




def doEpochOfTraining( loss, train, batches=55000/100, batch_size=100, rate=LEARNING_RATE ) :
  for index in range(1,batches+1) :
    result,_ = sess.run( [loss,train], feed_dict={x0:mnist.train.next_batch(batch_size)[0],learning_rate:rate})
    if index == 1 or index == batches :
        print "index :",index,", loss:", result




















import falcon
import json
import random
import os.path
from scipy import spatial

def getImageWithIndex(index) :
    return mnist.test.images[index:index+1]

def getExample(index,layer) :
    return sess.run(layer,feed_dict={x0:getImageWithIndex(index)} ).reshape([SIZE,SIZE])

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



class LayerImage:
    def on_get(self, req, resp, layer, index, junk) :
      try :
        ml_layer = [x_noisy,x_out_1,x_out_2,x_out_3,x_out_4,x_out_5,x_in][int(layer)]
        falconRespondArrayAsImage( 
          getExample(int(index),ml_layer) , 
          resp 
          )
      except :
        pass


class DoLearning:
    def on_get(self, req, resp, index) :
        print "TRAINING WITH",index
        doEpochOfTraining([loss_1,loss_2,loss_3,loss_4,loss_5,loss_6][int(index)],[train_1,train_2,train_3,train_4,train_5,train_6][int(index)])
        resp.body = json.dumps( { 'response': 'done'} )

class ResetSession:
    def on_get(self, req, resp) :
      resetSession();
      resp.body = json.dumps( { 'response': 'done'} )


all_embeddings = []

def calculateDistance(index1,index2) :
    return spatial.distance.cosine( all_embeddings[index1], all_embeddings[index2] )

def updateEmbeddings() :
    all_embeddings = sess.run(conv5e,feed_dict={x0:mnist.test.images} ).reshape([-1,SIZE])

def nearestNeighbour(index) :
    index_list = range(len(mnist.test.images))
    distances = np.array([ calculateDistance(index,other) for other in index_list ])
    nearest = np.argsort( distances )[:10]
    return np.array(index_list)[nearest]

class Similar:
    def on_get(self, req, resp, index):
        names = nearestNeighbour(index).tolist()
        resp.body = json.dumps( { 'response' , names } )


print """
================================
Add the endpoints to the service
================================
"""
api = falcon.API()
api.add_route('/ping', Ping())
api.add_route('/view/{file_name}', Display())
api.add_route('/layer{layer}/{index}/{junk}', LayerImage())
api.add_route('/learn/{index}', DoLearning())
api.add_route('/resetSession', ResetSession())






