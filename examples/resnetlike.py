import tensorflow as tf
import time

execfile("layer.py")


def get_mnist_data() :
  from tensorflow.examples.tutorials.mnist import input_data
  return input_data.read_data_sets('./cache', one_hot=True)


def handler_wrapper( handler ) :
  # inputs
  tf.reset_default_graph()
  size         = 28
  x            = tf.placeholder(tf.float32, [None, size*size])
  y_           = tf.placeholder(tf.float32, [None, 10])
  keep_prob    = tf.placeholder(tf.float32)
  learning_rate = tf.placeholder(tf.float32)
  training     = tf.placeholder(tf.bool)
  h_pool       = tf.reshape(x, [-1,size,size,1])
  layers       = handler.convolve( h_pool, training,  keep_prob )
  y            = tf.reshape(layers[-1], [-1,10])
  loss         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train        = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  percent_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return x,y_,keep_prob,loss,train,percent_correct,training,learning_rate,layers


def train_model_and_report(mnist,model,learning_rate_value=1e-4,epochs=200,keep_prob_value=0.5):
  print "MODEL :",model.__class__.__name__
  x,y_,keep_prob,loss,train,percent_correct,training,learning_rate,layers = handler_wrapper( model )

  with tf.Session() as sess :
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.local_variables_initializer() )

    results = sess.run(layers, feed_dict={x:mnist.train.next_batch(4)[0], keep_prob:1.0, training:False})
    for result,stage in zip(results,layers) :
       print result.shape , stage.name


    items_per_batch = 200
    start_time = time.time()

    lambda_error = 100.0
    lambda_val   = 0.3

    for epoch in range(1,epochs+1) :
      if (epoch+1) % (epochs / 4) == 0 :
        learning_rate_value *= .5
        #print "Decreasing learning_rate to", learning_rate_value
        
      training_loss = 0.0
      for _ in range( mnist.test.labels.shape[0] / items_per_batch ) :
        batch_xs, batch_ys = mnist.train.next_batch(items_per_batch)
        result_loss,_ = sess.run([loss,train], feed_dict={
          x: batch_xs, y_: batch_ys, 
          keep_prob: keep_prob_value, training:True, learning_rate:learning_rate_value})
        training_loss += result_loss

      training_loss *= 1.0 * items_per_batch / mnist.test.labels.shape[0]

      if epoch % 5 == 0 :
        print "\tEpoch %5d"%epoch,", Training Loss %0.6f"%training_loss,
        correct = 0.0
        test_loss = 0.0
        for test_batch in range(100) :
          test_batch_start = test_batch * 100
          test_batch_end   = test_batch_start + 100
          result_loss,result_correct = sess.run([loss,percent_correct], feed_dict={
            x: mnist.test.images[test_batch_start:test_batch_end], 
            y_: mnist.test.labels[test_batch_start:test_batch_end], 
            keep_prob:1.0, training:False})
          test_loss += result_loss
          correct += result_correct
        test_loss /= 100.0
        if lambda_error == 100.0 :
          lambda_error = ( 100.0 - correct )
        else :
          lambda_error = lambda_error * ( 1.0 - lambda_val ) + ( 100.0 - correct ) * lambda_val
        print ", Test Loss %0.4f"%test_loss,
        print ", Train/Test %5.3f"%(training_loss/test_loss),", Percent Error = %4.2f" % (100.0 - correct),
        print ", Lambda Error %4.2f" % lambda_error,
        print ", ",int(time.time() - start_time),"seconds"
        if training_loss / test_loss < 0.01 :
          print "End condition met.  Rule 'training_loss / test_loss < 0.01'.  Possible overfitting."
          return lambda_error

  return lambda_error


mnist = get_mnist_data()


class moreresnet() :
  def convolve( self, image, training,  keep_prob, reses=1 ) :
    result = []
    result.append( batch_normalization( image , training ) )
    # 1 input convolution
    result.append( conv_relu( result[-1] , 1 , 16 , width=5, padding='VALID' ) )
    for _ in range(reses):
      result.append( resnet_block( result[-1] , 16, 3 , training, momentum=0.99 ) )
    result.append( resnet_downscale( result[-1] ) )
    for _ in range(reses):
      result.append( resnet_block( result[-1] , 32, 3 , training, momentum=0.99 ) )
    result.append( resnet_downscale( result[-1] ) )
    for _ in range(reses):
      result.append( resnet_block( result[-1] , 64, 3 , training, momentum=0.99 ) )
    result.append( resnet_downscale( result[-1] ) )
    for _ in range(reses):
      result.append( resnet_narrow( result[-1] , 128, 3 , training, narrowing=4, momentum=0.99 ) )
    result.append( avg_pool( result[-1] , stride=3 ) )
    result.append( tf.nn.dropout( result[-1] , keep_prob ) )
    # 1 output convolution
    result.append( conv_relu( result[-1] , 128 , 10 , width=1, padding='VALID' ) )
    return result



train_model_and_report( mnist, moreresnet() , epochs=200 , learning_rate_value=1e-4)
