## Lesson 0 - add 1 to a variable 10 times

import tensorflow as tf

# set state to a variable set to 0
state = tf.Variable(0)

# set one to a constant set to 1
one = tf.constant(1)

# update phase adds state and one and then assigns to state
update = tf.assign(state, tf.add(state, one) )

# create a session
with tf.Session() as sess:
  # initialize session variables
  sess.run( tf.initialize_all_variables() )

  print "The starting state is",sess.run(state)

  print "Run the update 10 times..."
  for count in range(10):
    # execute the update
    sess.run(update)

  print "The end state is",sess.run(state)

