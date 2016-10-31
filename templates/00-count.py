## Lesson 0 - add 1 to a variable 10 times

import tensorflow as tf

# set state to a variable set to 0
state = # TODO

# set one to a constant set to 1
one = # TODO

# update phase adds state and one and then assigns to state
update = # TODO

# create a session
with tf.Session() as sess:
  # initialize session variables
  # TODO session execution command here

  print "The starting state is",# TODO session execution command here

  print "Run the update 10 times..."
  for count in range(10):
    # execute the update
    # TODO session execution command here

  print "The end state is",# TODO session execution command here
  # the end state should be 10.
