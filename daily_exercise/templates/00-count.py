# imports
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# set state to a variable set to 0
state = # TODO

# set one to a constant set to 1
one = # TODO

# update phase adds state and one and then assigns to state
addition = # TODO
update = # TODO

# create a session
with tf.compat.v1.Session() as sess:
    # initialize session variables
    # TODO session execution command here

    print("The starting state is", # TODO session execution command here

    print("Run the update 10 times...")
    for count in range(10):
        # execute the update
        # TODO session execution command here

    print("The end state is", # TODO session execution command here
