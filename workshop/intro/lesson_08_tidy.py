# -- imports --
import tensorflow as tf
import math
from importlib import import_module
import argparse

# -- command line arguments --
parser = argparse.ArgumentParser()
parser.add_argument('--model', nargs='?', default="poly_model", help="The tensorflow model")
parser.add_argument('--data', nargs='?', default="poly_data", help="The data file with x & y")
parser.add_argument('--plot', action='store_const', const=True, default=False, help="Plot the data")
parser.add_argument('--iterations', nargs='?', type=int, default=10000, help="The number of iterations")
args = parser.parse_args()

# -- get the model and data --
data_directory = "lesson_08"
model = import_module(data_directory + "." + args.model)
data = import_module(data_directory + "." + args.data)

# start a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for iteration in range(1, args.iterations + 1):

    # learn
    sess.run(model.learn, feed_dict={model.x: data.x, model.y: data.y})

    # print feedback once in a while
    if iteration == 1 or iteration == 10 or iteration == 100 or iteration % 1000 == 0:
        print "iteration =", iteration, ", RMS error is =",
        print sess.run(model.rms_error, feed_dict={model.x: data.x, model.y: data.y})


print "\ndone training\n"
print "The equation is f(x) =",
model.printEquation(sess)


if args.plot:
    print "\nGenerating estimates\n"
    estimate = sess.run(model.fx, feed_dict={model.x: data.x})
    import matplotlib.pyplot as plt
    plt.plot(data.x, data.y)
    plt.plot(data.x, estimate)
    plt.grid(True)
    plt.show()
