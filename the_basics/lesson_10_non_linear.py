# -- imports --
import tensorflow as tf
from tensorflow.compat.v1 import Session, global_variables_initializer
from importlib import import_module
import argparse

# -- command line arguments --
parser = argparse.ArgumentParser()
parser.add_argument('--model', nargs='?', default="linear_model", help="The tensorflow model")
parser.add_argument('--data', nargs='?', default="xor_data", help="The data file with x & y")
parser.add_argument('--iterations', nargs='?', type=int, default=1000, help="The number of iterations")
args = parser.parse_args()

# -- get the model and data --
data_directory = "lesson_10"
model = import_module(data_directory + "." + args.model)
data = import_module(data_directory + "." + args.data)

# start a session
sess = Session()
sess.run(global_variables_initializer())

print()
print(f"Using model {args.model} on the dataset {args.data}")
print()
for iteration in range(1, args.iterations + 1):

    # learn
    sess.run(model.learn, feed_dict={model.x: data.x, model.y: data.y})

    # print(feedback once in a while)
    if iteration == 1 or iteration == 10 or iteration == 100 or iteration % 1000 == 0:
        print(f"iteration {iteration:5}, RMS error = {sess.run(model.rms_error, feed_dict={model.x: data.x, model.y: data.y}):.2f}")


print("\ndone training\n")
print("The equation is")
model.printEquation(sess)

print("\nGenerating estimates\n")
estimate = sess.run(model.fx, feed_dict={model.x: data.x})
print("The real vs estimate is")
print("[x1 , x2 ] y   fx")
for x, y, fx in zip(data.x, data.y, estimate.tolist()):
    print(x, y, round(fx, 1))

print()