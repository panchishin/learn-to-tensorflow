import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
np.set_printoptions(precision= # TODO

# The data

x = # TODO
y = # TODO


filename = # TODO

try :
    # Try and restore the graph
    model = # TODO
    print(f"\nRetrieved model '{filename}' from disk")

except :
    # Define the graph

    layer1 = # TODO
    layer2 = # TODO
    model = # TODO


    # Turn the graph into a model

    model.compile(loss= # TODO
                  optimizer= # TODO

    model.fit(x, y, steps_per_epoch= # TODO

    tf.keras.models.save_model(model,filepath= # TODO
    print(f"\nSaved model to disk as '{filename}'")
    print("\n** RUN SCRIPT AGAIN TO USE SAVED MODEL **")


# Show the results

print("\nExpected output")
print("    ",y.reshape([-1]))

print("\nPredicted output from model")
print("    ",model.predict(x).reshape([-1]))

print("\nModel weights")
for name,w in zip("m1 b1 m2 b2".split(" "),model.get_weights()) :
    print("    ",name,w.reshape([-1]))

print()
