import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, Input

model = tf.keras.Sequential([
    Input(shape=(1,)),  # Define the input shape here
    Dense(units=1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

#Datas for model
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


# Training the model
model.fit(xs, ys, epochs=500)

# Making a prediction
print(model.predict(np.array([10.0])))

#so the machine will learn that Y=3x+1 equation after 500 epochs XD
