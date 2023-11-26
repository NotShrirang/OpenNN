"""
An example of using the neural network library to train a model to classify
a set of sine waves.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import dataset_generator as dg
import nn

X, y = dg.sine_wave_data(1000, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = nn.models.Sequential([
    nn.layers.Layer_Dense(2, 128, activation=nn.activations.ReLU()),
    nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU()),
    nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU()),
    nn.layers.Layer_Dense(128, 3, activation=nn.activations.Softmax()),
])

model.fit(X_train, y_train.astype(np.uint8), epoch=2,
          iteration=5000, smooth_output=False, verbose=True)
model.save("SineWaveModel")
model = nn.models.load_model(model_path="SineWaveModel")
predictions = model.predict(X_test)
print(np.mean([np.argmax(prediction, axis=2)
      for prediction in predictions] == y_test))
