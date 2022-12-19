import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import DatasetGenerator

X, y = DatasetGenerator.sine_wave_data(1000, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = nn.models.Sequential([
            nn.layers.Layer_Dense(2, 128, activation=nn.activations.ReLU()),
            nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU()),
            nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU()),
            nn.layers.Layer_Dense(128, 3, activation=nn.activations.Softmax()),
        ])

model.fit(X_train, y_train.astype(np.uint8), epoch=2, iteration=5000, smooth_output=False, verbose=True)
model.save("SineWaveModel")
model = nn.models.load_model(model_name="SineWaveModel")
predictions = model.predict(X_test)
print(np.mean([np.argmax(prediction, axis=2) for prediction in predictions] == y_test))

