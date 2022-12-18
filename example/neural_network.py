import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import DatasetGenerator

X, y = DatasetGenerator.Positive_Negative_data(1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = nn.models.Sequential([
            nn.layers.Layer_Dense(2, 128, activation=nn.activations.ReLU()),
            nn.layers.Layer_Dense(128, 2, activation=nn.activations.Softmax()),
        ])

model.fit(X_train, y_train.astype(np.uint8), epoch=2, iteration=5000, smooth_output=False, verbose=True)
model.save("MyModel_pn")
model = nn.models.load_model(model_name="MyModel_pn")
predictions = model.predict(X_test)
print(np.mean([np.argmax(prediction, axis=2) for prediction in predictions] == y_test))

# Has accuracy of 99.9978%
