import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np 

labels = [
    'red-ish',
    'green-ish',
    'blue-ish',
    'pink-ish',
    'purple-ish',
    'orange-ish',
    'brown-ish',
    'grey-ish',
    'yellow-ish'
]

model = load_model('colorClassifier.model')

# prepare data to predict
x = np.array([43, 178, 26])
x = x / 255.0 # normalize before feeding into network
x = x.reshape((1, len(x)))

pred = model.predict(x)
color = np.argmax(pred)
print(labels[color]) # green-ish