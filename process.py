import json
import numpy as np 
import tensorflow as tf
import pickle
# process data

X = []
y = []

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
with open('colorData.json') as json_file:
    data = json.load(json_file)
    entries = data['entries']
    # print(len(entries))
    for entry in entries:
        features = [entry['r'], entry['g'], entry['b']]
        label = entry['label']
        X.append(features)
        y.append(labels.index(label))

X = np.array(X)
y = np.array(y)

# one-hot encoding
Y = np.zeros((len(y), 9))
Y[np.arange(len(y)), y] = 1

# save data
pickle.dump(X, open('X.pickle', 'wb'))
pickle.dump(Y, open('Y.pickle', 'wb'))


