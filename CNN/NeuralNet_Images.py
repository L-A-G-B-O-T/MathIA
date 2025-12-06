import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import SparseCategoricalCrossentropy
from matplotlib import pyplot as plt
import pickle

with open("MNIST/All.pkl", "rb") as outfile:
    train_images, train_labels = pickle.load(outfile)

print("Frequencies: ")


train_images = train_images / 255 #normalize color values to floats between 0 and 1

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(10),
])

#compile model
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True), #don't worry too much about logits
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

with open("MNIST/Test.pkl", "rb") as outfile:
    test_images, test_labels = pickle.load(outfile)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)