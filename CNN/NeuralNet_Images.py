import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from custom_models import Regular, FSRS
import pickle

loaded_train_images : list[list] = [None]*10 # pyright: ignore[reportAssignmentType]

for i in range(10):
    with open("MNIST/All"+str(i)+"s.pkl", "rb") as infile:
        loaded_train_images[i] = pickle.load(infile)

def load_train_images_to_Array(arr:list, targetLabel:int, maxSize:int=99999999999) -> None:
    with open("MNIST/All"+str(targetLabel)+"s.pkl", "rb") as infile:
        train_images = pickle.load(infile)

    for i in range(len(train_images)):
        if len(arr) >= maxSize:
            return
        arr.append(train_images[i])

def test(majorityDigit: int, minorityDigit: int, majorityDigitCount:int = 5421, minorityDigitCount:int=2711, fillerCount:int=8132):
    assert majorityDigitCount >= minorityDigitCount
    train_images_majority = []
    train_images_minority = []
    train_images_filler = []

    load_train_images_to_Array(train_images_majority, targetLabel=majorityDigit, maxSize=majorityDigitCount)
    load_train_images_to_Array(train_images_minority, targetLabel=minorityDigit, maxSize=minorityDigitCount)
    for i in range(10):
        if i == majorityDigit or i == minorityDigit:
            continue
        load_train_images_to_Array(train_images_filler, targetLabel=i)
    
    newIndexes = list(range(len(train_images_filler)))
    shuffle(newIndexes)
    train_images_filler_shuffled = [train_images_filler[newIndexes[i]] for i in range(fillerCount)]
    train_images_array = train_images_majority + train_images_minority + train_images_filler_shuffled
    train_labels_array = [1]*(majorityDigitCount+minorityDigitCount) + [0]*fillerCount
    newIndexes = list(range(majorityDigitCount+minorityDigitCount+fillerCount))
    shuffle(newIndexes)
    train_images = np.array([train_images_array[newIndex] for newIndex in newIndexes])
    train_labels = np.array([train_labels_array[newIndex] for newIndex in newIndexes])

    train_images = train_images / 255 #normalize color values to floats between 0 and 1

    model = Regular([
        Flatten(input_shape=(28,28)),
        Dense(4, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    #compile model
    model.compile(optimizer='adam',
                loss=BinaryCrossentropy(), 
                metrics=[BinaryAccuracy()])

    model.fit(train_images, train_labels, epochs=2)

    with open("MNIST/Test"+str(majorityDigit)+"s.pkl", "rb") as infile:
        test_images_majority = pickle.load(infile)

    test_labels_majority = [1] * len(test_images_majority)

    test_loss, test_acc = model.evaluate(np.array(test_images_majority),  np.array(test_labels_majority), verbose=0) # type: ignore

    print(f'\nMajority digit {majorityDigit}: {round(test_acc*len(test_labels_majority))} correct, {round((1-test_acc)*len(test_labels_majority))} incorrect')

    with open("MNIST/Test"+str(minorityDigit)+"s.pkl", "rb") as infile:
        test_images_minority = pickle.load(infile)

    test_labels_minority = [1] * len(test_images_minority)

    test_loss, test_acc = model.evaluate(np.array(test_images_minority),  np.array(test_labels_minority), verbose=0) # type: ignore

    print(f'\nMinority digit {minorityDigit}: {round(test_acc*len(test_labels_minority))} correct, {round((1-test_acc)*len(test_labels_minority))} incorrect')

    return {"majority" : [round((1-test_acc)*len(test_labels_majority)), round(test_acc*len(test_labels_majority))],
            "minority" : [round((1-test_acc)*len(test_labels_minority)), round(test_acc*len(test_labels_minority))]}

test(majorityDigit=0,minorityDigit=4)
exit()

for i in range(0,10):
    for j in range(0, 10):
        if i == j:
            continue
        log = test(majorityDigit=i,minorityDigit=j)