from keras.models import Sequential
import tensorflow as tf
from fsrs import Scheduler, Card, Rating, ReviewLog
import heapq
import numpy as np

class Regular(Sequential):
    def train(self, x, y, epochs:int = 1, batch_size:int=32):
        self.fit(x, y, batch_size=batch_size, epochs=epochs)
        
    def train_step(self, data):
        batch_x, batch_y = data
        with tf.GradientTape() as tape:
            y_predict = self(batch_x, training=True)
            loss = self.compute_loss(y=batch_y, y_pred=y_predict)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply(gradients, trainable_vars)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(batch_y, y_predict)

        return {m.name: m.result() for m in self.metrics}

class FSRS_Train_Card:
    def __init__(self, x, y, card=Card(), id=0):
        self.id = id
        self.x = x
        self.y = y
        self.card = card
    def __lt__(self, other):
        return self.card.due < other.card.due

class FSRS(Sequential):
    def __init__(self, layers):
        super().__init__(layers)
        self.scheduler = Scheduler()
        self.debug = False

    def train_step(self, data):
        batch_x, batch_y, batch_card = data
        batch_size = len(batch_x)

        with tf.GradientTape() as tape:
            y_predict = self(batch_x, training=True)
            loss = self.compute_loss(y=batch_y, y_pred=y_predict)
        
        quality = self.rate_accuracy(y=batch_y, y_pred=y_predict)
        review_logs = np.array([None] * batch_size)

        for i in range(batch_size):    
            batch_card[i], review_logs[i] = self.scheduler.review_card(batch_card[i], quality[i]) # type: ignore
            heapq.heappush(self.pq, FSRS_Train_Card(batch_x[i], batch_y[i], batch_card[i]))
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else: 
                metric.update_state(batch_y, y_predict)

        return {m.name: m.result() for m in self.metrics}

    def fit(self, x, y, epochs: int=1, batch_size: int = 32):
        #set up min heap
        self.pq = []#min heap that puts the first due training card at the front. 
        for id, [xi, yi] in enumerate(zip(x, y)):
            heapq.heappush(self.pq, FSRS_Train_Card(xi, yi, id=id))
        
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")
            stepsPerEpoch: int = len(x) // batch_size + 1
            for stepID in range(stepsPerEpoch):
                ##put cards into batches 
                batch_x = np.empty((batch_size, 28, 28))
                batch_y = np.empty(batch_size)
                batch_card = [Card()] * batch_size
                if self.debug:
                    print(f"Batch {stepID} Contents: ")
                for i in range(batch_size):
                    train_card = heapq.heappop(self.pq)
                    batch_x[i] = train_card.x
                    batch_y[i] = train_card.y
                    batch_card[i] = train_card.card
                    if self.debug:
                        print(train_card.id, end=" ")
                if self.debug:
                    print()

                results = self.train_step((batch_x, batch_y, batch_card))
                for metricname in results:
                    if metricname == "compile_metrics": 
                        for submetricname in results[metricname]:
                            print(f"\t{submetricname}: {float(results[metricname][submetricname]):.4f}")
                    print(f"\t{metricname}: {float(results[metricname]):.4f}")

                print(f"\tSeen so far: {(stepID + 1) * batch_size} samples")

            for metric in self.metrics:
                metric.reset_state()

    def rate_loss(self, loss_value): #assumes the loss is based off sparse categorical crossentropy for MNIST dataset
        if (loss_value > 1):
            return Rating.Again
        elif (loss_value > 0.2):
            return Rating.Hard
        elif (loss_value > 0.05):
            return Rating.Good
        return Rating.Easy
    
    def rate_accuracy(self, y, y_pred):
        pred = np.argmax(y_pred, axis=1)
        ret = [0] * len(y)
        for i in range(len(y)):
            if pred[i] == y[i]:
                ret[i] = Rating.Good
            else:
                ret[i] = Rating.Again
        return ret



        
class SM2(Sequential):
    pass
