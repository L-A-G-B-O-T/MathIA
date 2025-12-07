from keras.models import Sequential
import tensorflow as tf
from fsrs import Scheduler, Card, Rating, ReviewLog
import heapq
import numpy as np

class Regular(Sequential):
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
    def __init__(self, x, y, card=Card()):
        self.x = x
        self.y = y
        self.card = card
    def __lt__(self, other):
        return self.card.due < other.card.due

class FSRS(Sequential):
    def __init__(self):
        super().__init__()
        self.scheduler = Scheduler()

    def train(self, x, y, epochs: int, batch_size: int):
        #set up min heap
        self.pq = []#min heap that puts the first due training card at the front. 
        for xi, yi in zip(x, y):
            heapq.heappush(self.pq, FSRS_Train_Card(xi, yi))
        
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")
            stepsPerEpoch: int = len(x) // batch_size + 1
            for stepID in range(stepsPerEpoch):
                ##put cards into batches 
                batch_x = np.empty((batch_size, 28, 28))
                batch_y = np.empty(batch_size)
                batch_card = np.empty(batch_size)
                for i in range(batch_size):
                    train_card = heapq.heappop(self.pq)
                    batch_x[i] = train_card.x
                    batch_y[i] = train_card.y
                    batch_card[i] = train_card.card

                with tf.GradientTape() as tape:
                    y_predict = self(batch_x, training=True)
                    loss = self.compute_loss(y=batch_y, y_pred=y_predict)
                
                quality = self.rate(loss)
                review_logs = np.array([None] * batch_size)

                for i in range(batch_size):    
                    batch_card[i], review_logs[i] = self.scheduler.review_card(batch_card[i], quality[i])
                    heapq.heappush(self.pq, FSRS_Train_Card(batch_x[i], batch_y[i], batch_card[i]))
                
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                self.optimizer.apply(gradients, trainable_vars)

                if stepID % 100 == 0:
                    print(
                        f"Training loss (for 1 batch) at step {stepID}: {float(loss):.4f}"
                    )
                    print(f"Seen so far: {(stepID + 1) * batch_size} samples")

    def rate(self, loss_value): #assumes the loss is based off sparse categorical crossentropy for MNIST dataset
        if (loss_value > 1):
            return Rating.Again
        elif (loss_value > 0.2):
            return Rating.Hard
        elif (loss_value > 0.05):
            return Rating.Good
        return Rating.Easy

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

        
class SM2(Sequential):
    pass
