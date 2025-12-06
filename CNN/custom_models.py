from keras.models import Sequential
import tensorflow as tf

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

class FSRS(Sequential):
    pass
class SM2(Sequential):
    pass
