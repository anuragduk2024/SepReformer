import tensorflow as tf 
import numpy as np 
import os
import shutil
from datetime import datetime

class SVSRNN(tf.keras.Model):
    def __init__(self, num_features, num_rnn_layer=4, num_hidden_units=[256, 256, 256], tensorboard_directory='graphs/svsrnn', clear_tensorboard=True):
        super(SVSRNN, self).__init__()
        
        assert len(num_hidden_units) == num_rnn_layer

        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_hidden_units = num_hidden_units

        self.gstep = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        
        self.rnn_layers = [tf.keras.layers.GRU(units, return_sequences=True) for units in self.num_hidden_units]
        self.dense_src1 = tf.keras.layers.Dense(self.num_features, activation='relu', name='y_hat_src1')
        self.dense_src2 = tf.keras.layers.Dense(self.num_features, activation='relu', name='y_hat_src2')

        self.gamma = 0.001
        self.optimizer = tf.keras.optimizers.Adam()

        if clear_tensorboard:
            shutil.rmtree(tensorboard_directory, ignore_errors=True)
        self.writer = tf.summary.create_file_writer(tensorboard_directory)
        
    def call(self, inputs, training=False):
        x = inputs
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        y_hat_src1 = self.dense_src1(x)
        y_hat_src2 = self.dense_src2(x)
        
        mask_logits = tf.stack([y_hat_src1, y_hat_src2], axis=-1)
        mask = tf.nn.softmax(mask_logits, axis=-1)
    
        y_tilde_src1 = mask[..., 0] * inputs
        y_tilde_src2 = mask[..., 1] * inputs
        
        return y_tilde_src1, y_tilde_src2

    def loss_fn(self, y_src1, y_src2, y_pred_src1, y_pred_src2):
        return tf.reduce_mean(tf.square(y_src1 - y_pred_src1) + tf.square(y_src2 - y_pred_src2))

    @tf.function
    def train_step(self, x, y1, y2):
        with tf.GradientTape() as tape:
            y_pred_src1, y_pred_src2 = self(x, training=True)
            loss = self.loss_fn(y1, y2, y_pred_src1, y_pred_src2)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, y_pred_src1, y_pred_src2

    def train(self, x, y1, y2, learning_rate):
        self.optimizer.learning_rate = learning_rate
        loss, y_pred_src1, y_pred_src2 = self.train_step(x, y1, y2)
        
        self.summary(x, y1, y2, y_pred_src1, y_pred_src2, loss)
        
        return loss.numpy()

    @tf.function
    def validate(self, x, y1, y2):
        y1_pred, y2_pred = self(x, training=False)
        validate_loss = self.loss_fn(y1, y2, y1_pred, y2_pred)
        return y1_pred, y2_pred, validate_loss

    def test(self, x):
        return self(x, training=False)

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not filename.endswith('.weights.h5'):
            filename = filename.rsplit('.', 1)[0] + '.weights.h5'
        self.save_weights(os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):
        self.load_weights(filepath)

    def summary(self, x, y1, y2, y_pred_src1, y_pred_src2, loss):
        with self.writer.as_default():
            tf.summary.scalar('loss', loss, step=self.gstep)
            tf.summary.histogram('x_mixed', x, step=self.gstep)
            tf.summary.histogram('y_src1', y1, step=self.gstep)
            tf.summary.histogram('y_src2', y2, step=self.gstep)
            tf.summary.histogram('y_pred_src1', y_pred_src1, step=self.gstep)
            tf.summary.histogram('y_pred_src2', y_pred_src2, step=self.gstep)
            self.writer.flush()
        
        self.gstep.assign_add(1)