import datetime
import math

import tensorflow as tf


from nets import Conv_net

from data_factory import Timeseries_Factory
class Run_network():

    def __init__(self):

        # Create an instance of the model
        # self.model = LSTM_net_direct_param()
        # self.model = LSTM_net()
        self.model = Conv_net()

        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.MeanAbsolutePercentageError(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.MeanAbsolutePercentageError(name='test_accuracy')
        self.epoch = 0

        # tensorboard writer
        self.writer = tf.summary.create_file_writer("logs/")


    def forward_loop_train(self, batch_signal, batch_labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions, _, _ = self.model(batch_signal, training=True)
            loss = self.loss_object(batch_labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        return predictions, loss, gradients

    # function that defines training
    @tf.function
    def train_step(self, signal):
        batch_no = 0
        # loop over batches
        for batch_step, (batch_signal, batch_labels) in enumerate(signal):

            predictions, loss, gradients = self.forward_loop_train(batch_signal, batch_labels)

            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # calculate loss and train accuracy (same in this case) for this epoch
            self.train_loss(loss)
            self.train_accuracy(batch_labels, predictions)

            # show progress
            template = 'Batch {}, Loss: {}, Accuracy: {}'
            batch_no += 1
            print(template.format(batch_no,
                                  self.train_loss.result(),
                                  self.train_accuracy.result()))

        with self.writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=self.epoch)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=self.epoch)

    # function that defines testing
    @tf.function
    def test_step(self, signal, label):

        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, timeseries_params, layer_vals = self.model(signal, training=False)
        t_loss = self.loss_object(label, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(label, predictions)

        with self.writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=self.epoch)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=self.epoch)

        return label, predictions, timeseries_params, layer_vals

    # Train and test network each epoch
    def run_epoch(self, signal_set, test_signal_set):

        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        # train step

        # tensorboard trace
        if self.epoch == 0: tf.summary.trace_on(graph=True, profiler=True)
        self.train_step(signal_set)
        # stop tensorboard tracing
        if self.epoch == 0:
            with self.writer.as_default():
                tf.summary.trace_export("train graph", step=self.epoch, profiler_outdir="logs")

        # test step
        # tensorboard trace
        if self.epoch == 0: tf.summary.trace_on(graph=True, profiler=True)
        labels, predictions, timeseries_params, layer_vals = self.test_step(test_signal_set[0], test_signal_set[1])
        # stop tensorboard tracing
        if self.epoch == 0:
            with self.writer.as_default():
                tf.summary.trace_export("test graph", step=self.epoch, profiler_outdir="logs")

        # show progress
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        self.epoch += 1
        print(template.format(self.epoch,
                              self.train_loss.result(),
                              self.train_accuracy.result(),
                              self.test_loss.result(),
                              self.test_accuracy.result()))

        diagnostics = [self.epoch, self.train_loss.result(), self.test_loss.result(), self.test_accuracy.result()]
        # write logs
        # self.writer.flush()

        return labels, predictions, timeseries_params, layer_vals, diagnostics


if __name__ == "__main__":

    import numpy as np

    # create 2 timeseries sample

    Data = Timeseries_Factory()
    t = Data.time_signal(100)
    scaler1 = 0.5; sig1 = Data.sinusoids(t, scaler1)
    scaler2 = 1; sig2 = Data.sinusoids(t, scaler2)
    test_scaler1 = 0.75; test_sig1 = Data.sinusoids(t, test_scaler1)

    # re-shape data
    sig = tf.reshape(np.array(sig1, dtype="float32"), [1, 100, 1]).numpy()
    scaler = tf.reshape(np.array(scaler1, dtype="float32"), [1, 1, 1]).numpy()
    signal_set = [sig, sig]

    test_sig = tf.reshape(np.array(test_sig1, dtype="float32"), [1, 100, 1]).numpy()
    test_scaler = tf.reshape(np.array(test_scaler1, dtype="float32"), [1, 1, 1]).numpy()
    test_signal_set = [test_sig, test_sig]

    # define model
    NN_lstm = Run_network()

    labels, prediction,timeseries_params = NN_lstm.run_epoch(signal_set, test_signal_set)

