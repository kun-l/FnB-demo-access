import tensorflow as tf
from data_factory import Timeseries_Factory

class Sinusoid_Layer(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=1):
        super(Sinusoid_Layer, self).__init__()
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(
        #     initial_value=w_init(shape=(input_dim, units), dtype="float32"),
        #     trainable=False,
        # )
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(
        #     initial_value=b_init(shape=(units,), dtype="float32"), trainable=False
        # )

    def call(self, inputs):
        # generate timeseries based on prediction
        t = Timeseries_Factory().time_signal(n_points=100)
        # inputs = inputs.eval()
        print(inputs)
        input_sig = Timeseries_Factory().sinusoids(t, inputs)


        return input_sig