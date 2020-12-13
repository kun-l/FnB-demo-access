from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, LayerNormalization, Dropout, Conv1D
from tensorflow.keras import Model

from layers import Sinusoid_Layer

class Conv_net(Model):
    def __init__(self):
        super(Conv_net, self).__init__()

        self.layerNorm = LayerNormalization(axis=1, center=True, scale=True)
        self.dropout = Dropout(.3)

        self.conv1 = Conv1D(100, 2, strides=1, activation='tanh', padding='causal')

        self.d1 = Dense(20, activation='relu')
        self.d2 = Dense(40, activation='relu')

        layer_width = int(50)
        self.d50_1 = Dense(layer_width, activation='relu')

        self.primer = Dense(10, activation='sigmoid')
        self.coefficent_layer = Dense(1)
        self.sin = Sinusoid_Layer()


    def call(self, x):
        layer_vals = []

        '''
        ============================================================
        conv layers
        ============================================================
        '''
        x = self.conv1(x)
        # x = self.layerNorm(x)
        layer_vals.append(x)

        '''
        ============================================================
        fcl layers
        ============================================================
        '''

        x = self.dropout(self.d50_1(x))
        layer_vals.append(x)

        '''
        ============================================================
        reconstruction layers
        ============================================================
        '''

        x = self.primer(x)
        layer_vals.append(x)

        x = self.coefficent_layer(x)
        timeseries_params = x

        return self.sin(x), timeseries_params, layer_vals

