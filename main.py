print('begin')
import os, sys
import numpy as np

output_stream = sys.stdout
'''============================================================================
create data
============================================================================'''
from data_factory import Timeseries_Factory
import tensorflow as tf
n_points = 100
n_signals = int(1e3)
batch_size = int(1e2)

test_train = 0.7
n_train = int(test_train * n_signals)

Data = Timeseries_Factory()

# define angle points
theta = Data.time_signal(n_points)

# generate signals
c1 = []; test_c1 = []
c2 = []; test_c2 = []
sig = []; test_sig = []
for idx in range(n_signals):

    # display progress
    # output_stream.write('making data... %s\r' % (idx/n_signals*100))
    # output_stream.flush()
    print('making data... %d' % (idx/n_signals*100), end='\r')

    # scaler for theta
    c1_gen = np.random.uniform(0.1, 0.9, 1)
    c2_gen = np.random.uniform(0.4, 0.7, 1)

    if idx < (n_train):
        c1.extend(c1_gen)
        c2.extend(c2_gen)
        sig.append(Data.sinusoids(theta, c1_gen[0], c2_gen[0]))
    else:
        test_c1.extend(c1_gen)
        test_c2.extend(c2_gen)
        test_sig.append(Data.sinusoids(theta, c1_gen[0], c2_gen[0]))

# for conv
# define datasets
sig = tf.reshape(np.array(sig,dtype="float32"), [n_train, 1, n_points]).numpy()
test_sig = tf.reshape(np.array(test_sig,dtype="float32"), [n_signals-n_train, 1, n_points]).numpy()
test_dataset = [test_sig, test_sig]

train_dataset = tf.data.Dataset.from_tensor_slices((sig, sig))
train_dataset = train_dataset.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True).batch(batch_size)

'''============================================================================
nets
============================================================================'''

from propagator import Run_network
tf.random.set_seed(1234)
nn_pde = Run_network()
for _ in range(50):
    labels, prediction, timeseries_params, layer_vals, diagnostics = nn_pde.run_epoch(train_dataset, test_dataset)

print('done')