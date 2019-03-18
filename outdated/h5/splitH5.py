import h5py

f = h5py.File('./pre-trained-rnn.h5', 'r')
dataset = f['model_weights/dense_1/dense_1/bias:0']
h5f = h5py.File('dense_1_bias.h5', 'w')
h5f.create_dataset('dense_1/bias:0', data=dataset)
h5f.close()
