"""
example:
	python splitH5.py --iname=./pre-trained-rnn.h5 \
					--dataset=/model_weights/dense_1/dense_1/bias:0 \
					--oname=./dense_1_bias.h5 \
					--new_dataset=dense_1/bias:0
"""

import h5py
import argparse  


if __name__ == "__main__":

	parser = argparse.ArgumentParser() 
	parser.add_argument('--iname', type=str, default="", 
		help="the input H5 file") 
	parser.add_argument('--dataset', type=str, default="", 
		help="the dataset we want to split") 
	parser.add_argument('--oname', type=str, default="", 
		help="the output H5 file") 
	parser.add_argument('--new_dataset', type=str, default="", 
		help="the dataset we want to generate in the new H5 file") 
	args = parser.parse_args() 
	iname = args.iname 
	dname = args.dataset
	oname = args.oname
	ndname = args.new_dataset

	f = h5py.File(iname, 'r')
	dataset = f[dname]
	h5f = h5py.File(oname, 'w')
	h5f.create_dataset(ndname, data=dataset)
	h5f.close()

	# f = h5py.File('./pre-trained-rnn.h5', 'r')
	# dataset = f['model_weights/dense_1/dense_1/bias:0']
	# h5f = h5py.File('dense_1_bias.h5', 'w')
	# h5f.create_dataset('dense_1/bias:0', data=dataset)
	# h5f.close()