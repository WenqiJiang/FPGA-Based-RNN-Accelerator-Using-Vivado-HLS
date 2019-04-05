"""
example:
	python weights_to_txt.py --iname="./simple_rnn_1_bias.h5" \
					--dataset="simple_rnn_1/bias:0" \
					--dim=1
					--length1=128
"""

import h5py
import argparse  


if __name__ == "__main__":

	parser = argparse.ArgumentParser() 
	parser.add_argument('--iname', type=str, default="", 
		help="the input H5 file") 
	parser.add_argument('--dataset', type=str, default="", 
		help="the dataset name we want to print") 
	parser.add_argument('--dim_num', type=int, default=0, 
		help="the dimension number of the weights, e.g. 1 or 2") 
	parser.add_argument('--length1', type=int, default=0, 
		help="length of dimension 1") 
	parser.add_argument('--length2', type=int, default=0, 
		help="length of dimension 2") 

	args = parser.parse_args() 
	iname = args.iname 
	dname = args.dataset
	dim_num = args.dim_num
	length1 = args.length1
	length2 = args.length2

	f = h5py.File(iname, 'r')
	dataset = f[dname]
	
	if dim_num == 1:
		for i in range(length1):
			print("%.30f"%(dataset[i]), end=" ")

	if dim_num == 2:
		for i in range(length1):
			for j in range(length2):
				print("%.30f"%(dataset[i][j]), end=" ")
