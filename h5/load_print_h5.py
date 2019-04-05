"""
example:
	python load_print_h5.py --iname=./simple_rnn_1_bias.h5
"""

import h5py
import argparse  


if __name__ == "__main__":

	parser = argparse.ArgumentParser() 
	parser.add_argument('--iname', type=str, default="", 
		help="the input H5 file") 
	args = parser.parse_args() 
	iname = args.iname 

	f = h5py.File(iname, 'r')
	
	print(type(f))
