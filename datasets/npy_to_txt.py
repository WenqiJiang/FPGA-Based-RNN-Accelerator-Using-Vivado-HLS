"""
example:
	python npy_to_txt.py --fname="./org_seq.npy" --print_length=True
"""

import argparse 
import numpy as np

if __name__ == "__main__":

	parser = argparse.ArgumentParser() 
	parser.add_argument('--fname', type=str, default="", 
		help="the input npy file name") 
	parser.add_argument('--print_length', type=bool, default=True,
		help="whether to print length of input data")
	args = parser.parse_args() 
	fname = args.fname 
	print_length = args.print_length

	data = np.load(fname)
	dim = len(data.shape)
	data.tolist() # 1D or 2D list
	if dim == 2:
		# if 2D list, flat it to 1D
		data = [item for sublist in data for item in sublist]

	if print_length:
	    print("length of {}: {}".format(fname, len(data)))

	with open(fname[:-3] + "txt", 'w') as myfile:
	    for elements in data:
	        myfile.write("%s\n" % elements)