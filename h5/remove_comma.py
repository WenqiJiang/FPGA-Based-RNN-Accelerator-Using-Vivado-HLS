"""
example:
	python remove_comma.py --fname=./embedding_1_embeddings.txt --print_length=True
"""

import argparse 

if __name__ == "__main__":

	parser = argparse.ArgumentParser() 
	parser.add_argument('--fname', type=str, default="", 
		help="the input txt file name") 
	parser.add_argument('--print_length', type=bool, default=True,
		help="whether to print length of input data")
	args = parser.parse_args() 
	fname = args.fname 
	print_length = args.print_length

	with open(fname, 'r') as myfile:
	    data = myfile.read().split(",")
	    if print_length:
	    	count = len(data)
	    	print("length of {}: {}".format(fname[:-4], count))

	with open(fname, 'w') as myfile:
	    for elements in data:
	        myfile.write("%s" % elements)