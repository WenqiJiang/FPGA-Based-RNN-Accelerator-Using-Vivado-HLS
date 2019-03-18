The model structure is shown in train-embeddings-rnn-50.png

This model is trained with pre-trained GloVe word vectors. For those words don't in the dataset, we randomly initialized them and train them. The whole embedding layer in this model is trainable.

To split the H5 file into individual components (split it into weights matrix and bias vector in each layers), we first use a package called h5dump to analyze the h5 file. Then, we use a python program to split it.

(1) Analyze the components in the H5 file
	h5dump -n train-embeddings-rnn-50.h5

    These are the weights and biases we need:

	 dataset    /model_weights/embedding_1/embedding_1/embeddings:0

	 dataset    /model_weights/simple_rnn_1/simple_rnn_1/bias:0
	 dataset    /model_weights/simple_rnn_1/simple_rnn_1/kernel:0
	 dataset    /model_weights/simple_rnn_1/simple_rnn_1/recurrent_kernel:0

	 dataset    /model_weights/dense_1/dense_1/bias:0
	 dataset    /model_weights/dense_1/dense_1/kernel:0

(2) View these weights
	We can view any of these weights:
	h5dump -d "/model_weights/embedding_1/embedding_1/embeddings:0" train-embeddings-rnn-50.h5

(3) Split them into individual files
	"splitH5.py" is our code to split H5 files, we write a bash file "split.sh" to run the python file multiple times. Simply run "./split.sh". We can neglect warnings.
	
	We can see these weights by typing, for example, "h5dump -d "/embedding_1/embeddings:0"  embedding_1_embeddings.h5". These commands are in the comments of "split.sh".

(4) Convert H5 to txt
	run "./h5_to_txt.sh", below is an example command

	h5dump -o dense_1_bias.txt -y -w 1000000000 dense_1_bias.h5

	-o F, --output=F     Output raw data into file F
	-y,   --noindex      Do not print array indices with the data
	-w N, --width=N      Set the number of columns of output. A value of 0 (zero)

(5) Remove commas
	Now, if you see these output txt files, you will find that the floats are splitted by commas, which is not friendly to our C program. For example, 

	-0.14107, -0.039722, 0.28277, 0.14393, 0.23464, -0.31021, 0.086173

	We will remove these commas by python.
	Run "./remove_comma.sh", which calls "remove_comma.py"

(6) Load weights into C
	We can load these txt files into C now, a template of it is in "C_laod.c"
