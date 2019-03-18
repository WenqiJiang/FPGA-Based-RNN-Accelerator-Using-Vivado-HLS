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
	h5dump -o dense_1_bias.txt -y -w 1000000000 dense_1_bias.h5
