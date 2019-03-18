# embedding layer
python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/embedding_1/embedding_1/embeddings:0" \
				--oname="./embedding_1_embeddings.h5" \
				--new_dataset="/embedding_1/embeddings:0"
				# h5dump -d "/embedding_1/embeddings:0" embedding_1_embeddings.h5

# RNN layer
python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/simple_rnn_1/simple_rnn_1/bias:0" \
				--oname="./simple_rnn_1_bias.h5" \
				--new_dataset="/simple_rnn_1/bias:0"
				# h5dump -d "/simple_rnn_1/bias:0" simple_rnn_1_bias.h5

python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/simple_rnn_1/simple_rnn_1/kernel:0" \
				--oname="./simple_rnn_1_kernel.h5" \
				--new_dataset="/simple_rnn_1/kernel:0"
				# h5dump -d "/simple_rnn_1/kernel:0" simple_rnn_1_kernel.h5

python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/simple_rnn_1/simple_rnn_1/recurrent_kernel:0" \
				--oname="./simple_rnn_1_recurrent_kernel.h5" \
				--new_dataset="/simple_rnn_1/recurrent_kernel:0"
				# h5dump -d "/simple_rnn_1/recurrent_kernel:0" simple_rnn_1_recurrent_kernel.h5

# dense layer

python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/dense_1/dense_1/bias:0" \
				--oname="./dense_1_bias.h5" \
				--new_dataset="/dense_1/bias:0"
				# h5dump -d "/dense_1/bias:0" dense_1_bias.h5

python splitH5.py --iname="./train-embeddings-rnn-50.h5" \
				--dataset="/model_weights/dense_1/dense_1/kernel:0" \
				--oname="./dense_1_kernel.h5" \
				--new_dataset="/dense_1/kernel:0"
				# h5dump -d "/dense_1/kernel:0" dense_1_kernel.h5