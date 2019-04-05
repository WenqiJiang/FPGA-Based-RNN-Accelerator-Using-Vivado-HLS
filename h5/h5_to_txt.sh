# embedding layer
python h5_to_txt.py --iname="./embedding_1_embeddings.h5" \
				--dataset="/embedding_1/embeddings:0" \
				--dim_num=2 --length1=16192 --length2=100 \
				> embedding_1_embeddings.txt
				# h5dump -d "/embedding_1/embeddings:0" embedding_1_embeddings.h5

# RNN layer
python h5_to_txt.py --iname="./simple_rnn_1_bias.h5" --dataset="/simple_rnn_1/bias:0" --dim_num=1 --length1=128 > simple_rnn_1_bias.txt
				# h5dump -d "/simple_rnn_1/bias:0" simple_rnn_1_bias.h5
python h5_to_txt.py --iname="./simple_rnn_1_kernel.h5" --dataset="/simple_rnn_1/kernel:0" --dim_num=2 --length1=100 --length2=128 > simple_rnn_1_kernel.txt
				# h5dump -d "/simple_rnn_1/kernel:0" simple_rnn_1_kernel.h5
python h5_to_txt.py --iname="./simple_rnn_1_recurrent_kernel.h5" --dataset="/simple_rnn_1/recurrent_kernel:0" --dim_num=2 --length1=128 --length2=128 > simple_rnn_1_recurrent_kernel.txt
				# h5dump -d "/simple_rnn_1/recurrent_kernel:0" simple_rnn_1_recurrent_kernel.h5

# dense layer

python h5_to_txt.py --iname="./dense_1_bias.h5" \
				--dataset="/dense_1/bias:0" \
				--dim_num=1 --length1=16192 > dense_1_bias.txt
				# h5dump -d "/dense_1/bias:0" dense_1_bias.h5

python h5_to_txt.py --iname="./dense_1_kernel.h5" \
				--dataset="/dense_1/kernel:0" \
				--dim_num=2 --length1=128 --length2=16192 \
					> dense_1_kernel.txt
				# h5dump -d "/dense_1/kernel:0" dense_1_kernel.h5
