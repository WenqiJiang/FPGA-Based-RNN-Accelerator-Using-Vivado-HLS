# first, we need to compute how large are our weights
# embedding layer: 16192 x 100 = 1,619,200
# recurrent layer: 128 * 128 / 128 * 100 / 128, all of them are very small
# dense layer: 128 * 16192 = 2,072,576
# in case the width can overflow, we will use the max int: 2 ^ 31 - 1 = 2,147,483,647
# or smaller is enough: 100,000,000 = 100000000

h5dump -o embedding_1_embeddings.txt -y -w 100000000 embedding_1_embeddings.h5

h5dump -o simple_rnn_1_bias.txt -y -w 100000000 simple_rnn_1_bias.h5
h5dump -o simple_rnn_1_kernel.txt -y -w 100000000 simple_rnn_1_kernel.h5
h5dump -o simple_rnn_1_recurrent_kernel.txt -y -w 100000000 simple_rnn_1_recurrent_kernel.h5

h5dump -o dense_1_bias.txt -y -w 100000000 dense_1_bias.h5
h5dump -o dense_1_kernel.txt -y -w 100000000 dense_1_kernel.h5