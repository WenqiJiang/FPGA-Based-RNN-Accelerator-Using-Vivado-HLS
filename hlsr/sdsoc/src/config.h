// This file defines where to load data, i.e. weights of the RNN, 
// input sequences, and output labels 

#pragma once

#ifdef __SDSCC__
char const * EMBEDDINGS_FILE        = "embedding_1_embeddings.txt";
char const * SIMPLE_RNN_BIAS_FILE   = "simple_rnn_1_bias.txt";
char const * SIMPLE_RNN_KERNEL_FILE = "simple_rnn_1_kernel.txt";
char const * SIMPLE_RNN_RECURRENT_KERNEL_FILE = 
    "simple_rnn_1_recurrent_kernel.txt";
char const * DENSE_BIAS_FILE        = "dense_1_bias.txt";
char const * DENSE_KERNEL_FILE      = "dense_1_kernel.txt";

char const * ORG_SEQ_FILE           = "org_seq.txt";
char const * RNN_RESULT_FILE        = "rnn_result.txt";
char const * ACTUAL_RESULT_FILE     = "actual_result.txt";

#else
char const * EMBEDDINGS_FILE        = "../../model/embedding_1_embeddings.txt";
char const * SIMPLE_RNN_BIAS_FILE   = "../../model/simple_rnn_1_bias.txt";
char const * SIMPLE_RNN_KERNEL_FILE = "../../model/simple_rnn_1_kernel.txt";
char const * SIMPLE_RNN_RECURRENT_KERNEL_FILE = 
    "../../model/simple_rnn_1_recurrent_kernel.txt";
char const * DENSE_BIAS_FILE        = "../../model/dense_1_bias.txt";
char const * DENSE_KERNEL_FILE      = "../../model/dense_1_kernel.txt";

char const * ORG_SEQ_FILE           = "../../datasets/org_seq.txt";
char const * RNN_RESULT_FILE        = "../../datasets/rnn_result.txt";
char const * ACTUAL_RESULT_FILE     = "../../datasets/actual_result.txt";
#endif

