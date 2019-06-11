// This file defines the constants, i.e. the size of the RNN

#pragma once

#define TRUE 1
#define FALSE 0

// The arguments below are NOT independent: they are defined dependently for
// computational efficiency if you need to change them, change them TOGETHER!

// Embedding layer : 16192 * 100
#define WORD_NUM            16192
#define WORD_SIZE           100

// ============================== RNN layers ===================================
#define RNN_BATCH_SIZE      64
#define RNN_STATE_SIZE      128 // state length
#define RNN_INPUT_SIZE      100 // input of RNN layer, e.g. word vector

// RNN layer dimension:
//        input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
//        last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
//        bias: RNN_STATE_SIZE (128)
//        kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
//        recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
//        output_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None, 128)
//
//   computation:
//        for each sample in batch:
//        output_state = input_state mul kernel +
//                       last_state mul recurrent_kernel +
//                       bias

// ========================== Fully-Connected layers ===========================
#define FC_BATCH_SIZE   64
#define FC_INPUT_SIZE   128   // same as STATE_SIZE
#define FC_OUTPUT_SIZE  16192 // say, vocabulary number

// FC layer dimension:
//      input_feature_map: FC_BATCH_SIZE * FC_INPUT_SIZE (None * 128)
//      bias: FC_OUTPUT_SIZE (16192)
//      kernel: FC_INPUT_SIZE * FC_OUTPUT_SIZE (128 * 16192)
//      output_feature_map: FC_BATCH_SIZE * FC_OUTPUT_SIZE (None * 16192)
//
// computation:
//      for each sample in batch:
//          output_feature_map = input_feature_map mul kernel +
//                               bias

// ============================ Softmax layers =================================
#define SM_BATCH_SIZE   64
#define SM_CLASS_SIZE   16192 // how many classes

// Softmax layer dimension:
//    input_feature_map: SM_BATCH_SIZE * SM_CLASS_SIZE
//    output_feature_map: SM_BATCH_SIZE * SM_CLASS_SIZE

// Dateset -> 1,000 samples, each of them has 50 words
#define SAMPLE_NUM 1000
#define SAMPLE_LEN 50

#define BATCH_SIZE RNN_BATCH_SIZE
#define COMPUTE_TIME SAMPLE_NUM / BATCH_SIZE

