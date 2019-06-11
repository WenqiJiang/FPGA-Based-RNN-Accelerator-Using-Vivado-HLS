#include "types.h"
#include "constants.h"
#include "rnn.h"
#include "fc.h"

// #pragma SDS data copy(rnn_kernel[0: RNN_INPUT_SIZE * RNN_STATE_SIZE])
// #pragma SDS data copy(rnn_recurrent_kernel \
//                            [0: RNN_STATE_SIZE * RNN_STATE_SIZE])
// #pragma SDS data copy(rnn_bias[0: RNN_STATE_SIZE])

// #pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
// #pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])

// #pragma SDS data zero_copy( \
//     input_state[0: COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE])
// #pragma SDS data zero_copy(output[0:COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE])

// #pragma DATA ACCESS_PATTERN(rnn_kernel:SEQUENTIAL, \
//                             rnn_recurrent_kernel:SEQUENTIAL, \
//                             rnn_bias:SEQUENTIAL)

void copy_array(FDATA_T* src, FDATA_T* dst, LDATA_T len) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < len; i++)
    dst[i] = src[i];
}

void init_state(state[BATCH_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_STATE_SIZE; i++)
    state[i] = 0;
}
// finish 1 batch, e.g. 64, of computation, return the probability distribution
void wrapper_rnn_fc(
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_bias[RNN_STATE_SIZE], 
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T fc_bias[FC_OUTPUT_SIZE], 
    FDATA_T input_state[COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE*RNN_INPUT_SIZE], 
    FDATA_T output[COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // init last state inside the function, since the "first" last state is 0
  // use these two buffers alternately as last state and output state in RNN
  state0[BATCH_SIZE * RNN_STATE_SIZE];
  state1[BATCH_SIZE * RNN_STATE_SIZE];


  // store data to BRAM
  rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE];
  rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  rnn_bias_BRAM[RNN_STATE_SIZE];
  // fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
  // fc_bias_BRAM[FC_OUTPUT_SIZE];

  copy_array(rnn_kernel, rnn_kernel_BRAM, RNN_INPUT_SIZE * RNN_STATE_SIZE);
  copy_array(rnn_recurrent_kernel, rnn_recurrent_kernel_BRAM,
             RNN_STATE_SIZE * RNN_STATE_SIZE);
  // copy_array(rnn_bias, rnn_bias_BRAM, RNN_STATE_SIZE);
  // copy_array(fc_kernel, fc_kernel_BRAM, FC_OUTPUT_SIZE * FC_INPUT_SIZE);
  // copy_array(fc_bias, fc_bias_BRAM, FC_OUTPUT_SIZE);

  // 1,000 samples in total, 64 for 1 batch, so 15 iterations
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME; compute_time++) {

    // initialize all states to 0
    init_state(state0);
    init_state(state1);

    // go through 50 rnn layers
    for (LDATA_T i = 0; i < 25; i++) {

      // input state start address
      LDATA_T addr_offset1 = 2 * i * BATCH_SIZE * RNN_INPUT_SIZE +
          compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE;
      LDATA_T addr_offset2 = (2 * i + 1) * BATCH_SIZE * RNN_INPUT_SIZE +
          compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE;
      rnn(/* last state = */state0, 
          /* input_state = */input_state + addr_offset1,
          rnn_bias_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM, 
          /* output_state = */state1);
      rnn(/* last state = */state1, 
          /* input_state = */input_state + addr_offset2, 
          rnn_bias_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM, 
          /* output_state = */state0);
    }

    LDATA_T addr_offset_fc = compute_time * BATCH_SIZE * FC_OUTPUT_SIZE;
    // the last output state is state0, feed LDATA_To fc layer
    fc(/* input_feature_map = */state0, fc_bias, fc_kernel, 
       /* output_feature_map = */output);
  }
}

// advanced architecture 3
// for fc layer, only compute a tile at a time, 
// use load, compute, store structure and cover the DRAM access time