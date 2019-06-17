#include "wrapper.h"

#include "types.h"
#include "constants.h"
#include "rnn.h"
#include "fc.h"

// software control of 2 HW functions
// finish 15 batches of computations, 64 samples each batch
void wrapper_rnn_fc(
    FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE], 
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_bias[RNN_STATE_SIZE], 
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T fc_bias[FC_OUTPUT_SIZE], 
    FDATA_T input_state[COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE*RNN_INPUT_SIZE], 
    FDATA_T output[COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // malloc for rnn layer outputs
  FDATA_T* rnn_output_state = (FDATA_T*) 
      MALLOC(sizeof(FDATA_T) * BATCH_SIZE * RNN_STATE_SIZE);
  FDATA_T* output_transpose = (FDATA_T*)
      MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE * BATCH_SIZE);
  // 1,000 samples in total, 64 for 1 batch, so 15 iterations
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME; compute_time++) {

    // rnn wrapper, 50 timesteps, 64 samples
    LDATA_T input_state_offset = 
        compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE; 
    
    wrapper_rnn(rnn_bias, rnn_kernel, rnn_recurrent_kernel, 
                input_state + input_state_offset, rnn_output_state); 

    // fc wrapper, 64 samples
    LDATA_T output_state_offset = compute_time * BATCH_SIZE * FC_OUTPUT_SIZE;
    wrapper_fc(/* input_feature_map = */rnn_output_state, fc_bias, fc_kernel, 
               /* output_feature_map = */output_transpose);
    transpose(/* src = */output_transpose, 
              /* dst = */output + output_state_offset, 
              /* src_row = */FC_OUTPUT_SIZE, /* src_col = */BATCH_SIZE);
  }

  MFREE(rnn_output_state);
  MFREE(output_transpose);
}

// advanced architecture 3
// for fc layer, only compute a tile at a time, 
// use load, compute, store structure and cover the DRAM access time

// advanced architecture 4
// apply dataflow on control -> pipeline rnn wrapper and fc wrapper
