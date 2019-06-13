#include "types.h"
#include "constants.h"
#include "rnn.h"
#include "fc.h"

#pragma SDS data copy(rnn_kernel[0: RNN_STATE_SIZE * RNN_INPUT_SIZE])
#pragma SDS data copy(rnn_recurrent_kernel \
                           [0: RNN_STATE_SIZE * RNN_STATE_SIZE])
#pragma SDS data copy(rnn_bias[0: RNN_STATE_SIZE])

#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])

#pragma SDS data zero_copy( \
    input_state[0: COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE])
#pragma SDS data zero_copy(output[0:COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE])

#pragma SDS data access_pattern(rnn_kernel:SEQUENTIAL, \
                                rnn_recurrent_kernel:SEQUENTIAL, \
                                rnn_bias:SEQUENTIAL)


void fc(FDATA_T input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE], 
        FDATA_T bias[FC_OUTPUT_SIZE], 
        FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
        FDATA_T output_feature_map[FC_BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // please do INITIALIZATION before input output_feature_map
  // ------- DIMENSION SETTING  ----------

  //  input_feature_map: FC_BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (16192)
  //  kernel: tranposed -> FC_OUTPUT_SIZE * FC_INPUT_SIZE  (16192 * 128)
  //  output_feature_map: FC_BATCH_SIZE * FC_OUTPUT_SIZE (None * 16192)

  for (LDATA_T batch_index = 0; batch_index < FC_BATCH_SIZE; batch_index++) {
    // compute each sample in a batch

    for (LDATA_T output_feature_map_index = 0;
         output_feature_map_index < FC_OUTPUT_SIZE;
         output_feature_map_index++) {

      // compute output_feature_map[batch_index][output_feature_map_index]
      // each output_feature_map has FC_OUTPUT_SIZE elements, compute each of them
      //  * each computation is a vector vector multiplication
      //  * vector 1: input_feature_map
      //  * vector 2: a row of weights

      // output_feature_map[batch_index][output_feature_map_index]
      LDATA_T current_output_feature_map_index = batch_index * FC_OUTPUT_SIZE +
          output_feature_map_index;

      // initialize to 0
      output_feature_map[current_output_feature_map_index] = 0;

      for (LDATA_T input_feature_map_index = 0;
          input_feature_map_index < FC_INPUT_SIZE;
          input_feature_map_index++) {

        // output_feature_map[batch_index][output_feature_map_index] +=
        //      input_feature_map[batch_index][input_feature_map_index] *
        //      kernel[output_feature_map_index][input_feature_map_index]

        // input_feature_map[batch_index][input_feature_map_index]
        LDATA_T current_input_feature_map_index = 
            batch_index * FC_INPUT_SIZE + input_feature_map_index;

        // kernel[output_feature_map_index][input_feature_map_index]
        LDATA_T current_kernel_index = 
            output_feature_map_index * FC_INPUT_SIZE + input_feature_map_index;

        // do multiplication, add to previous value
        output_feature_map[current_output_feature_map_index] +=
            input_feature_map[current_input_feature_map_index] *
            kernel[current_kernel_index];
      }
      // add bias: bias[current_output_feature_map_index]
      output_feature_map[current_output_feature_map_index] +=
          bias[output_feature_map_index];
    }
  }
}

void rnn(FDATA_T last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE], 
         FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE], 
         FDATA_T bias[RNN_STATE_SIZE], 
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE], 
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
         FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]) {
  // please do INITIALIZATION before input output_state
  // ------- DIMENSION SETTING  ---------- *
  //
  //   input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
  //   last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
  //   bias: RNN_STATE_SIZE (128)
  //   kernel: transposed -> RNN_STATE_SIZE * RNN_INPUT_SIZE (128 * 100)
  //   recurrent_kernel: transposed -> RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
  //   output_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None, 128)

  //  computation:
  //
  //    for each sample in batch:
  //    output_state = input_state mul kernel +
  //                   last_state mul recurrent_kernel +
  //                   bias

  for (LDATA_T batch_index = 0; batch_index < RNN_BATCH_SIZE; batch_index++) {
    // placeholder: loop naming
    // compute each sample in a batch

    for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE; 
         output_state_index++) {
      // placeholder: loop naming
      // compute output_state[batch_index][output_state_index]

      // each output_state state has STATE_SIZE elements, compute each of them
      // * each computation is a vector vector multiplication
      // * vector 1: last_state concatenate input_state
      // * vector 2: a row of weights

      // output_state[batch_index][output_state_index]
      LDATA_T current_output_state_index = 
          batch_index * RNN_STATE_SIZE + output_state_index;

      // initialize to 0
      output_state[current_output_state_index] = 0;

      // do multiplication: weights by last state
      for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
           last_state_index++) {
        // placeholder: loop naming

        // output_state[batch_index][output_state_index] +=
        //                 last_state[batch_index][last_state_index] *
        //                recurrent_kernel[output_state_index][last_state_index]

        // last_state[batch_index][last_state_index]
        LDATA_T current_last_state_index = 
            batch_index * RNN_STATE_SIZE + last_state_index;

        // recurrent_kernel[output_state_index][last_state_index]
        LDATA_T current_recurrent_kernel_index = 
            output_state_index * RNN_STATE_SIZE + last_state_index;

        // do multiplication, add to previous value
        // pr f("%f", last_state[current_last_state_index]);
        output_state[current_output_state_index] += 
            last_state[current_last_state_index] *
            recurrent_kernel[current_recurrent_kernel_index];
      }

      // do multiplication: weights by input_state
      for(LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
          input_state_index++) {
        // placeholder: loop naming

        // output_state[batch_index][output_state_index] +=
        //                input_state[batch_index][input_state_index] *
        //                kernel[output_state_index][input_state_index]

        // input_state[batch_index][input_state_index]
        LDATA_T current_input_state_index = 
            batch_index * RNN_INPUT_SIZE + input_state_index;

        // kernel[output_state_index][input_state_index]
        LDATA_T current_kernel_index = output_state_index * RNN_INPUT_SIZE +
            input_state_index;

        // do multiplication, add to previous value
        output_state[current_output_state_index] += 
            input_state[current_input_state_index] *
            kernel[current_kernel_index];
      }

      // add bias
      // bias[output_state_index]
      // HACKING!! should do this without conversion
      output_state[current_output_state_index] = FDATA_T(tanh(TOFLOAT(
          output_state[current_output_state_index]) + TOFLOAT(bias[output_state_index])));
      //output_state[current_output_state_index] += bias[output_state_index];
      //output_state[current_output_state_index] = tanh<FXD_W_LENGTH, FXD_I_LENGTH>(
      //     output_state[current_output_state_index]);
    }
  }
}

void copy_array(FDATA_T* src, FDATA_T* dst, LDATA_T len) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < len; i++)
    dst[i] = src[i];
}

void init_state(FDATA_T state[BATCH_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_STATE_SIZE; i++)
    state[i] = 0;
}
// finish 1 batch, e.g. 64, of computation, return the probability distribution
void wrapper_rnn_fc(
    FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE], 
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_bias[RNN_STATE_SIZE], 
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T fc_bias[FC_OUTPUT_SIZE], 
    FDATA_T input_state[COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE*RNN_INPUT_SIZE], 
    FDATA_T output[COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // init last state inside the function, since the "first" last state is 0
  // use these two buffers alternately as last state and output state in RNN
  FDATA_T state0[BATCH_SIZE * RNN_STATE_SIZE];
  FDATA_T state1[BATCH_SIZE * RNN_STATE_SIZE];


  // store data to BRAM
  FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];
  // fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
  // fc_bias_BRAM[FC_OUTPUT_SIZE];

  copy_array(rnn_kernel, rnn_kernel_BRAM, RNN_STATE_SIZE * RNN_INPUT_SIZE);
  copy_array(rnn_recurrent_kernel, rnn_recurrent_kernel_BRAM,
             RNN_STATE_SIZE * RNN_STATE_SIZE);
  copy_array(rnn_bias, rnn_bias_BRAM, RNN_STATE_SIZE);
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
      LDATA_T addr_offset1 = 
          compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE +
          2 * i * BATCH_SIZE * RNN_INPUT_SIZE;
      LDATA_T addr_offset2 = 
          compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE +
          (2 * i + 1) * BATCH_SIZE * RNN_INPUT_SIZE;
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
       /* output_feature_map = */output + addr_offset_fc);
  }
}

// advanced architecture 3
// for fc layer, only compute a tile at a time, 
// use load, compute, store structure and cover the DRAM access time
