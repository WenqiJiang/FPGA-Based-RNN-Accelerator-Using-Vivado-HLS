#include "rnn.h"
#include "types.h"
#include "constants.h"

#include <cstdio>

template<>
void rnn(FDATA_T* last_state, FDATA_T* input_state, FDATA_T* bias, FDATA_T* kernel, FDATA_T* recurrent_kernel, FDATA_T* output_state) {
    // please do INITIALIZATION before input output_state
    // ------- DIMENSION SETTING  ---------- *
    //
    //   input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
    //   last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
    //   bias: RNN_STATE_SIZE (128)
    //   kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
    //   recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
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

        for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE; output_state_index++) {
            // placeholder: loop naming
            // compute output_state[batch_index][output_state_index]

            // each output_state state has STATE_SIZE elements, compute each of them
            // * each computation is a vector vector multiplication
            // * vector 1: last_state concatenate input_state
            // * vector 2: a row of weights

            // output_state[batch_index][output_state_index]
            LDATA_T current_output_state_index = batch_index * RNN_STATE_SIZE + output_state_index;

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
                LDATA_T current_last_state_index = batch_index * RNN_STATE_SIZE + last_state_index;

                // recurrent_kernel[last_state_index][output_state_index]
                LDATA_T current_recurrent_kernel_index = last_state_index * RNN_STATE_SIZE + output_state_index;

                // do multiplication, add to previous value
                // pr f("%f", last_state[current_last_state_index]);
                output_state[current_output_state_index] += last_state[current_last_state_index] *
                    recurrent_kernel[current_recurrent_kernel_index];
            }

            // do multiplication: weights by input_state
            for(LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
                    input_state_index++) {
                // placeholder: loop naming

                // output_state[batch_index][output_state_index] +=
                //                input_state[batch_index][input_state_index] *
                //                kernel[output_state_index][input_state_index + STATE_SIZE]

                // input_state[batch_index][input_state_index]
                LDATA_T current_input_state_index = batch_index * RNN_INPUT_SIZE + input_state_index;

                // kernel[input_state_index][output_state_index]
                LDATA_T current_kernel_index = input_state_index * RNN_STATE_SIZE +
                    output_state_index;

                // do multiplication, add to previous value
                output_state[current_output_state_index] += input_state[current_input_state_index] *
                    kernel[current_kernel_index];
            }

            // add bias
            // bias[output_state_index]
            output_state[current_output_state_index] += bias[output_state_index];
        }

    }

}

#pragma SDS data zero_copy(last_state[0: (RNN_BATCH_SIZE * RNN_STATE_SIZE)])
#pragma SDS data zero_copy(input_state[0: (RNN_BATCH_SIZE * RNN_INPUT_SIZE)])
#pragma SDS data zero_copy(bias[0: RNN_STATE_SIZE])
#pragma SDS data zero_copy(kernel[0: (RNN_INPUT_SIZE * RNN_STATE_SIZE)])
#pragma SDS data zero_copy(recurrent_kernel[0: (RNN_STATE_SIZE * RNN_STATE_SIZE)])
#pragma SDS data zero_copy(output_state[0: (RNN_BATCH_SIZE * RNN_STATE_SIZE)])


void wrapper_rnn(FDATA_T* last_state, FDATA_T* input_state, FDATA_T* bias, 
    FDATA_T* kernel, FDATA_T* recurrent_kernel, FDATA_T* output_state) {

    rnn<FDATA_T> (last_state, input_state, bias, kernel, recurrent_kernel, output_state);
}


