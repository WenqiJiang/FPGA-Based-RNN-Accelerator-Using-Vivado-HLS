#include "constants.h"
#include <stdio.h>

void float_rnn(float* last_state, 
         float* input_state, 
         float* bias, 
         float* kernel,
         float* recurrent_kernel,
         float* output_state) {
    /* please do INITIALIZATION before input output_state */
    /* ------- DIMENSION SETTING  ---------- *

        input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
        last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
        bias: RNN_STATE_SIZE (128)
        kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
        recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128) 
        output_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None, 128) */

    /*  computation:

        for each sample in batch:
        output_state = input_state mul kernel + 
                       last_state mul recurrent_kernel +
                       bias     */ 

    for (int batch_index = 0; batch_index < RNN_BATCH_SIZE; batch_index++) {
        /* placeholder: loop naming */
        /* compute each sample in a batch */

        for (int output_state_index = 0; output_state_index < RNN_STATE_SIZE; output_state_index++) { 
            /* placeholder: loop naming */
            /* compute output_state[batch_index][output_state_index] */

            /* each output_state state has STATE_SIZE elements, compute each of them
             * each computation is a vector vector multiplication
             * vector 1: last_state concatenate input_state
             * vector 2: a row of weights */

            /* output_state[batch_index][output_state_index] */
            int current_output_state_index = batch_index * RNN_STATE_SIZE + output_state_index;

            /* initialize to 0 */
            output_state[current_output_state_index] = 0;
            
            /* do multiplication: weights by last state */
            for (int last_state_index = 0; last_state_index < RNN_STATE_SIZE; 
                last_state_index++) {
                /* placeholder: loop naming */

                /* output_state[batch_index][output_state_index] += 
                                last_state[batch_index][last_state_index] *
                                recurrent_kernel[output_state_index][last_state_index] */

                /* last_state[batch_index][last_state_index] */
                int current_last_state_index = batch_index * RNN_STATE_SIZE + last_state_index;

                /* recurrent_kernel[last_state_index][output_state_index] */
                int current_recurrent_kernel_index = last_state_index * RNN_STATE_SIZE + output_state_index;

                /* do multiplication, add to previous value */
                // printf("%f", last_state[current_last_state_index]);
                output_state[current_output_state_index] += last_state[current_last_state_index] *
                                                            recurrent_kernel[current_recurrent_kernel_index];
            }

            /* do multiplication: weights by input_state */
            for(int input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
                input_state_index++) {
                /* placeholder: loop naming */

                /* output_state[batch_index][output_state_index] += 
                                input_state[batch_index][input_state_index] * 
                                kernel[output_state_index][input_state_index + STATE_SIZE] */

                /* input_state[batch_index][input_state_index] */
                int current_input_state_index = batch_index * RNN_INPUT_SIZE + input_state_index;

                /* kernel[input_state_index][output_state_index] */
                int current_kernel_index = input_state_index * RNN_STATE_SIZE + 
                                            output_state_index;

                /* do multiplication, add to previous value */
                output_state[current_output_state_index] += input_state[current_input_state_index] *
                                                            kernel[current_kernel_index];
            }

            /* add bias */
            /* bias[output_state_index] */
            output_state[current_output_state_index] += bias[output_state_index];
        }

    }

}

void double_rnn(double* last_state, 
         double* input_state, 
         double* bias, 
         double* kernel,
         double* recurrent_kernel,
         double* output_state) {
    /* please do INITIALIZATION before input output_state */
    /* ------- DIMENSION SETTING  ---------- *

        input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
        last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
        bias: RNN_STATE_SIZE (128)
        kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
        recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128) 
        output_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None, 128) */

    /*  computation:

        for each sample in batch:
        output_state = input_state mul kernel + 
                       last_state mul recurrent_kernel +
                       bias     */ 

    for (int batch_index = 0; batch_index < RNN_BATCH_SIZE; batch_index++) {
        /* placeholder: loop naming */
        /* compute each sample in a batch */

        for (int output_state_index = 0; output_state_index < RNN_STATE_SIZE; output_state_index++) { 
            /* placeholder: loop naming */
            /* compute output_state[batch_index][output_state_index] */

            /* each output_state state has STATE_SIZE elements, compute each of them
             * each computation is a vector vector multiplication
             * vector 1: last_state concatenate input_state
             * vector 2: a row of weights */

            /* output_state[batch_index][output_state_index] */
            int current_output_state_index = batch_index * RNN_STATE_SIZE + output_state_index;

            /* initialize to 0 */
            output_state[current_output_state_index] = 0;
            
            /* do multiplication: weights by last state */
            for (int last_state_index = 0; last_state_index < RNN_STATE_SIZE; 
                last_state_index++) {
                /* placeholder: loop naming */

                /* output_state[batch_index][output_state_index] += 
                                last_state[batch_index][last_state_index] *
                                recurrent_kernel[output_state_index][last_state_index] */

                /* last_state[batch_index][last_state_index] */
                int current_last_state_index = batch_index * RNN_STATE_SIZE + last_state_index;

                /* recurrent_kernel[last_state_index][output_state_index] */
                int current_recurrent_kernel_index = last_state_index * RNN_STATE_SIZE + output_state_index;

                /* do multiplication, add to previous value */
                // printf("%f", last_state[current_last_state_index]);
                output_state[current_output_state_index] += last_state[current_last_state_index] *
                                                            recurrent_kernel[current_recurrent_kernel_index];
            }

            /* do multiplication: weights by input_state */
            for(int input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
                input_state_index++) {
                /* placeholder: loop naming */

                /* output_state[batch_index][output_state_index] += 
                                input_state[batch_index][input_state_index] * 
                                kernel[output_state_index][input_state_index + STATE_SIZE] */

                /* input_state[batch_index][input_state_index] */
                int current_input_state_index = batch_index * RNN_INPUT_SIZE + input_state_index;

                /* kernel[input_state_index][output_state_index] */
                int current_kernel_index = input_state_index * RNN_STATE_SIZE + 
                                            output_state_index;

                /* do multiplication, add to previous value */
                output_state[current_output_state_index] += input_state[current_input_state_index] *
                                                            kernel[current_kernel_index];
            }

            /* add bias */
            /* bias[output_state_index] */
            output_state[current_output_state_index] += bias[current_output_state_index];
        }

    }

}