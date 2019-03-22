#include "constants.h"

void fc(float* input_feature_map,
        float* bias,
        float* kernel,
        float* output_feature_map) {
    /* please do INITIALIZATION before input output_feature_map */
    /* ------- DIMENSION SETTING  ---------- *

     *  input_feature_map: FC_BATCH_SIZE * FC_INPUT_SIZE (None * 128)
     *  bias: FC_OUTPUT_SIZE (16192)
     *  kernel: FC_INPUT_SIZE * FC_OUTPUT_SIZE (128 * 16192) 
     *  output_feature_map: FC_BATCH_SIZE * FC_OUTPUT_SIZE (None * 16192) */

    for (int batch_index = 0; batch_index < FC_BATCH_SIZE; batch_index++) {
        /* compute each sample in a batch */

        for (int output_feature_map_index = 0; 
             output_feature_map_index < FC_OUTPUT_SIZE;
             output_feature_map_index++) {

            /* compute output_feature_map[batch_index][output_feature_map_index] */
            /* each output_feature_map has FC_OUTPUT_SIZE elements, compute each of them
                * each computation is a vector vector multiplication
                * vector 1: input_feature_map
                * vector 2: a row of weights */
            
            /* output_feature_map[batch_index][output_feature_map_index] */
            int current_output_feature_map_index = batch_index * FC_OUTPUT_SIZE +
                                                    output_feature_map_index;
            
            /* initialize to 0 */
            output_feature_map[current_output_feature_map_index] = 0;
            
            for (int input_feature_map_index = 0;
                input_feature_map_index < FC_INPUT_SIZE;
                input_feature_map_index++) {
                
                /* output_feature_map[batch_index][output_feature_map_index] += 
                 *      input_feature_map[batch_index][input_feature_map_index] *
                 *      kernel[output_feature_map_index][input_feature_map_index] */

                /* input_feature_map[batch_index][input_feature_map_index] */
                int current_input_feature_map_index = batch_index * FC_INPUT_SIZE +
                                                        input_feature_map_index;

                /* kernel[output_feature_map_index][input_feature_map_index] */
                int current_kernel_index = input_feature_map_index * FC_OUTPUT_SIZE +
                                            output_feature_map_index;

                /* do multiplication, add to previous value */
                output_feature_map[current_output_feature_map_index] += 
                            input_feature_map[current_input_feature_map_index] *
                            kernel[current_kernel_index];
            }
            /* add bias: bias[current_output_feature_map_index] */
            output_feature_map[current_output_feature_map_index] += 
                            bias[output_feature_map_index];       
        }

    }

}