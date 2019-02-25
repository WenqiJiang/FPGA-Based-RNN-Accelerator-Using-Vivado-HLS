#include "constants.h"

void fc(float* input_feature_map,
        float* weights,
        float* output_feature_map) {
    /* please do INITIALIZATION before input output_feature_map */
    /* ------- DIMENSION SETTING  ---------- *

     * input_feature_map:		(FC_BATCH_SIZE,     FC_INPUT_SIZE)
     * weights:			        (FC_WEIGHT_DIM1,    FC_WEIGHT_DIM2) =
                                 (FC_OUTPUT_SIZE,    FC_INPUT_SIZE)
     * output_feature_map: 	    (FC_BATCH_SIZE,     FC_OUTPUT_SIZE) */

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

            for (int input_feature_map_index = 0;
                input_feature_map_index < FC_INPUT_SIZE;
                input_feature_map_index++) {
                
                /* output_feature_map[batch_index][output_feature_map_index] += 
                 *      input_feature_map[batch_index][input_feature_map_index] *
                 *      weights[output_feature_map_index][input_feature_map_index] */

                /* input_feature_map[batch_index][input_feature_map_index] */
                int current_input_feature_map_index = batch_index * FC_INPUT_SIZE +
                                                        input_feature_map_index;

                /* weights[output_feature_map_index][input_feature_map_index] */
                int current_weights_index = output_feature_map_index * FC_WEIGHT_DIM2 +
                                            input_feature_map_index;

                /* do multiplication, add to previous value */
                output_feature_map[current_output_feature_map_index] += 
                            input_feature_map[current_input_feature_map_index] *
                            weights[current_weights_index];
            
            }
            
        }

    }

}