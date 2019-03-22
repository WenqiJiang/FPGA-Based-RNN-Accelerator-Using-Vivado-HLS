#include "constants.h"
#include <math.h>       /* import exponential function: exp (val) */
#include <stdlib.h>
// #include <stdio.h>

void softmax (float* input_feature_map,
              float* output_probability_distribution) {
    /* please do INITIALIZATION before input output_feature_map */
    /* ------- DIMENSION SETTING  ---------- *

     * input_feature_map:		            (SM_BATCH_SIZE,     SM_INPUT_SIZE)
     * output_probability_distribution:     (SM_BATCH_SIZE,     SM_OUTPUT_SIZE) =
     *                                      (SM_BATCH_SIZE,     SM_INPUT_SIZE) */

    /* used to cache the exponential result */
    double* input_feature_map_exp = malloc(sizeof(double) * SM_CLASS_SIZE);

    for (int batch_index = 0; batch_index < SM_BATCH_SIZE; batch_index++) {
        /* compute each sample in a batch */

        /* compute denominator, which is the sum of exponential
         * of each input_feature_map */
        double denominator = 0;

        for (int input_feature_map_index = 0; 
            input_feature_map_index < SM_CLASS_SIZE;
            input_feature_map_index++) {
            
            /* denominator += input_feature_map[batch_index][input_feature_map_index] */
            int current_input_feature_map_index = batch_index * SM_CLASS_SIZE + 
                                            input_feature_map_index;
            /* compute it, cache it */
            input_feature_map_exp[input_feature_map_index] = 
                            exp((double) input_feature_map[current_input_feature_map_index]);
            
            /* partial sum */
            denominator += input_feature_map_exp[input_feature_map_index];
        }

        /* now compute each output_probability_distribution */
        for (int output_probability_distribution_index = 0; 
             output_probability_distribution_index < SM_CLASS_SIZE;
             output_probability_distribution_index++) {
        
            /* output_probability_distribution [batch_index][output_probability_distribution_index]
               = input_feature_map_exp [output_probability_distribution_index] / denominator */
            int current_output_probability_distribution_index = batch_index * SM_CLASS_SIZE
                                                    + output_probability_distribution_index;
            output_probability_distribution [current_output_probability_distribution_index] =
                                input_feature_map_exp [output_probability_distribution_index] / denominator;
        }
    }
    free(input_feature_map_exp);
}

int argmax(float* input, int* result) {
    /*
    input: a probability distribution (SM_BATCH_SIZE, SM_OUTPUT_SIZE)
    result: the index of each output (SM_OUTPUT_SIZE, )
    */
    for (int batch_index = 0; batch_index < SM_BATCH_SIZE; batch_index++) {
        int max_index = 0;
        float max_val = input[batch_index * SM_CLASS_SIZE];
        for (int find_max_index = 0; find_max_index < SM_CLASS_SIZE; find_max_index++) {
            /* input[batch_index][find_max_index] */
            int input_index = batch_index * SM_CLASS_SIZE + find_max_index;
            if (input[input_index] > max_val) {
                max_index = find_max_index;
                max_val = input[input_index];
            }
        }
        // printf("%d %f\n", max_index, max_val);
        result[batch_index] = max_index;
    }
}