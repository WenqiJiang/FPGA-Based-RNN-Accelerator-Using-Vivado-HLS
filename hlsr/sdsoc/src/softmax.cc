// This file defines the softmax function and the argmax function

#include "types.h"
#include "softmax.h"
#include "constants.h"

// #include <cmath> // import exponential function: exp (val)
#include <cstdlib>

template<>
void softmax (FDATA_T* input_feature_map, 
              FDATA_T* output_probability_distribution) {
  // please do INITIALIZATION before input output_feature_map
  // ------- DIMENSION SETTING  ---------- *

  // input_feature_map:                (SM_BATCH_SIZE,     SM_INPUT_SIZE)
  // output_probability_distribution:     (SM_BATCH_SIZE,     SM_OUTPUT_SIZE) =
  //                                      (SM_BATCH_SIZE,     SM_INPUT_SIZE)

  // used to cache the exponential result
  double* input_feature_map_exp = 
      (double*) malloc(sizeof(double) * SM_CLASS_SIZE);

  // compute each sample in a batch
  for (LDATA_T batch_index = 0; batch_index < SM_BATCH_SIZE; batch_index++) {

    // compute denominator, which is the sum of exponential
    // of each input_feature_map
    double denominator = 0;

    for (LDATA_T input_feature_map_index = 0;
            input_feature_map_index < SM_CLASS_SIZE;
            input_feature_map_index++) {

      // denominator += input_feature_map[batch_index][input_feature_map_index]
      LDATA_T current_input_feature_map_index = batch_index * SM_CLASS_SIZE +
          input_feature_map_index;

      // compute it, cache it
      input_feature_map_exp[input_feature_map_index] =
          exp((double) input_feature_map[current_input_feature_map_index]);

      // partial sum
      denominator += input_feature_map_exp[input_feature_map_index];
    }

    // now compute each output_probability_distribution
    for (LDATA_T output_probability_distribution_index = 0;
            output_probability_distribution_index < SM_CLASS_SIZE;
            output_probability_distribution_index++) {

      // output_probability_distribution [batch_index][output_probability_distribution_index]
      //   = input_feature_map_exp [output_probability_distribution_index] / denominator
      LDATA_T current_output_probability_distribution_index = 
          batch_index * SM_CLASS_SIZE + output_probability_distribution_index;
      output_probability_distribution[current_output_probability_distribution_index] =
          input_feature_map_exp[output_probability_distribution_index] / denominator;
    }
  }
  free(input_feature_map_exp);
}

template<>
void argmax(FDATA_T* input, IDATA_T* result) {
    // input: a probability distribution (SM_BATCH_SIZE, SM_OUTPUT_SIZE)
    // result: the index of each output (SM_OUTPUT_SIZE, )
    for (LDATA_T batch_index = 0; batch_index < SM_BATCH_SIZE; batch_index++) {
        LDATA_T max_index = 0;
        FDATA_T max_val = input[batch_index * SM_CLASS_SIZE];
        for (LDATA_T find_max_index = 0; find_max_index < SM_CLASS_SIZE; 
             find_max_index++) {
            // input[batch_index][find_max_index]
            LDATA_T input_index = batch_index * SM_CLASS_SIZE + find_max_index;
            if (input[input_index] > max_val) {
                max_index = find_max_index;
                max_val = input[input_index];
            }
        }
        result[batch_index] = max_index;
    }
}

