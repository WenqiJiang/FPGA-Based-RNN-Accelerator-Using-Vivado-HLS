#include "fc.h"
#include "types.h"
#include "constants.h"

#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])

#pragma SDS data zero_copy( \
    input_feature_map[0: BATCH_SIZE * RNN_STATE_SIZE])
#pragma SDS data zero_copy(output_feature_map[0: BATCH_SIZE * FC_OUTPUT_SIZE])

void wrapper_fc(FDATA_T input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE], 
                FDATA_T fc_bias[FC_OUTPUT_SIZE], 
                FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
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
            fc_kernel[current_kernel_index];
      }
      // add bias: bias[current_output_feature_map_index]
      output_feature_map[current_output_feature_map_index] +=
          fc_bias[output_feature_map_index];
    }
  }
}
