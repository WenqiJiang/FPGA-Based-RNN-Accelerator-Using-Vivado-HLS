#include "fc.h"
#include "types.h"
#include "constants.h"

template<>
void fc(FDATA_T* input_feature_map, FDATA_T* bias, FDATA_T* kernel, FDATA_T* output_feature_map) {

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
                LDATA_T current_input_feature_map_index = batch_index * FC_INPUT_SIZE +
                    input_feature_map_index;

                // kernel[output_feature_map_index][input_feature_map_index]
                LDATA_T current_kernel_index = output_feature_map_index * FC_INPUT_SIZE +
                    input_feature_map_index;

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
