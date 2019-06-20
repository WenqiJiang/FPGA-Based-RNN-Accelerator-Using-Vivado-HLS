#include "fc.h"
#include "types.h"
#include "constants.h"

// to take advantage of constant loop bound, write load input FM and load kernel
// separately
void fc_load_input_feature_map(
    FDATA_T input_feature_map_reg[BATCH_SIZE][FC_INPUT_SIZE], 
    FDATA_T input_feature_map_BRAM[BATCH_SIZE * FC_INPUT_SIZE]) {

  // load BATCH_SIZE inputs at a time
  for (LDATA_T batch_iter = 0; batch_iter < BATCH_SIZE; batch_iter++) {

    LDATA_T start_idx = batch_iter * FC_INPUT_SIZE;
    for (LDATA_T input_feature_map_index = 0;
        input_feature_map_index < FC_INPUT_SIZE; input_feature_map_index++)
    {
#pragma HLS PIPELINE
      input_feature_map_reg[batch_iter][input_feature_map_index] =
          input_feature_map_BRAM[input_feature_map_index + start_idx];
    }
  }
}

// to take advantage of constant loop bound, write load input FM and load kernel
// separately
void fc_load_kernel(FDATA_T kernel_DRAM_part[FC_INPUT_SIZE], 
                    FDATA_T kernel_reg[FC_INPUT_SIZE]) {
  // kernel_DRAM: FC_OUTPUT_SIZE * FC_INPUT_SIZE
  // kernel_reg: FC_INPUT_SIZE
  // output_feature_map_index: which column to read LDATA_To reg

  for (LDATA_T input_feature_map_index = 0;
       input_feature_map_index < FC_INPUT_SIZE;
       input_feature_map_index++) {
#pragma HLS PIPELINE

    kernel_reg[input_feature_map_index] = 
        kernel_DRAM_part[input_feature_map_index];
  }
}

void fc_compute(FDATA_T input_feature_map_reg[BATCH_SIZE][FC_INPUT_SIZE],  
                FDATA_T kernel_reg[FC_INPUT_SIZE],
                FDATA_T output_feature_map_reg[BATCH_SIZE]) {

  // initialization
  FDATA_T local_reg[BATCH_SIZE][FC_INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_reg dim=2 cyclic factor=32
#pragma HLS ARRAY_PARTITION variable=local_reg dim=1 cyclic factor=2
  for (LDATA_T iter = 0; iter < BATCH_SIZE / BATCH_SIZE; iter++) {

    LDATA_T start_batch = iter * BATCH_SIZE;

    for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
#pragma HLS UNROLL complete
      // compute
      for (LDATA_T i = 0; i < FC_INPUT_SIZE; i++) {
#pragma HLS UNROLL complete
//#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = 
            kernel_reg[i] * input_feature_map_reg[start_batch + batch_idx][i];
      }

      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 2; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 2];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 4; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 4];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 8; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 8];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 16; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 16];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 32; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 32];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 64; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 64];
      }
      for (LDATA_T i = 0; i < FC_INPUT_SIZE / 128; i++) {
#pragma HLS UNROLL complete
        // MAC: output_FM_reg[i][output_feature_map_index] +=
        //          input_FM[i][j] * kernel[?][j]
        local_reg[batch_idx][i] = local_reg[batch_idx][i] + 
            local_reg[batch_idx][i + FC_INPUT_SIZE / 128];
      }
      output_feature_map_reg[start_batch + batch_idx] = local_reg[batch_idx][0];
    }
  }
}

void fc_load_bias(FDATA_T bias[FC_OUTPUT_SIZE], 
                  FDATA_T bias_reg[FC_OUTPUT_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE
    bias_reg[i] = bias[i];
  }
}

void fc_save_output_feature_map(
    FDATA_T output_feature_map_reg[BATCH_SIZE], FDATA_T bias_reg_single,
    FDATA_T output_feature_map_part[BATCH_SIZE]) {
  // save  outputs a time
  // output_feature_map_reg: BATCH_SIZE x FC_OUTPUT_SIZE
  // output_feature_map_DRAM -> transposed: FC_OUTPUT_SIZE x BATCH_SIZE
  // start_batch_index: which batch to save to BRAM

  for (LDATA_T i = 0; i < BATCH_SIZE; i++) {
#pragma HLS PIPELINE
    output_feature_map_part[i] =
        bias_reg_single + output_feature_map_reg[i];
  }
}

#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])

#pragma SDS data zero_copy( \
    input_feature_map[0: BATCH_SIZE * RNN_STATE_SIZE])
#pragma SDS data zero_copy(output_feature_map[0: BATCH_SIZE * FC_OUTPUT_SIZE])

void wrapper_fc(FDATA_T input_feature_map[BATCH_SIZE * FC_INPUT_SIZE],
                FDATA_T fc_bias[FC_OUTPUT_SIZE], 
                FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
                FDATA_T output_feature_map[BATCH_SIZE * FC_OUTPUT_SIZE]) {

  // please do INITIALIZATION before input output_feature_map
  // ------- DIMENSION SETTING  ----------

  //  input_feature_map: BATCH_SIZE * FC_INPUT_SIZE (None * 128)
  //  bias: FC_OUTPUT_SIZE (16192)
  //  kernel: tranposed -> FC_OUTPUT_SIZE * FC_INPUT_SIZE  (16192 * 128)
  //  output_feature_map -> ///transposed!!/// FC_OUTPUT_SIZE * BATCH_SIZE

  // declare registers and use array partition
  FDATA_T input_feature_map_reg[BATCH_SIZE][FC_INPUT_SIZE];
  FDATA_T output_feature_map_reg[BATCH_SIZE];
#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
    dim=2 cyclic factor=32
////////////////////       MENTION       /////////////////////
// unroll in dimension 1 should be equal to BATCH_SIZE 
#pragma HLS ARRAY_PARTITION variable=input_feature_map_reg \
    dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=output_feature_map_reg \
    dim=1 cyclic factor=32
  // output feature map will be transposed later in CPU + DRAM

  FDATA_T kernel_reg[FC_INPUT_SIZE];
//  FDATA_T bias_reg[FC_OUTPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1 cyclic factor=32
  // bias read one at a time, so don't need to unroll

  // load preliminary data
  fc_load_input_feature_map(input_feature_map_reg, input_feature_map);
//  fc_load_bias(fc_bias, bias_reg);

  // load, compute, save
EACH_OUT_FM:
  for (LDATA_T output_feature_map_index = 0;
       output_feature_map_index < FC_OUTPUT_SIZE;
       output_feature_map_index++) {
#pragma HLS DATAFLOW

    // load
    LDATA_T kernel_offset = output_feature_map_index * FC_INPUT_SIZE;
    fc_load_kernel(fc_kernel + kernel_offset, kernel_reg);

    // compute
    fc_compute(input_feature_map_reg, kernel_reg, output_feature_map_reg);

    // save
    LDATA_T output_feature_map_offset = output_feature_map_index * BATCH_SIZE;
    fc_save_output_feature_map(
        output_feature_map_reg, fc_bias[output_feature_map_index],
        output_feature_map + output_feature_map_offset);
  }
}
