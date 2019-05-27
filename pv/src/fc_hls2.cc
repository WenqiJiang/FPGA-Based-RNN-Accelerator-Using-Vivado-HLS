
#include <cstring> // memcpy
#include <stdio.h>

#include "fc.h"
#include "types.h"
#include "constants.h"
#include "utils.h"

#define TILE_BATCH 32

void load_kernel_and_compute(
    FDATA_T* kernel,
    FDATA_T* bias, 
    FDATA_T** input_feature_map_reg,
    FDATA_T output_feature_map[FC_BATCH_SIZE][FC_OUTPUT_SIZE],
    LDATA_T start_batch);


void compute(FDATA_T** input_feature_map_reg,
             FDATA_T* kernel_reg,
             FDATA_T* bias,
             FDATA_T* output_feature_map,
             LDATA_T output_feature_map_index, LDATA_T start_batch);

void load_input_feature_map(
    FDATA_T** input_feature_map_reg,
    FDATA_T* input_feature_map_BRAM,
    LDATA_T start_batch);

void load_kernel(FDATA_T* kernel_BRAM, 
                 FDATA_T* kernel_reg,
                 LDATA_T output_feature_map_index);

void load_kernel_and_compute(
    FDATA_T* kernel,
    FDATA_T* bias, 
    FDATA_T** input_feature_map_reg,
    FDATA_T* output_feature_map,
    LDATA_T start_batch) {
    // combine load_kernel and compute
    // finish compute one batch sample, i.e. a probability distribution

    // declare registers and use array partition
    FDATA_T kernel_reg[FC_INPUT_SIZE];
    // #pragma HLS ARRAY_PARTITION variable=kernel_reg dim=1


    for (LDATA_T output_feature_map_index = 0;
         output_feature_map_index < FC_OUTPUT_SIZE;
         output_feature_map_index++) {
       // #pragma HLS DATAFLOW

        // load
        load_kernel(kernel, kernel_reg, output_feature_map_index);

        // compute a batch of output
        compute(input_feature_map_reg, kernel_reg, bias,
            output_feature_map, output_feature_map_index, start_batch);

    }
}

void compute(FDATA_T** input_feature_map_reg,
             FDATA_T* kernel_reg,
             FDATA_T* bias,
             FDATA_T* output_feature_map,
             LDATA_T output_feature_map_index, LDATA_T start_batch) {

    // initialization
    //////////// different from HLS, using MALLOC ////////////
    // FDATA_T local_reg[TILE_BATCH][FC_INPUT_SIZE];
    FDATA_T **local_reg = malloc_2d_array(TILE_BATCH, FC_INPUT_SIZE);

    // compute
    for (LDATA_T batch_iter = 0; batch_iter < TILE_BATCH; batch_iter++) {
        // #pragma HLS UNROLL complete

        for (LDATA_T i = 0; i < FC_INPUT_SIZE; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = kernel_reg[i] * 
                input_feature_map_reg[batch_iter][i];
        }

        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 2; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 2];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 4; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 4];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 8; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                local_reg[batch_iter][i + FC_INPUT_SIZE / 8];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 16; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 16];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 32; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 32];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 64; i++) {
            // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 64];
        }
        for (LDATA_T i = 0; i < FC_INPUT_SIZE / 128; i++) {
            // // #pragma HLS UNROLL complete
            // MAC: output_FM_reg[i][output_feature_map_index] +=
            //          input_FM[i][j] * kernel[?][j]
            local_reg[batch_iter][i] = local_reg[batch_iter][i] + 
                local_reg[batch_iter][i + FC_INPUT_SIZE / 128];
        }
        output_feature_map[(start_batch + batch_iter) * FC_OUTPUT_SIZE + 
            output_feature_map_index] = 
            local_reg[batch_iter][0] + bias[output_feature_map_index];
    }

    //////////// different from HLS, using MALLOC ////////////
    free_2d_array(local_reg, TILE_BATCH, FC_INPUT_SIZE);

}


// to take advantage of constant loop bound, write load input FM and load kernel
// separately
void load_input_feature_map(
    FDATA_T** input_feature_map_reg,
    FDATA_T* input_feature_map_BRAM,
    LDATA_T start_batch) {
    // input_feature_map_reg: a register with a size of
    // FC_INPUT_SIZE * FC_BATCH_SIZE

    // load TILE_BATCH inputs at a time
    for (LDATA_T batch_iter = 0; batch_iter < TILE_BATCH; batch_iter++) {

        LDATA_T start_idx = (start_batch + batch_iter) * FC_INPUT_SIZE;

        for (LDATA_T input_feature_map_index = 0;
            input_feature_map_index < FC_INPUT_SIZE; input_feature_map_index++)
        {
            // #pragma HLS UNROLL factor=2
            // #pragma HLS PIPELINE
            // load to register
            input_feature_map_reg[batch_iter][input_feature_map_index] =
                input_feature_map_BRAM[input_feature_map_index + start_idx];
        }

    }
}

// to take advantage of constant loop bound, write load input FM and load kernel
// separately
void load_kernel(FDATA_T* kernel_BRAM, 
                 FDATA_T* kernel_reg,
                 LDATA_T output_feature_map_index) {
    // kernel_BRAM: FC_INPUT_SIZE x FC_OUTPUT_SIZE
    // kernel_reg: FC_INPUT_SIZE
    // output_feature_map_index: which column to read LDATA_To reg

    for (LDATA_T input_feature_map_index = 0;
         input_feature_map_index < FC_INPUT_SIZE;
         input_feature_map_index++) {
        // #pragma HLS UNROLL factor=2
        // #pragma HLS PIPELINE

        // kernel[input_feature_map_index][output_feature_map_index]
        LDATA_T current_kernel_index = output_feature_map_index +
            input_feature_map_index * FC_OUTPUT_SIZE;

        kernel_reg[input_feature_map_index] = kernel_BRAM[current_kernel_index];
    }
}

template<>
void fc(FDATA_T* input_feature_map, 
        FDATA_T* bias, 
        FDATA_T* kernel,
        FDATA_T* output_feature_map) {
    // please do INITIALIZATION before input output_feature_map
    // ------- DIMENSION SETTING  ----------

    //  input_feature_map: FC_BATCH_SIZE * FC_INPUT_SIZE (None * 128)
    //  bias: FC_OUTPUT_SIZE (16192)
    //  kernel: FC_INPUT_SIZE * FC_OUTPUT_SIZE (128 * 16192)
    //  output_feature_map: FC_BATCH_SIZE * FC_OUTPUT_SIZE (None * 16192)

    // STRATEGY -> tile, dataflow, pipeline, unroll, array_partition, prefix sum
    // total registers 548k
    // kernel -> 16,192 ~= 16k  16 x 16 = 256k
    // input_FM -> 128 128 x 64 x 4 = 32k
    // reuse time: 64x (= batch size) -> actually don't need to tile kernel

    // declare registers and use array partition

    //////////// different from HLS, using MALLOC ////////////
    // FDATA_T ** input_feature_map_reg;
    FDATA_T** input_feature_map_reg = malloc_2d_array(TILE_BATCH, FC_INPUT_SIZE);

    // #pragma HLS ARRAY_PARTITION variable=input_feature_map_reg dim=2
    // #pragma HLS ARRAY_PARTITION variable=output_feature_map_reg dim=1

    // reshape kernel so that it can be read in parallel [128 * 16192]
    // // #pragma HLS ARRAY_RESHAPE variable=kernel cyclic factor=32 dim=1

BATCH:
    for (LDATA_T batch_iter = 0; batch_iter < FC_BATCH_SIZE / TILE_BATCH;
         batch_iter++) {
        // #pragma HLS DATAFLOW
        // load
        load_input_feature_map(input_feature_map_reg, input_feature_map,
            batch_iter * TILE_BATCH);

        // compute + load
        load_kernel_and_compute(kernel, bias, input_feature_map_reg,
            output_feature_map, batch_iter * TILE_BATCH);
    }

    //////////// different from HLS, using MALLOC ////////////
    free_2d_array(input_feature_map_reg, TILE_BATCH, FC_INPUT_SIZE);
}
