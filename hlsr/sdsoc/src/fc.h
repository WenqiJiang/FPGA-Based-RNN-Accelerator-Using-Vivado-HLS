#pragma once

#include "types.h"
void wrapper_fc(FDATA_T* input_feature_map, FDATA_T* bias, FDATA_T* kernel,
                FDATA_T* output_feature_map);

template <typename DT>
void fc(DT input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE], 
        DT bias[FC_OUTPUT_SIZE], 
        DT kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        DT output_feature_map[FC_BATCH_SIZE * FC_OUTPUT_SIZE]);

#define TILE_BATCH 32

void load_input_feature_map(
    FDATA_T input_feature_map_reg[TILE_BATCH][FC_INPUT_SIZE],
    FDATA_T input_feature_map_BRAM[FC_BATCH_SIZE * FC_INPUT_SIZE],
    LDATA_T start_batch);

void load_kernel(FDATA_T kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
                FDATA_T kernel_reg[FC_INPUT_SIZE],
                LDATA_T output_feature_map_index);

void compute(FDATA_T input_feature_map_reg[TILE_BATCH][FC_INPUT_SIZE], 
             FDATA_T kernel_reg[FC_INPUT_SIZE],
             FDATA_T output_feature_map_reg[TILE_BATCH][FC_OUTPUT_SIZE], 
             LDATA_T output_feature_map_index);

void load_kernel_and_compute(
    FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T input_feature_map_reg[TILE_BATCH][FC_INPUT_SIZE],
    FDATA_T output_feature_map_reg[TILE_BATCH][FC_OUTPUT_SIZE]);

void save_output_feature_map(
    FDATA_T output_feature_map_reg[TILE_BATCH][FC_OUTPUT_SIZE],
    FDATA_T bias[FC_OUTPUT_SIZE],
    FDATA_T output_feature_map_BRAM[FC_BATCH_SIZE * FC_OUTPUT_SIZE],
    LDATA_T start_batch);
