#pragma once
#include "types.h"
#include "constants.h"

#define FC_TILE_BATCH 4

void wrapper_fc(FDATA_T input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE], 
                FDATA_T fc_bias[FC_OUTPUT_SIZE], 
                FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
                FDATA_T output_feature_map[FC_BATCH_SIZE * FC_OUTPUT_SIZE]);

void fc_load_input_feature_map(
    FDATA_T **input_feature_map_reg,
    FDATA_T input_feature_map_BRAM[BATCH_SIZE * FC_INPUT_SIZE]);

void fc_load_kernel(FDATA_T kernel_DRAM_part[FC_INPUT_SIZE], 
                    FDATA_T kernel_reg[FC_INPUT_SIZE]);

void fc_compute(FDATA_T **input_feature_map_reg,
                FDATA_T kernel_reg[FC_INPUT_SIZE],
                FDATA_T output_feature_map_reg[FC_TILE_BATCH]);

void fc_load_bias(FDATA_T bias[FC_OUTPUT_SIZE], 
                  FDATA_T bias_reg[FC_OUTPUT_SIZE]);

void fc_save_output_feature_map(
    FDATA_T output_feature_map_reg[BATCH_SIZE], FDATA_T bias_reg_single,
    FDATA_T output_feature_map_part[BATCH_SIZE]);
