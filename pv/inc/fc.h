#pragma once
#include "types.h"
#include "constants.h"

void fc(FDATA_T input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE], 
        FDATA_T bias[FC_OUTPUT_SIZE], 
        FDATA_T kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
        FDATA_T output_feature_map[FC_BATCH_SIZE * FC_OUTPUT_SIZE]);

