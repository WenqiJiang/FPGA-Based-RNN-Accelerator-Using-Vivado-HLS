#pragma once

#include "constants.h"
#include "types.h"

void wrapper_rnn_fc(
    FDATA_T rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
    FDATA_T rnn_bias[RNN_STATE_SIZE], 
    FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE], 
    FDATA_T fc_bias[FC_OUTPUT_SIZE], 
    FDATA_T input_state[COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE*RNN_INPUT_SIZE], 
    FDATA_T output[COMPUTE_TIME * BATCH_SIZE * FC_OUTPUT_SIZE]);
