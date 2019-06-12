#pragma once
#include "types.h"
#include "constants.h"

void rnn(FDATA_T last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE], 
         FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE], 
         FDATA_T bias[RNN_STATE_SIZE], 
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE], 
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
         FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]);

