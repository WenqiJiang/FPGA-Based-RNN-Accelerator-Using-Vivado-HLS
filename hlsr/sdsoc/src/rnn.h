#pragma once
#include "types.h"
#include "constants.h"

template <typename DT>
void rnn(DT last_state[RNN_BATCH_SIZE][RNN_STATE_SIZE],
         DT input_state[RNN_BATCH_SIZE][RNN_INPUT_SIZE],
         DT bias[RNN_STATE_SIZE],
         DT kernel[RNN_INPUT_SIZE][RNN_STATE_SIZE],
         DT recurrent_kernel[RNN_STATE_SIZE][RNN_STATE_SIZE],
         DT output_state[RNN_BATCH_SIZE][RNN_STATE_SIZE]);

void wrapper_rnn(FDATA_T last_state[RNN_BATCH_SIZE][RNN_STATE_SIZE],
                 FDATA_T input_state[RNN_BATCH_SIZE][RNN_INPUT_SIZE],
                 FDATA_T bias[RNN_STATE_SIZE],
                 FDATA_T kernel[RNN_INPUT_SIZE][RNN_STATE_SIZE],
                 FDATA_T recurrent_kernel[RNN_STATE_SIZE][RNN_STATE_SIZE],
                 FDATA_T output_state[RNN_BATCH_SIZE][RNN_STATE_SIZE]);