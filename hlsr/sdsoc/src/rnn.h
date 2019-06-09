#pragma once
#include "types.h"
#include "constants.h"

#define TILE_BATCH 64

void wrapper_rnn(FDATA_T last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
		         FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
				 FDATA_T bias[RNN_STATE_SIZE],
                 FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
				 FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
                 FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]);

template <typename DT>
void rnn(DT last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
         DT input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
         DT bias[RNN_STATE_SIZE],
         DT kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         DT recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         DT output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]);

void rnn_load_input_state(FDATA_T input_state_part[TILE_BATCH * RNN_INPUT_SIZE],
                          FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE]);

void rnn_load_last_state(FDATA_T last_state_part[TILE_BATCH * RNN_STATE_SIZE],
                         FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE]);

void rnn_load_kernel(FDATA_T kernel_part[RNN_INPUT_SIZE],
                     FDATA_T kernel_reg[RNN_INPUT_SIZE]);

void rnn_load_recurrent_kernel(FDATA_T recurrent_kernel_part[RNN_STATE_SIZE],
                               FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE]);

void rnn_compute(FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
                 FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
                 FDATA_T kernel_reg[RNN_INPUT_SIZE],
                 FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
                 FDATA_T output_state_reg_part[TILE_BATCH]);

void rnn_load_kernels_and_compute(
    FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
    FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
    FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T output_state_reg[RNN_STATE_SIZE][TILE_BATCH]);


void rnn_save_output_state(FDATA_T output_state_reg[RNN_STATE_SIZE][TILE_BATCH],
                           FDATA_T bias[RNN_STATE_SIZE],
                           FDATA_T output_state[TILE_BATCH * RNN_STATE_SIZE]);
