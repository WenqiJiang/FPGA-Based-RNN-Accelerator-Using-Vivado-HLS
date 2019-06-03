#pragma once
#include "types.h"
#include "constants.h"


#define TILE_BATCH 16


void wrapper_rnn(FDATA_T* last_state, FDATA_T* input_state, FDATA_T* bias,
                 FDATA_T* kernel, FDATA_T* recurrent_kernel, 
                 FDATA_T* output_state);

template <typename DT>
void rnn(DT last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
         DT input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
         DT bias[RNN_STATE_SIZE],
         DT kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         DT recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         DT output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]);

void load_input_state(FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
                      FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
                      LDATA_T start_batch);

void load_last_state(FDATA_T last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
                     FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
                     LDATA_T start_batch);

void load_kernel(FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
                 FDATA_T kernel_reg[RNN_INPUT_SIZE],
                 LDATA_T output_state_index);

void load_recurrent_kernel(
    FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
    LDATA_T output_state_index);

void compute(FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
             FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
             FDATA_T kernel_reg[RNN_INPUT_SIZE],
             FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
             LDATA_T output_state_index,
             FDATA_T output_state_reg[TILE_BATCH][RNN_STATE_SIZE]);

void load_kernels_and_compute(
    FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
    FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
    FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T output_state_reg[TILE_BATCH][RNN_STATE_SIZE]);


void save_output_state(FDATA_T output_state_reg[TILE_BATCH][RNN_STATE_SIZE],
                       FDATA_T bias[RNN_STATE_SIZE],
                       FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
                       LDATA_T start_batch_index);