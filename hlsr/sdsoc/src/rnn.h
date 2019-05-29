#pragma once
#include "types.h"

template <typename DT>
void rnn(DT* last_state, DT* input_state, DT* bias, DT* kernel, DT* recurrent_kernel, DT* output_state);

void wrapper_rnn(FDATA_T* last_state, FDATA_T* input_state, FDATA_T* bias, 
    FDATA_T* kernel, FDATA_T* recurrent_kernel, FDATA_T* output_state);