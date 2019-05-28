#pragma once

template <typename DT>
void rnn(DT* last_state, DT* input_state, DT* bias, DT* kernel, 
		DT* recurrent_kernel, DT* output_state);

template <typename DT>
void wrapper_rnn(DT* last_state, DT* input_state, DT* bias, 
    DT* kernel, DT* recurrent_kernel, DT* output_state)