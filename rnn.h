#pragma once

void rnn(float* last_state, 
         float* input_state, 
         float* bias, 
         float* kernel,
         float* recurrent_kernel,
         float* output_state);