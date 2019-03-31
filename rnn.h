#pragma once

void float_rnn(float* last_state, 
         float* input_state, 
         float* bias, 
         float* kernel,
         float* recurrent_kernel,
         float* output_state);


void double_rnn(double* last_state, 
         double* input_state, 
         double* bias, 
         double* kernel,
         double* recurrent_kernel,
         double* output_state);