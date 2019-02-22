#pragma once

void rnn(float last_state[STATE_LEN], 
		 float input[INPUT_LEN], 
		 float weights[WEIGHT_DIM1][WEIGHT_DIM2], 
		 float output[STATE_LEN]);