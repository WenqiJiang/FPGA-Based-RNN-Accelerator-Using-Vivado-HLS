#include "constants.h"


void rnn(float last_state[STATE_LEN], 
		 float input[INPUT_LEN], 
		 float weights[WEIGHT_DIM1][WEIGHT_DIM2], 
		 float output[STATE_LEN])
{
	for(int out_idx = 0; out_idx < WEIGHT_DIM1; out_idx++) 
	/* placeholder: loop naming */
	{
		/* do multiplication: weights by last state */
		for(int last_state_idx = 0; last_state_idx < STATE_LEN; 
			last_state_idx++) 
		/* placeholder: loop naming */
		{
			output[out_idx] += last_state[last_state_idx] *
								weights[out_idx][last_state_idx];
		}

		/* do multiplication: weights by input */
		for(int input_idx = 0; input_idx < INPUT_LEN;
			input_idx++)
		{
		/* placeholder: loop naming */
			output[out_idx] += input[input_idx] * 
					weights[out_idx][input_idx + STATE_LEN];

		} 

	}
}