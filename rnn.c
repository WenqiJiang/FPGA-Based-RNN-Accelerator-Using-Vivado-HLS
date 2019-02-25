#include "constants.h"

void rnn(float* last_state, 
		 float* input_state, 
		 float* weights, 
		 float* output_state) {
	/* please do INITIALIZATION before input output_state */
	/* ------- DIMENSION SETTING  ---------- *

	 * last_state:  	(BATCH_SIZE, STATE_SIZE) 
	 * input_state:		(BATCH_SIZE, INPUT_SIZE)
	 * weights:			(WEIGHT_DIM1, WEIGHT_DIM2) =
	 					(STATE_SIZE, STATE_SIZE + INPUT_SIZE)
	 * output_state: 	(BATCH_SIZE, STATE_SIZE) */

	for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
		/* placeholder: loop naming */
		/* compute each sample in a batch */

		for (int output_state_index = 0; output_state_index < WEIGHT_DIM1; output_state_index++) { 
			/* placeholder: loop naming */
			/* compute output_state[batch_index][output_state_index] */

			/* each output_state state has STATE_SIZE elements, compute each of them
			 * each computation is a vector vector multiplication
			 * vector 1: last_state concatenate input_state
			 * vector 2: a row of weights */

			/* output_state[batch_index][output_state_index] */
			int current_output_state_index = batch_index * STATE_SIZE + output_state_index;

			/* do multiplication: weights by last state */
			for (int last_state_index = 0; last_state_index < STATE_SIZE; 
				last_state_index++) {
				/* placeholder: loop naming */

				/* output_state[batch_index][output_state_index] += 
								last_state[batch_index][last_state_index] *
								weights[output_state_index][last_state_index] */

				/* last_state[batch_index][last_state_index] */
				int current_last_state_index = batch_index * STATE_SIZE + last_state_index;

				/* weights[output_state_index][last_state_index] */
				int current_weights_index = output_state_index * WEIGHT_DIM2 + last_state_index;

				/* do multiplication, add to previous value */
				output_state[current_output_state_index] += last_state[current_last_state_index] *
															weights[current_weights_index];
			}

			/* do multiplication: weights by input_state */
			for(int input_state_index = 0; input_state_index < INPUT_SIZE;
				input_state_index++) {
				/* placeholder: loop naming */

				/* output_state[batch_index][output_state_index] += 
								input_state[batch_index][input_state_index] * 
								weights[output_state_index][input_state_index + STATE_SIZE] */

				/* input_state[batch_index][input_state_index] */
				int current_input_state_index = batch_index * INPUT_SIZE + input_state_index;

				/* weights[output_state_index][input_state_index + STATE_SIZE] */
				int current_weights_index = output_state_index *WEIGHT_DIM2 + 
											input_state_index + STATE_SIZE;

				/* do multiplication, add to previous value */
				output_state[current_output_state_index] += input_state[current_input_state_index] *
															weights[current_weights_index];						
			}

		}

	}

}