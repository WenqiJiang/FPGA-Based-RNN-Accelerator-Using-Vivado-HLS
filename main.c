#include <stdio.h>
#include "constants.h"
#include "rnn.h"
#include "params_init.h"
//#include "testing.h"

int main(int argc, char *argv[])
{
	/* generate some data */
	float last_state[STATE_LEN];
	float input[INPUT_LEN];
	float weights[WEIGHT_DIM1][WEIGHT_DIM2];
	float output[STATE_LEN];
    
    printf("%s", "Hi I'm here");
    char str1[10];
    scanf("%s", str1);
	/* don't need to initialize output, 'rnn' function will initialized it */
	linear_init(last_state, STATE_LEN, 0, 1);
	linear_init(input, INPUT_LEN, 0, 1);
	linear_init(weights, WEIGHT_DIM1 * WEIGHT_DIM2, 0, 1);
	zero_init(output, STATE_LEN);
	/* do inference and print the result */

	rnn(last_state, input, weights, output);
//	print_output(output)
    printf("%s", "Hi I'm here");
    char str2[10];
    scanf("%s", str2);
	return 0;
}