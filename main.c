#include <stdio.h>
#include "constants.h"
#include "rnn.h"
#include "params_init.h"
#include "fc.h"
//#include "testing.h"

int main(int argc, char *argv[])
{
	/* generate some data */
	float last_state[BATCH_SIZE * STATE_SIZE];
	float input[BATCH_SIZE * INPUT_SIZE];
	float weights[WEIGHT_DIM1 * WEIGHT_DIM2];
	float output[BATCH_SIZE * STATE_SIZE];
    
    printf("%s", "Press 1 and ENTER to start computing");
    char str1[10];
    scanf("%s", str1);
	/* don't need to initialize output, 'rnn' function will initialized it */
	linear_init(last_state, BATCH_SIZE * STATE_SIZE, 0, 1);
	linear_init(input, BATCH_SIZE * INPUT_SIZE, 0, 1);
	linear_init(weights, WEIGHT_DIM1 * WEIGHT_DIM2, 0, 1);
	zero_init(output, BATCH_SIZE * STATE_SIZE);
	/* do inference and print the result */

	rnn(last_state, input, weights, output);
	//	print_output(output)
	for (int i = 0; i < BATCH_SIZE; i++) {
		for (int j = 0; j < STATE_SIZE; j++) {
			printf("%f	", output[i * BATCH_SIZE + j]);
		}
		printf("/n");
	}

    printf("%s", "Press 2 and ENTER to end the program");
    char str2[10];
    scanf("%s", str2);

	return 0;
}