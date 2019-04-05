#include "constants.h"

void linear_init(float* input, int length, float lower_bound, float upper_bound)
{
	float dif = upper_bound - lower_bound;
	int pieces = length - 1;
	float val_assigned = lower_bound;

	for(int idx = 0; idx < length; idx++)
	{
		input[idx] = val_assigned;
		val_assigned += dif / pieces;
	}
}

void float_zero_init(float* input, int length)
{
	for(int idx = 0; idx < length; idx++)
		input[idx] = 0;
}


void double_zero_init(double* input, int length)
{
	for(int idx = 0; idx < length; idx++)
		input[idx] = 0;
}