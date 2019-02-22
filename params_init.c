#include "constants.h"

void linear_init(float input[], int length, float lower_bound, float upper_bound)
{
	float dif = upper_bound - lower_bound;
	int pieces = length - 1;
	float val_assigned;

	for(int idx = 0; idx < length; idx++)
	{
		val_assigned = lower_bound + dif * idx / pieces;
		input[idx] = val_assigned;
	}
}

void zero_init(float input[], int length)
{
	for(int idx = 0; idx < length; idx++)
		input[idx] = 0;
}