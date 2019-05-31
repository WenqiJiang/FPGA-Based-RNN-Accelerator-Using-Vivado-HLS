// This file defines functions for array initializations

#include "init.h"
#include "types.h"

template<>
void linear_init(FDATA_T* input, FDATA_T lower_bound, FDATA_T upper_bound, 
                 LDATA_T length) {
  // linear initialization to an array

  FDATA_T dif = upper_bound - lower_bound;
  LDATA_T pieces = length - 1;
  FDATA_T val_assigned = lower_bound;

  for(LDATA_T idx = 0; idx < length; idx++)
  {
    input[idx] = val_assigned;
    val_assigned += dif / pieces;
  }
}

template<>
void zero_init(FDATA_T* input, LDATA_T length)
{
  for(LDATA_T idx = 0; idx < length; idx++)
    input[idx] = 0;
}

