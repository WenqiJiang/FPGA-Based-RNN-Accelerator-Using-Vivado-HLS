// This file declares some auxilliary functions such as loading data from txt 
// files, copying arrays and printing arrays.

#pragma once

#include "types.h"

template <typename DT, typename LT>
void load_data(char const* fname, DT* array, LT length);

template <typename DT, typename LT>
void copy_data(DT* copy_from, DT* copy_to, LT length);

template <typename DT, typename LT>
void print_data(DT* input, LT length);

