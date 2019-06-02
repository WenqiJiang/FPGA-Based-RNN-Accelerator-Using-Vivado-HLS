#pragma once

#include "types.h"

template <typename DT, typename LT>
void load_data(char const* fname, DT* array, LT length);

template <typename DT, typename LT>
void copy_data(DT* copy_from, DT* copy_to, LT length);

template <typename DT, typename LT>
void print_data(DT* input, LT length);

template <typename IT>
FDATA_T** malloc_2d_array(IT row, IT col);

template <typename DT, typename IT>
void free_2d_array(DT** arr, IT row, IT col);

template <typename DT, typename IT>
void transpose(DT* src, DT* dst, const IT row, const IT col);
