
// This file declares functions for array initializations

#pragma once

template <typename DT, typename LT>
void linear_init(DT* input, DT lower_bound, DT upper_bound, LT length);

template <typename DT, typename LT>
void zero_init(DT* input, LT length);

