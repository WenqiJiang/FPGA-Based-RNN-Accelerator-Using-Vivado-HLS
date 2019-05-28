#pragma once

#include "types.h"
// void wrapper_fc(FDATA_T* input_feature_map, FDATA_T* bias, FDATA_T* kernel, FDATA_T* output_feature_map);

template <typename DT>
void fc(DT* input_feature_map, DT* bias, DT* kernel, DT* output_feature_map);

