#pragma once

template <typename DT>
void softmax (DT* input_feature_map, DT* output_probability_distribution);

template <typename DT1, typename DT2>
void argmax(DT1* input, DT2* result);

