// This file declares the activation functions inlcude ReLU amd tanh

#pragma once

template <typename DT, typename DL>
void act_relu(DT* input_feature_map, DL length);

// C has tanh in stdlib
template <typename DT, typename DL>
void act_tanh(DT* input_feature_map, DL length);

