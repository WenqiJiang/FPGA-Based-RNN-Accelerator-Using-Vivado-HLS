#pragma once

void float_act_relu(float* input_feature_map, int length);
void double_act_relu(double* input_feature_map, int length);
/* C has tanh in stdlib */
void float_act_tanh(float* input_feature_map, int length);
void double_act_tanh(double* input_feature_map, int length);