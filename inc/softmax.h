#pragma once

void float_softmax (float* input_feature_map, float* output_probability_distribution);
void double_softmax (double* input_feature_map, double* output_probability_distribution);
int float_argmax(float* input, int* result);
int double_argmax(double* input, int* result);