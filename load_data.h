#pragma once

void load_int(char* fname, int length, int* array);
void load_float(char* fname, int length, float* array);
void copy_float(float* copy_from, float* copy_to, int length);
void copy_int(int* copy_from, int* copy_to, int length);
void print_float(float* input, int length);
void print_int(int* input, int length);