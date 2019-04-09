#pragma once

void load_int(char* fname, int length, int* array);
void load_float(char* fname, int length, float* array);
void load_double(char* fname, int length, double* array);
void copy_int(int* copy_from, int* copy_to, int length);
void copy_float(float* copy_from, float* copy_to, int length);
void copy_double(double* copy_from, double* copy_to, int length);
void print_int(int* input, int length);
void print_float(float* input, int length);
void print_double(double* input, int length);