// This file defines some auxilliary functions such as loading data from txt 
// files, copying arrays and printing arrays.
#include "utils.h"

#include <stdio.h>
#include <cstdio>

#include "types.h"

template <>
void load_data(char const* fname, FDATA_T* array, LDATA_T length) {

  FILE *data_file;
  data_file = fopen(fname, "r");

  if (data_file == NULL) {
    printf("ERROR: cannot open file: %s\n", fname);
    exit(1);
  }

  // Read floating point values from file and convert them to FDATA_T.
  float *flt_array = (float*) malloc(length * sizeof(double));

  for (LDATA_T i = 0; i < length; i++) {
    LDATA_T r = fscanf(data_file, "%40f", &flt_array[i]);
    (void)r;  // suppress warning unused variable

    array[i] = FDATA_T(flt_array[i]);
  }

  free(flt_array);

  fclose(data_file);
}

template <>
void load_data(char const* fname, IDATA_T* array, LDATA_T length) {

  FILE *data_file;
  data_file = fopen(fname, "r");

  if (data_file == NULL) {
    printf("ERROR: cannot open file: %s\n", fname);
    exit(1);
  }

  for(LDATA_T i = 0; i < length; i++)
  {
    LDATA_T r = fscanf(data_file,"%d", &array[i]);
    (void) r; // suppress warning unused variable
  }

  fclose(data_file);
}

template <>
void copy_data(FDATA_T* copy_from, FDATA_T* copy_to, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i++) {
    copy_to[i] = copy_from[i];
  }
}

template <>
void copy_data(IDATA_T* copy_from, IDATA_T* copy_to, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i++) {
    copy_to[i] = copy_from[i];
  }
}

template <>
void print_data(FDATA_T* input, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i ++) {
    printf("%.10f\n", TOFLOAT(input[i]));
  }
}

template <>
void print_data(IDATA_T* input, LDATA_T length) {
  for (LDATA_T i = 0; i < length; i ++) {
    printf("%d\n", input[i]);
  }
}

template <>
void transpose(FDATA_T* src, FDATA_T* dst, IDATA_T ROW, IDATA_T COL)
{
  // transpose array
  // the source array has shape of (row, col)

  for (IDATA_T row = 0; row < ROW; row++) {
    for (IDATA_T col = 0; col < COL; col++) 
      dst[col * ROW + row] = src[row * COL + col];
  }
}
