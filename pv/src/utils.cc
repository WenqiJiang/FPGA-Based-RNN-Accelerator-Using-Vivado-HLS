#include "utils.h"

#include <cstdio>
#include <stdlib.h>

template <>
void load_data(char const* fname, FDATA_T* array, LDATA_T length) {

    FILE *data_file;
    data_file = fopen(fname, "r");

    if (data_file == NULL) {
        printf("ERROR: cannot open file: %s\n", fname);
        exit(1);
    }

    for(LDATA_T i = 0; i < length; i++)
    {
        LDATA_T r = fscanf(data_file,"%40f", &array[i]);
        (void)r; // suppress warning unused variable
    }

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
        (void)r; // suppress warning unused variable
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
        printf("%.10f\n", input[i]);
    }
}

template <>
void print_data(IDATA_T* input, LDATA_T length) {
    for (LDATA_T i = 0; i < length; i ++) {
        printf("%d\n", input[i]);
    }
}

template <>
FDATA_T** malloc_2d_array(IDATA_T row, IDATA_T col) {
    FDATA_T** arr = (FDATA_T**) malloc(row * sizeof(FDATA_T*));
    for (int i = 0; i < row; i++) {
        arr[i] = (FDATA_T*) malloc(col * sizeof(FDATA_T));
    }

    return arr;
}

template <>
void free_2d_array(FDATA_T** arr, IDATA_T row, IDATA_T col) {
    for (int i = 0; i < row; i++) {
        free(arr[i]);
    }
    free(arr);
}

template <>
void transpose(FDATA_T* src, FDATA_T* dst, const IDATA_T ROW, const IDATA_T COL)
{
  // transpose array
  // the source array has shape of (row, col)

  for (IDATA_T row = 0; row < ROW; row++) {
    for (IDATA_T col = 0; col < COL; col++) 
      dst[col * ROW + row] = src[row * COL + col];
  }
}  
