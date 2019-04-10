#include "utils.h"

#include <cstdio>

template <>
void load_data(char const* fname, FDATA_T* array, LDATA_T length) {
    FILE *myfile;
    myfile=fopen(fname, "r");

    for(LDATA_T i = 0; i < length; i++)
    {
        LDATA_T r = fscanf(myfile,"%40f", &array[i]);
        (void)r; // suppress warning unused variable
    }

    fclose(myfile);
}

template <>
void load_data(char const* fname, IDATA_T* array, LDATA_T length) {
    FILE *myfile;
    myfile=fopen(fname, "r");

    for(LDATA_T i = 0; i < length; i++)
    {
        LDATA_T r = fscanf(myfile,"%d", &array[i]);
        (void)r; // suppress warning unused variable
    }

    fclose(myfile);
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

