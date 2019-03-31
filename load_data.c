#include <stdio.h>

void load_int(char* fname, int length, int* array)
{
  FILE *myfile;
  myfile=fopen(fname, "r");

  for(int i = 0; i < length; i++)
  {
      fscanf(myfile,"%d", &array[i]);
  }

  fclose(myfile);
}

void load_float(char* fname, int length, float* array)
{
  FILE *myfile;
  myfile=fopen(fname, "r");

  for(int i = 0; i < length; i++)
  {
      fscanf(myfile,"%f", &array[i]);
  }

  fclose(myfile);
}

void load_double(char* fname, int length, double* array)
{
  FILE *myfile;
  myfile=fopen(fname, "r");

  for(int i = 0; i < length; i++)
  {
      fscanf(myfile,"%lf", &array[i]);
  }

  fclose(myfile);
}


void copy_int(int* copy_from, int* copy_to, int length) {
  for (int i = 0; i < length; i++) {
    copy_to[i] = copy_from[i];
  }
}

void copy_float(float* copy_from, float* copy_to, int length) {
  for (int i = 0; i < length; i++) {
    copy_to[i] = copy_from[i];
  }
}

void copy_double(double* copy_from, double* copy_to, int length) {
  for (int i = 0; i < length; i++) {
    copy_to[i] = copy_from[i];
  }
}

void print_int(int* input, int length) {
  for (int i = 0; i < length; i ++) {
    printf("%d\n", input[i]);
  }
}

void print_float(float* input, int length) {
  for (int i = 0; i < length; i ++) {
    printf("%.10f\n", input[i]);
  }
}


void print_double(double* input, int length) {
  for (int i = 0; i < length; i ++) {
    printf("%.10lf\n", input[i]);
  }
}