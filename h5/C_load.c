#include <stdio.h>

#define LEN 128

int main(void)
{
  FILE *myfile;
  float bias[LEN];

  myfile=fopen("./dense_1_bias.txt", "r");

  for(int i = 0; i < LEN; i++)
  {
      fscanf(myfile,"%f", &bias[i]);
      printf("%d: %f \n", i, bias[i]);
  }

  fclose(myfile);
}
