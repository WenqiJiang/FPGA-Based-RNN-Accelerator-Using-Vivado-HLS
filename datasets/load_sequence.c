#include <stdio.h>

#define LEN 50000 // 1000 * 50

int main(void)
{
  FILE *myfile;
  int weights[LEN];

  myfile=fopen("./org_seq.txt", "r");

  for(int i = 0; i < LEN; i++)
  {
      fscanf(myfile,"%d", &weights[i]);
      printf("%d: %d \n", i, weights[i]);
  }

  int sampleNum = 5;
  for (int i = 0; i <= sampleNum; i++) {
  	int index = ((int) (float) i / (float) sampleNum) * LEN - 1;
  	index = index < 0? 0 : index;
  	printf("index:%d value:%d\n", index, weights[index]);
  }

  fclose(myfile);
}
