#include <stdio.h>

#define LEN 1619200

int main(void)
{
  FILE *myfile;
  float weights[LEN];

  myfile=fopen("./embedding_1_embeddings.txt", "r");

  for(int i = 0; i < LEN; i++)
  {
      fscanf(myfile,"%30f", &weights[i]);
      printf("%d: %.30f \n", i, weights[i]);
  }

  int sampleNum = 5;
  for (int i = 0; i <= sampleNum; i++) {
  	int index = ((int) (float) i / (float) sampleNum) * LEN - 1;
  	index = index < 0? 0 : index;
  	printf("index:%d value:%f\n", index, weights[index]);
  }

  fclose(myfile);
}
