#include <stdio.h>
#include "../constants.h"

void zero_init(float* input, int length)
{
	for(int idx = 0; idx < length; idx++)
		input[idx] = 0;
}

void load_float(char* fname, int length, float* array, int choice)
{
  FILE *myfile;
  switch(choice) {
    case 1:
      myfile=fopen("./embedding_1_embeddings.txt", "r");
      break;
    case 2:
      myfile=fopen("./simple_rnn_1_bias.txt", "r");
    case 3:
      myfile=fopen("./simple_rnn_1_kernel.txt", "r");
    case 4:
      myfile=fopen("./simple_rnn_1_recurrent_kernel.txt", "r");
  }

  // for(int i = 0; i < length; i++)
  // {
  //     fscanf(myfile,"%f", &array[i]);
  // }

  fclose(myfile);
}

int main(void)
{
  float word_embedding[WORD_NUM * WORD_SIZE];
  float rnn_bias[RNN_STATE_SIZE];
  float rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE];
  float rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE]; 
  float fc_bias[FC_OUTPUT_SIZE];
  float fc_kernel[FC_INPUT_SIZE * FC_OUTPUT_SIZE];

  zero_init(word_embedding, WORD_NUM * WORD_SIZE);
  zero_init(rnn_bias, RNN_STATE_SIZE);
  zero_init(rnn_kernel, RNN_INPUT_SIZE * RNN_STATE_SIZE);
  zero_init(rnn_recurrent_kernel, RNN_STATE_SIZE * RNN_STATE_SIZE);
  // zero_init(fc_bias, FC_OUTPUT_SIZE);
  // zero_init(fc_kernel, FC_INPUT_SIZE * FC_OUTPUT_SIZE);
  
  // char str1[] = "./embedding_1_embeddings.txt";
  // char str2[] = "./simple_rnn_1_bias.txt";
  // char str3[] = "./simple_rnn_1_kernel.txt";
  // char str4[] = "./simple_rnn_1_recurrent_kernel.txt";
  // load_float(str1, WORD_NUM * WORD_SIZE, word_embedding, 1);
  // load_float(str2, RNN_STATE_SIZE, rnn_bias, 2);
  // load_float(str3, RNN_INPUT_SIZE * RNN_STATE_SIZE, rnn_kernel, 3);
  // load_float(str4, RNN_STATE_SIZE * RNN_STATE_SIZE, rnn_recurrent_kernel, 4);
  // load_float("./dense_1_bias.txt", FC_OUTPUT_SIZE, fc_bias);
  // load_float("./dense_1_kernel.txt", FC_INPUT_SIZE * FC_OUTPUT_SIZE, fc_kernel);

  int sampleNum = 5;
  for (int i = 0; i <= sampleNum; i++) {
  	int index = ((int) (float) i / (float) sampleNum) * RNN_STATE_SIZE * RNN_STATE_SIZE - 1;
  	index = index < 0? 0 : index;
  	printf("index:%d value:%f\n", index, rnn_recurrent_kernel[index]);
  }

}


