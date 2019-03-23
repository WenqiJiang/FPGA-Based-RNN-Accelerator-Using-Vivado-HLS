#include <stdio.h>
#include <stdlib.h>

#include "activation.h"
#include "constants.h"
#include "fc.h"
#include "load_data.h"
#include "params_init.h"
#include "rnn.h"
#include "softmax.h"

int main(int argc, char *argv[])
{
    /* declare weights */
    /* embedding */
    // float word_embedding[WORD_NUM * WORD_SIZE];
    float* word_embedding = malloc(sizeof(float) * WORD_NUM * WORD_SIZE);

    /* rnn */
    // float rnn_last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE];
    // float rnn_input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE];
    // float rnn_bias[RNN_STATE_SIZE];
    // float rnn_kernel[RNN_INPUT_SIZE * RNN_STATE_SIZE];
    // float rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE]; 
    // float rnn_output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE];

    float* rnn_last_state = malloc(sizeof(float) * RNN_BATCH_SIZE * RNN_STATE_SIZE);
    float* rnn_input_state = malloc(sizeof(float) * RNN_BATCH_SIZE * RNN_INPUT_SIZE);
    float* rnn_bias = malloc(sizeof(float) * RNN_STATE_SIZE);
    float* rnn_kernel = malloc(sizeof(float) * RNN_INPUT_SIZE * RNN_STATE_SIZE);
    float* rnn_recurrent_kernel = malloc(sizeof(float) * RNN_STATE_SIZE * RNN_STATE_SIZE); 
    float* rnn_output_state = malloc(sizeof(float) * RNN_BATCH_SIZE * RNN_STATE_SIZE);
    
    /* fc */
    // float fc_input_feature_map[FC_BATCH_SIZE * FC_INPUT_SIZE];
    // float fc_bias[FC_OUTPUT_SIZE];
    // float fc_kernel[FC_INPUT_SIZE * FC_OUTPUT_SIZE];
    // float fc_output_feature_map[FC_BATCH_SIZE * FC_OUTPUT_SIZE];
    float* fc_input_feature_map = malloc(sizeof(float) * FC_BATCH_SIZE * FC_INPUT_SIZE);
    float* fc_bias = malloc(sizeof(float) * FC_OUTPUT_SIZE);
    float* fc_kernel = malloc(sizeof(float) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);
    float* fc_output_feature_map = malloc(sizeof(float) * FC_BATCH_SIZE * FC_OUTPUT_SIZE);
    float* softmax_result = malloc(sizeof(float) * SM_BATCH_SIZE * SM_CLASS_SIZE);
    int* argmax_result = malloc(sizeof(int) * SM_BATCH_SIZE);

    /* load model in */
    load_float("./model/embedding_1_embeddings.txt", WORD_NUM * WORD_SIZE, word_embedding);
    load_float("./model/simple_rnn_1_bias.txt", RNN_STATE_SIZE, rnn_bias);
    load_float("./model/simple_rnn_1_kernel.txt", RNN_INPUT_SIZE * RNN_STATE_SIZE, rnn_kernel);
    load_float("./model/simple_rnn_1_recurrent_kernel.txt", 
                RNN_STATE_SIZE * RNN_STATE_SIZE, rnn_recurrent_kernel);
    load_float("./model/dense_1_bias.txt", FC_OUTPUT_SIZE, fc_bias);
    load_float("./model/dense_1_kernel.txt", FC_INPUT_SIZE * FC_OUTPUT_SIZE, fc_kernel);

    /* load dataset in */
    #define SAMPLE_NUM 1000
    #define SAMPLE_LEN 50
    int* sequences = malloc(sizeof(int) * SAMPLE_LEN * SAMPLE_NUM);
    int* C_result = malloc(sizeof(int) * SAMPLE_NUM);
    int* Keras_result = malloc(sizeof(int) * SAMPLE_NUM);
    load_int("./datasets/org_seq.txt", SAMPLE_LEN * SAMPLE_NUM, sequences);
    load_int("./datasets/rnn_result.txt", SAMPLE_NUM, Keras_result);

    /* do inference and print the result */
    for (int compute_time = 0; compute_time < SAMPLE_NUM / FC_BATCH_SIZE; compute_time++) {
        /* initialize last state to 0 */
        zero_init(rnn_last_state, RNN_BATCH_SIZE * RNN_STATE_SIZE);
        for (int i = 0; i < SAMPLE_LEN; i++) {
            for (int j = 0; j < RNN_BATCH_SIZE; j++) {
                int sample_index = compute_time * SAMPLE_LEN * RNN_BATCH_SIZE + j * SAMPLE_LEN + i;
                int word_index = sequences[sample_index];
                int word_embedding_index = word_index * WORD_SIZE;
                int rnn_input_state_index = j * RNN_INPUT_SIZE;
                copy_float(&word_embedding[word_embedding_index], 
                        &rnn_input_state[rnn_input_state_index], RNN_INPUT_SIZE);
            }

            rnn(rnn_last_state, rnn_input_state, rnn_bias, rnn_kernel, rnn_recurrent_kernel, rnn_output_state);
            act_tanh(rnn_output_state, RNN_BATCH_SIZE * RNN_STATE_SIZE);
            float* temp = rnn_last_state;
            rnn_last_state = rnn_output_state;
            rnn_output_state = temp;
        }
        /* no ReLu after FC layer, only softmax */
        fc(rnn_last_state, fc_bias, fc_kernel, fc_output_feature_map);
        // act_relu(fc_output_feature_map, FC_BATCH_SIZE * FC_OUTPUT_SIZE);
        softmax(fc_output_feature_map, softmax_result);
        // print_float(softmax_result, SM_CLASS_SIZE);
        argmax(fc_output_feature_map, argmax_result);
        copy_int(argmax_result, &C_result[compute_time * RNN_BATCH_SIZE], RNN_BATCH_SIZE);
        // print_int(argmax_result, SM_BATCH_SIZE);
        for (int i = compute_time * RNN_BATCH_SIZE; i < (compute_time + 1) * RNN_BATCH_SIZE; i++) {
            if (C_result[i] == Keras_result[i])
                printf("Sample %d:\t result: %d\n", i, C_result[i]);
            else {
                printf("Sample %d:\t C_result: %d\t Keras_result: %d\t P(%d): %f\t P(%d): %f\n", 
                i, C_result[i], Keras_result[i], 
                C_result[i], softmax_result[(i - compute_time * RNN_BATCH_SIZE) * SM_CLASS_SIZE + C_result[i]],
                Keras_result[i], softmax_result[(i - compute_time * RNN_BATCH_SIZE) * SM_CLASS_SIZE + Keras_result[i]]);
            }           
        }
    }
    
    printf("%s", "Press any key then ENTER to end the program\n");
    char str2[10];
    // scanf("%s", str2);

    return 0;
}
