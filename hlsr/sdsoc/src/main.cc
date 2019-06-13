#include <cstdio>
#include <cstdlib>

#include "activation.h"
#include "config.h"
#include "constants.h"
// #include "fc.h"
#include "init.h"
// #include "rnn.h"
#include "softmax.h"
#include "types.h"
#include "utils.h"
#include "wrapper.h"

#define abs(x) x > 0? x: 0

#ifdef __SDSCC__
#include "sds_lib.h"
class perf_counter {
  public:
    long unsigned tot, cnt, calls;
    perf_counter() : tot(0), cnt(0), calls(0) {};
    inline void reset() { tot = cnt = calls = 0; }
    inline void start() { cnt = sds_clock_counter(); calls++; };
    inline void stop() { tot += (sds_clock_counter() - cnt); };
    inline long unsigned avg_cpu_cycles() { return (tot / calls); };
};
#define MALLOC sds_alloc
#define MFREE sds_free
#else
#define MALLOC MALLOC
#define MFREE free
#endif



int main(int argc, char *argv[]) {
  printf("INFO: C-RNN\n\r");
#ifdef __SDSCC__
  perf_counter f_ctr;
#endif
  printf("INFO: memory alloc\n\r");
  // declare weights
  // embedding
#ifdef __SDSCC__
  f_ctr.start();
#endif
  FDATA_T* word_embedding = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * WORD_NUM * WORD_SIZE);

  // RNN
  FDATA_T* rnn_bias = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE);
  FDATA_T* rnn_kernel = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_INPUT_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_kernel_transpose = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_INPUT_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_recurrent_kernel = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE * RNN_STATE_SIZE);
  FDATA_T* rnn_recurrent_kernel_transpose = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * RNN_STATE_SIZE * RNN_STATE_SIZE);

  FDATA_T* rnn_input_states = /* store all input states */
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * COMPUTE_TIME * SAMPLE_LEN * 
                        BATCH_SIZE * RNN_INPUT_SIZE);
printf("size:\t%d\n", sizeof(FDATA_T) * COMPUTE_TIME * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE);
  // FC
  FDATA_T* fc_bias = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);
  FDATA_T* fc_kernel_transpose = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_INPUT_SIZE * FC_OUTPUT_SIZE);

  FDATA_T* fc_output_feature_map = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T)*COMPUTE_TIME*BATCH_SIZE*FC_OUTPUT_SIZE);
  FDATA_T* softmax_result = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T)*COMPUTE_TIME*BATCH_SIZE*SM_CLASS_SIZE);
  IDATA_T* argmax_result = 
      (IDATA_T*) malloc(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);

  // Dataset
  IDATA_T* sequences = 
      (IDATA_T*) MALLOC(sizeof(IDATA_T)*COMPUTE_TIME * BATCH_SIZE * SAMPLE_LEN);
  IDATA_T* C_result = 
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);
  IDATA_T* Keras_result =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);
  IDATA_T* Actual_result =
      (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME * BATCH_SIZE);

#ifdef __SDSCC__
  f_ctr.stop();
  printf("INFO:   cpu cycles %lu\n\r", f_ctr.avg_cpu_cycles());
#endif
  printf("INFO: model load\n\r");

  // load weights
  load_data<FDATA_T, LDATA_T>(EMBEDDINGS_FILE, word_embedding, 
                              WORD_NUM * WORD_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_BIAS_FILE, rnn_bias, RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_KERNEL_FILE, rnn_kernel, 
                              RNN_INPUT_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(SIMPLE_RNN_RECURRENT_KERNEL_FILE, 
                              rnn_recurrent_kernel, 
                              RNN_STATE_SIZE * RNN_STATE_SIZE);
  load_data<FDATA_T, LDATA_T>(DENSE_BIAS_FILE, fc_bias, FC_OUTPUT_SIZE);
  load_data<FDATA_T, LDATA_T>(DENSE_KERNEL_FILE, fc_kernel, 
                              FC_INPUT_SIZE * FC_OUTPUT_SIZE);
// transpose kernels                     <
  transpose<FDATA_T, LDATA_T>(rnn_kernel, rnn_kernel_transpose, RNN_INPUT_SIZE, 
                              RNN_STATE_SIZE);
  transpose<FDATA_T, LDATA_T>(
      rnn_recurrent_kernel, rnn_recurrent_kernel_transpose, RNN_STATE_SIZE, 
      RNN_STATE_SIZE);
  transpose<FDATA_T, LDATA_T>(fc_kernel, fc_kernel_transpose, FC_INPUT_SIZE,
                              FC_OUTPUT_SIZE);

// free untransposed kernels
  MFREE(rnn_kernel);
  MFREE(rnn_recurrent_kernel);
  MFREE(fc_kernel);
#ifdef DEBUG
  print_data<FDATA_T, LDATA_T>(fc_kernel, FC_INPUT_SIZE * FC_OUTPUT_SIZE);
#endif

  printf("INFO: data load\n\r");
#ifdef __SDSCC__
  f_ctr.start();
#endif
  // load dataset
  load_data<IDATA_T, LDATA_T>(
      ORG_SEQ_FILE, sequences, COMPUTE_TIME * BATCH_SIZE * SAMPLE_LEN);
  load_data<IDATA_T, LDATA_T>(RNN_RESULT_FILE, Keras_result, 
                              COMPUTE_TIME * BATCH_SIZE);
  load_data<IDATA_T, LDATA_T>(ACTUAL_RESULT_FILE, Actual_result,
                              COMPUTE_TIME * BATCH_SIZE);
  // arrange input states 
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME; compute_time++) {

    for (LDATA_T seq_idx = 0; seq_idx < SAMPLE_LEN; seq_idx++) {

      for (LDATA_T batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
  
        // seq alignment [ ... first 50 words ...][ ... second 50 words ...]
        LDATA_T row = compute_time * BATCH_SIZE + batch_idx;
        LDATA_T col = seq_idx;
        LDATA_T sample_index = row * SAMPLE_LEN + col; 
        LDATA_T word_index = sequences[sample_index];
        LDATA_T word_embedding_index = word_index * WORD_SIZE;

        // input state alignment
        // [COMPUTE_TIME][SAMPLE_LEN][BATCH_SIZE][RNN_INPUT_SIZE]
        LDATA_T rnn_input_states_index = 
            compute_time * SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE +
            seq_idx * BATCH_SIZE * RNN_INPUT_SIZE + batch_idx * RNN_INPUT_SIZE;
        copy_data<FDATA_T, LDATA_T>(&word_embedding[word_embedding_index], 
                                    &rnn_input_states[rnn_input_states_index], 
                                    RNN_INPUT_SIZE);
      }
    }
  }
#ifdef __SDSCC__
  f_ctr.stop();
  printf("INFO:   cpu cycles %lu\n\r", f_ctr.avg_cpu_cycles());
#endif
  printf("INFO: run inference\n\r");
#ifdef __SDSCC__
  f_ctr.start();
#endif
/*
  wrapper_rnn_fc(rnn_kernel_transpose, rnn_recurrent_kernel_transpose,
                 rnn_bias, fc_kernel_transpose, fc_bias, rnn_input_states, 
                 fc_output_feature_map);
#ifdef __SDSCC__
  f_ctr.stop();
  printf("INFO:   cpu cycles %lu\n\r", f_ctr.avg_cpu_cycles());
#endif

  // Softmax and Argmax
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME; compute_time++) {

    // softmax and argmax are for a single batch
    LDATA_T fc_output_offset = compute_time * BATCH_SIZE * FC_OUTPUT_SIZE;
    LDATA_T softmax_offset = compute_time * BATCH_SIZE * SM_CLASS_SIZE;
    LDATA_T argmax_offset = compute_time * BATCH_SIZE;
    softmax<FDATA_T>(fc_output_feature_map + fc_output_offset, 
                     softmax_result + softmax_offset);
    argmax<FDATA_T, IDATA_T>(fc_output_feature_map + fc_output_offset, 
                             argmax_result + argmax_offset);
  }*/
  copy_data<IDATA_T, LDATA_T>(argmax_result, C_result, COMPUTE_TIME*BATCH_SIZE);

  // Correctness
  LDATA_T count_Keras = 0;   
  LDATA_T count_C = 0;       
  LDATA_T count_times = COMPUTE_TIME * BATCH_SIZE;

  for (int i = 0; i < COMPUTE_TIME * BATCH_SIZE; i++) {
    if (Keras_result[i] == Actual_result[i])
        count_Keras++;
    if (C_result[i] == Actual_result[i])
        count_C++;
#ifdef VERBOSE
    if (C_result[i] == Keras_result[i])
      printf("INFO: Sample %d:\t result: %d\n", i, C_result[i]);
    else {
      printf("INFO: Sample %d:\t C_result: %d\t Keras_result: %d\t Actual_result: %d\t P(%d): %f\t P(%d): %lf\n",
              i, C_result[i], Keras_result[i], Actual_result[i],
              C_result[i], TOFLOAT(softmax_result[i * SM_CLASS_SIZE + C_result[i]]),
              Keras_result[i], TOFLOAT(softmax_result[i * SM_CLASS_SIZE + Keras_result[i]]));
    }
#endif
  }

  printf("INFO: Correctness:\n\r");
  printf("INFO:   Keras: %f\n\r", (float) count_Keras / count_times);
  printf("INFO:   C:     %f\n\r", (float) count_C / count_times);

  MFREE(word_embedding);
  MFREE(rnn_bias);
  MFREE(rnn_kernel_transpose);
  MFREE(rnn_recurrent_kernel_transpose);
  MFREE(fc_bias);
  MFREE(fc_kernel_transpose);
  MFREE(fc_output_feature_map);
  MFREE(softmax_result);
  MFREE(argmax_result);
  MFREE(sequences);
  MFREE(C_result);
  MFREE(Keras_result);
  MFREE(Actual_result);

  return 0;
}
