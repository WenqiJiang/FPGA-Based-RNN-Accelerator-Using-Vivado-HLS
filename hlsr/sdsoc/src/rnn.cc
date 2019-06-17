#include "rnn.h"

#include "constants.h"
#include "types.h"
#include "wrapper.h"


// copy a constant amount of data [RNN_STATE_SIZE]
void copy_rnn_bias(FDATA_T src[RNN_STATE_SIZE], 
                   FDATA_T dst[RNN_STATE_SIZE]) {
#pragma HLS inline region
  for (LDATA_T i = 0; i < RNN_STATE_SIZE; i++) {
#pragma HLS pipeline rewind
    dst[i] = src[i];
  }
}

// copy a constant amount of data [RNN_STATE_SIZE * RNN_STATE_SIZE]
void copy_rnn_recurrent_kernel(FDATA_T src[RNN_STATE_SIZE * RNN_STATE_SIZE],
                               FDATA_T dst[RNN_STATE_SIZE * RNN_STATE_SIZE]) {
#pragma HLS inline region                                                       
  for (LDATA_T i = 0; i < RNN_STATE_SIZE * RNN_STATE_SIZE; i++) { 
#pragma HLS pipeline rewind
    dst[i] = src[i];
  }
}

// copy a constant amount of data [RNN_STATE_SIZE * RNN_INPUT_SIZE]
void copy_rnn_kernel(FDATA_T src[RNN_STATE_SIZE * RNN_INPUT_SIZE],
                     FDATA_T dst[RNN_STATE_SIZE * RNN_INPUT_SIZE]) {
#pragma HLS inline region                                                       
  for (LDATA_T i = 0; i < RNN_STATE_SIZE * RNN_INPUT_SIZE; i++) {                                
#pragma HLS pipeline rewind
    dst[i] = src[i];
  }                
}

// copy a constant amount of data [BATCH_SIZE * RNN_STATE_SIZE]
void copy_rnn_output_state(FDATA_T src[BATCH_SIZE * RNN_STATE_SIZE],
                       FDATA_T dst[BATCH_SIZE * RNN_STATE_SIZE] ) {
#pragma HLS inline region                                                       
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_STATE_SIZE; i++) {
#pragma HLS pipeline rewind 
    dst[i] = src[i];       
  }                                                     
}

// initialize a constant amount of data [BATCH_SIZE * RNN_STATE_SIZE]
void init_state(FDATA_T state[BATCH_SIZE * RNN_STATE_SIZE]) {                   
#pragma HLS inline region                                                       
  for (LDATA_T i = 0; i < BATCH_SIZE * RNN_STATE_SIZE; i++) {
#pragma HLS pipeline rewind 
    state[i] = 0;          
  }                                                     
}

void rnn_load_input_state(FDATA_T input_state_part[TILE_BATCH * RNN_INPUT_SIZE],
                          FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE]) {

  // load a batch of input state, the batch size is TILE_BATCH
  // start from a certain batch index (decided when function call)
  // input_state  --- load to ---> input_state_reg

  for (LDATA_T batch_iter = 0; batch_iter < TILE_BATCH; batch_iter++) {

    LDATA_T input_state_start_index = batch_iter * RNN_INPUT_SIZE;

    for (LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
         input_state_index++) {
//#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE rewind

      input_state_reg[batch_iter][input_state_index] =
          input_state_part[input_state_start_index + input_state_index];
    }
  }
}

void rnn_load_last_state(FDATA_T last_state_part[TILE_BATCH * RNN_STATE_SIZE],
                         FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE]) {

  // load a batch of last state, the batch size is TILE_BATCH
  // start from a certain start batch index (decided when function call)
  // last_state  --- load to ---> last_state_reg

  for (LDATA_T batch_iter = 0; batch_iter < TILE_BATCH; batch_iter++) {

    LDATA_T last_state_start_index = batch_iter * RNN_STATE_SIZE;
    for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
         last_state_index++) {
//#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE rewind

      last_state_reg[batch_iter][last_state_index] =
          last_state_part[last_state_start_index + last_state_index];
    }
  }
}

void rnn_load_kernel(FDATA_T kernel_part[RNN_INPUT_SIZE],
                     FDATA_T kernel_reg[RNN_INPUT_SIZE]) {
//#pragma HLS inline region
  // load the (output_state_index)'th column of kernel
  // used this column do matrix multiplication
  // the kernel start from a certain index (decided when function call)
  // kernel --- load to ---> kernel_reg

  for (LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
       input_state_index++) {
//#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE rewind

    kernel_reg[input_state_index] = kernel_part[input_state_index];
  }
}

void rnn_load_recurrent_kernel(FDATA_T recurrent_kernel_part[RNN_STATE_SIZE],
                               FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE]) {
// #pragma HLS inline region
  // load the (output_state_index)'th column of recurrent_kernel
  // used this column do matrix multiplication
  // recurrent_kernel --- load to ---> recurrent_kernel_reg

  for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
       last_state_index++) {
//#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE rewind

    recurrent_kernel_reg[last_state_index] = 
        recurrent_kernel_part[last_state_index];
  }
}

void rnn_compute(FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
                 FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
                 FDATA_T kernel_reg[RNN_INPUT_SIZE],
                 FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE],
                 FDATA_T output_state_reg_part[TILE_BATCH]) {
//#pragma HLS inline region
  // take a batch of input_state and last_state,
  //  rnn_compute the output state, and store into the output_state_reg
  // note that we don't add bias here, the bias addition will be done in
  //  function "rnn_save_output_state"
  // input: input_state_reg, last_state_reg, kernel_reg,
  //          recurrent_kernel_reg, output_state_index
  // output: output_state_reg

#define COMPUTE_UNROLL 8
  FDATA_T local_reg[COMPUTE_UNROLL][RNN_STATE_SIZE + RNN_INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_reg cyclic factor=16 dim=2

for (LDATA_T tile_iter = 0; tile_iter < TILE_BATCH / COMPUTE_UNROLL;
     tile_iter++) {
    for (LDATA_T batch_iter = 0; batch_iter < COMPUTE_UNROLL; batch_iter++) {
/////// HACKING, factor should be consistent with COMPUTE_UNROLL //////
/////// can not use macro as factor here due to the HLS syntax   //////
#pragma HLS UNROLL factor=8
//#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
//#pragma HLS RESOURCE variable=kernel_reg core=AddSub_DSP

      for (LDATA_T input_state_index = 0; input_state_index < RNN_INPUT_SIZE;
           input_state_index++) {
#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
#pragma HLS UNROLL factor=32

        local_reg[batch_iter][input_state_index] =
            kernel_reg[input_state_index] *
            input_state_reg[tile_iter * COMPUTE_UNROLL + batch_iter]
            [input_state_index];
      }

      for (LDATA_T last_state_index = 0; last_state_index < RNN_STATE_SIZE;
           last_state_index++) {
#pragma HLS RESOURCE variable=local_reg core=FMul_fulldsp
#pragma HLS UNROLL factor=32

        local_reg[batch_iter][RNN_INPUT_SIZE + last_state_index] =
            recurrent_kernel_reg[last_state_index] *
            last_state_reg[tile_iter * COMPUTE_UNROLL + batch_iter]
            [last_state_index];
      }

      ////// HACKING, suppose RNN_STATE_SIZE + RNN_INPUT_SIZE = 228 /////

      // prefix sum
      for (LDATA_T i = 0; i < 114; i++) {
#pragma HLS UNROLL factor=32
#pragma HLS RESOURCE variable=local_reg core=AddSub_DSP
  //#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][114 + i];
      }

      for (LDATA_T i = 0; i < 57; i++) {
#pragma HLS UNROLL factor=32
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][57 + i];
      }

      // 57 = 28 * 2 + 1 -> need 29 reg for next iteration
      // the 57'th number will be copy to 29'th reg
      for (LDATA_T i = 0; i < 28; i++) {
#pragma HLS UNROLL complete
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][28 + i];
      }
      local_reg[batch_iter][28] = local_reg[batch_iter][56];

      // 29 = 14 * 2 + 1 -> need 15 reg for next iteration
      // the 29'th number will be copy to 15'th reg
      for (LDATA_T i = 0; i < 14; i++) {
#pragma HLS UNROLL complete
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][14 + i];
      }
      local_reg[batch_iter][14] = local_reg[batch_iter][28];

      // 15 = 7 * 2 + 1 -> need 8 reg for next iteration
      // the 15'th number will be copy to 8'th reg
      for (LDATA_T i = 0; i < 7; i++) {
#pragma HLS UNROLL complete
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][7 + i];
      }
      local_reg[batch_iter][7] = local_reg[batch_iter][14];

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 4; i++) {
#pragma HLS UNROLL complete
  //#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][4 + i];
      }

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 2; i++) {
#pragma HLS UNROLL complete
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][2 + i];
      }

      // from 8, regular prefix sum
      for (LDATA_T i = 0; i < 1; i++) {
#pragma HLS UNROLL complete
#pragma HLS RESOURCE variable=local_reg core=FAddSub_fulldsp

        local_reg[batch_iter][i] = local_reg[batch_iter][i] +
                                   local_reg[batch_iter][1 + i];
      }

      output_state_reg_part[tile_iter * COMPUTE_UNROLL + batch_iter] =
          local_reg[batch_iter][0];
    }
  }
}

void rnn_load_kernels_and_compute(
    FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE],
    FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE],
    FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
    FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
    FDATA_T output_state_reg[RNN_STATE_SIZE][TILE_BATCH]) {

  // take loaded states as input, with a batch size of TILE_BATCH
  // load kernel then rnn_compute
  // input: input_state_reg, last_state_reg, kernel_reg, recurrent_kernel_reg
  // output: output_state_reg
//#pragma HLS inline region

  FDATA_T kernel_reg[RNN_INPUT_SIZE];
  FDATA_T recurrent_kernel_reg[RNN_STATE_SIZE];
#pragma HLS ARRAY_PARTITION variable=kernel_reg cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=recurrent_kernel_reg cyclic factor=16 dim=1

  for (LDATA_T output_state_index = 0; output_state_index < RNN_STATE_SIZE;
       output_state_index++) {
#pragma HLS DATAFLOW

    // load
    rnn_load_kernel(kernel + output_state_index * RNN_INPUT_SIZE, kernel_reg);
    rnn_load_recurrent_kernel(
        recurrent_kernel + output_state_index * RNN_STATE_SIZE, 
        recurrent_kernel_reg);

    // rnn_compute a batch of output (but haven't add bias)
    rnn_compute(input_state_reg, last_state_reg, kernel_reg,
                recurrent_kernel_reg,
                &output_state_reg[output_state_index][0]);
  }
}

void rnn_save_output_state(FDATA_T output_state_reg[RNN_STATE_SIZE][TILE_BATCH],
                           FDATA_T bias[RNN_STATE_SIZE],
                           FDATA_T output_state[TILE_BATCH * RNN_STATE_SIZE]) {

  // the output state in register is not the final result,
  // add bias to finish computing and store them into BRAM
  // the output state starts from a certain index (decided when function call)
  // output state memory layout [TILE_BATCH][RNN_STATE_SIZE]
  // output_state_reg + bias --- load to ---> output_state

  for (LDATA_T batch_iter = 0; batch_iter < TILE_BATCH; batch_iter++) {

    LDATA_T output_state_start_index = batch_iter * RNN_STATE_SIZE;
    for (LDATA_T output_state_index = 0;
         output_state_index < RNN_STATE_SIZE; output_state_index++) {
//#pragma HLS UNROLL factor=2
#pragma HLS PIPELINE
//    	FDATA_T tmp = bias[output_state_index] +
//    	          output_state_reg[output_state_index][batch_iter];

      output_state[output_state_start_index + output_state_index] =
//          hls::tanh<FXD_W_LENGTH, FXD_I_LENGTH>(
//          hls::tanh<FXD_W_LENGTH, FXD_I_LENGTH>(tmp);
    		  tanh(bias[output_state_index] +
    	          output_state_reg[output_state_index][batch_iter]);
    }
  }
}

void rnn(FDATA_T last_state[RNN_BATCH_SIZE * RNN_STATE_SIZE],
         FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
         FDATA_T bias[RNN_STATE_SIZE],
         FDATA_T kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE],
         FDATA_T recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE],
         FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]) {

  // please do INITIALIZATION before input output_state
  // ------- DIMENSION SETTING  ---------- *
  //
  //   input_state: RNN_BATCH_SIZE * RNN_INPUT_SIZE (None * 100)
  //   last_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None * 128)
  //   bias: RNN_STATE_SIZE (128)
  //   kernel: RNN_INPUT_SIZE * RNN_STATE_SIZE (100 * 128)
  //   recurrent_kernel: RNN_STATE_SIZE * RNN_STATE_SIZE (128 * 128)
  //   output_state: RNN_BATCH_SIZE * RNN_STATE_SIZE (None, 128)

  // declare registers and use array partition
  // tile = 32
//#pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=recurrent_kernel cyclic factor=2
  FDATA_T input_state_reg[TILE_BATCH][RNN_INPUT_SIZE];
  FDATA_T last_state_reg[TILE_BATCH][RNN_STATE_SIZE];
  FDATA_T output_state_reg[RNN_STATE_SIZE][TILE_BATCH];

#pragma HLS ARRAY_PARTITION variable=input_state_reg cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=last_state_reg cyclic factor=16 dim=2
///// HACKING! output_reg factor should be the same as COMPUTE_UNROLL /////
///// but HLS syntax does not support macro as parameter /////
#pragma HLS ARRAY_PARTITION variable=output_state_reg cyclic factor=8 dim=2

BATCH:
  for (LDATA_T batch_iter = 0; batch_iter < RNN_BATCH_SIZE / TILE_BATCH;
       batch_iter++) {
// #pragma HLS DATAFLOW

    // load
    rnn_load_input_state(input_state + batch_iter * TILE_BATCH * RNN_INPUT_SIZE, 
                         input_state_reg);
    rnn_load_last_state(last_state + batch_iter * TILE_BATCH * RNN_STATE_SIZE, 
                        last_state_reg);

    // rnn_compute + load kernel
    rnn_load_kernels_and_compute(input_state_reg, last_state_reg, kernel,
                                 recurrent_kernel, output_state_reg);

    // save
    rnn_save_output_state(output_state_reg, bias,
    		              &output_state[batch_iter * TILE_BATCH * RNN_STATE_SIZE]);
  }
}

#pragma SDS data copy(rnn_kernel[0: RNN_STATE_SIZE * RNN_INPUT_SIZE])           
#pragma SDS data copy(rnn_recurrent_kernel \
                      [0: RNN_STATE_SIZE * RNN_STATE_SIZE])                
#pragma SDS data copy(rnn_bias[0: RNN_STATE_SIZE])

#pragma SDS data zero_copy( \
    input_state[0: SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE])
#pragma SDS data zero_copy( \
    output_state[0: BATCH_SIZE * RNN_STATE_SIZE])

#pragma SDS data access_pattern(rnn_kernel:SEQUENTIAL, \
    rnn_recurrent_kernel:SEQUENTIAL, \
    rnn_bias:SEQUENTIAL) 

void wrapper_rnn(FDATA_T rnn_bias[RNN_STATE_SIZE], 
                 FDATA_T rnn_kernel[RNN_STATE_SIZE * RNN_INPUT_SIZE], 
                 FDATA_T rnn_recurrent_kernel[RNN_STATE_SIZE * RNN_STATE_SIZE], 
                 FDATA_T input_state[RNN_BATCH_SIZE * RNN_INPUT_SIZE],
                 FDATA_T output_state[RNN_BATCH_SIZE * RNN_STATE_SIZE]) {
        
  // init last state inside the function, since the "first" last state is 0     
  // use these two buffers alternately as last state and output state in RNN
  FDATA_T state0[BATCH_SIZE * RNN_STATE_SIZE];                                  
  FDATA_T state1[BATCH_SIZE * RNN_STATE_SIZE];
  
  init_state(state0);
  init_state(state1);

  // store data to BRAM
  FDATA_T rnn_kernel_BRAM[RNN_INPUT_SIZE * RNN_STATE_SIZE];                     
  FDATA_T rnn_recurrent_kernel_BRAM[RNN_STATE_SIZE * RNN_STATE_SIZE];           
  FDATA_T rnn_bias_BRAM[RNN_STATE_SIZE];

  copy_rnn_bias(rnn_bias, rnn_bias_BRAM);
  copy_rnn_kernel(rnn_kernel, rnn_kernel_BRAM);
  copy_rnn_recurrent_kernel(rnn_recurrent_kernel, rnn_recurrent_kernel_BRAM);

  // go through 50 rnn layers
  for (LDATA_T i = 0; i < SAMPLE_LEN / 2; i++) {

    // input state start address
    LDATA_T addr_offset1 = 
        SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE +
        2 * i * BATCH_SIZE * RNN_INPUT_SIZE;
    LDATA_T addr_offset2 = 
        SAMPLE_LEN * BATCH_SIZE * RNN_INPUT_SIZE +
        (2 * i + 1) * BATCH_SIZE * RNN_INPUT_SIZE;
    
    // rnn should be able to initialize the output to 0
    rnn(/* last state = */state0, 
        /* input_state = */input_state + addr_offset1,
        rnn_bias_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM, 
        /* output_state = */state1);
    rnn(/* last state = */state1, 
        /* input_state = */input_state + addr_offset2, 
        rnn_bias_BRAM, rnn_kernel_BRAM, rnn_recurrent_kernel_BRAM, 
        /* output_state = */state0);
  }
  
  copy_rnn_output_state(state0, output_state);
}
