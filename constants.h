#pragma once

typedef int bool;
#define TRUE 1
#define FALSE 0

/* the arguments below are NOT independent:
 * they are defined dependently for computational efficiency
 * if you need to change them, change them TOGETHER */

/* for RNN layers */
#define BATCH_SIZE 64

#define STATE_SIZE 128
#define INPUT_SIZE 128

#define WEIGHT_DIM1 128         /* same as STATE_SIZE */ 
#define WEIGHT_DIM2 256         /* equals to STATE_SIZE + INPUT_SIZE */

/* for Fully-Connected layers */
#define FC_BATCH_SIZE 64
#define FC_INPUT_SIZE 128       /* same as STATE_SIZE */
#define FC_OUTPUT_SIZE 10000    /* say, vocabulary number */
#define FC_WEIGHT_DIM1 10000    /* same as FC_OUTPUT_SIZE */
#define FC_WEIGHT_DIM2 128      /* same as FC_INPUT_SIZE */