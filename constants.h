#pragma once

typedef int bool;
#define TRUE 1
#define FALSE 0

/* the arguments below are NOT independent:
 * they are defined dependently for computational efficiency
 * if you need to change them, change them TOGETHER */
#define BATCH_SIZE 64

#define STATE_SIZE 128
#define INPUT_SIZE 128

#define WEIGHT_DIM1 128 /* same as STATE_SIZE */ 
#define WEIGHT_DIM2 256 /* equals to STATE_SIZE + INPUT_SIZE */
