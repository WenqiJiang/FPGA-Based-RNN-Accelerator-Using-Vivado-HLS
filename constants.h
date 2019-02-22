#pragma once

typedef int bool;
#define TRUE 1
#define FALSE 0

/* the arguments below are NOT independent:
 * they are defined dependently for computational efficiency
 * if you need to change them, change them TOGETHER */

#define STATE_LEN 128
#define INPUT_LEN 128

#define WEIGHT_DIM1 128 /* same as STATE_LEN */ 
#define WEIGHT_DIM2 256 /* equals to STATE_LEN + INPUT_LEN */
