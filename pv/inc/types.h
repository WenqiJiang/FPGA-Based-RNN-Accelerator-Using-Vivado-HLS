#pragma once

#include <cstdlib>

#define IDATA_T int
#define LDATA_T size_t

#if 1
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

// Set a default, when the variable are defined.
#ifndef FXD_W_LENGTH
#define FXD_W_LENGTH 32
#endif

#ifndef FXD_I_LENGTH
#define FXD_I_LENGTH 16
#endif

#define FDATA_T ap_fixed<FXD_W_LENGTH, FXD_I_LENGTH>

#define TOFLOAT(a) a.to_double()

#else
#include <cmath> // import exponential function: exp (val)
#define FDATA_T float
#define TOFLOAT(a) a
#endif
