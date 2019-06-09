// This file defines the datatype in this project
#pragma once

#include "ap_int.h"
#include "ap_fixed.h"

#define IDATA_T int
#define LDATA_T size_t

#if 1
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

// In this project, use the following fixed point setting
#define FXD_W_LENGTH 16
#define FXD_I_LENGTH 7

// Set a default, when the variable are defined.
#ifndef FXD_W_LENGTH
#define FXD_W_LENGTH 32
#endif

#ifndef FXD_I_LENGTH
#define FXD_I_LENGTH 16
#endif

#define FDATA_T ap_fixed<FXD_W_LENGTH, FXD_I_LENGTH, AP_RND, AP_SAT>

#define TOFLOAT(a) a.to_double()

#else
#include <cmath> // import exponential function: exp (val)
#define FDATA_T float
#define TOFLOAT(a) a
#endif