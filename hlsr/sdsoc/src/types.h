// This file defines the datatype in this project
#pragma once

#define IDATA_T int
// HACKING! sdsoc does not support size_t?
//#define LDATA_T size_t
#define LDATA_T int

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

// #define FDATA_T ap_fixed<FXD_W_LENGTH, FXD_I_LENGTH, AP_RND, AP_SAT>
#define FDATA_T ap_fixed<FXD_W_LENGTH, FXD_I_LENGTH>

#define TOFLOAT(a) a.to_double()

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
