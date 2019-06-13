// This file defines the activation functions inlcude ReLU amd tanh

#include "types.h"
#include "activation.h"
// #include "hls_math.h"

template <>
void act_relu(FDATA_T* input_feature_map, LDATA_T length) {
  // input_feature_map:   our input array / matrix / tensor, the total
  //            elements of which is 'length'
  // output_feature_map:  the output array / matrix / tensor, the result
  //            is did by doing x > 0? x : 0 elementwise
  // length:   the number of input / output FM elements
  for (LDATA_T i = 0; i < length; i++) {
    input_feature_map[i] = input_feature_map[i] > FDATA_T(0) ?
                           input_feature_map[i] : FDATA_T(0);
  }
}

template <>
void act_tanh(FDATA_T* input_feature_map, LDATA_T length) {
  // input_feature_map:   our input array / matrix / tensor, the total
  //            elements of which is 'length'
  //   output_feature_map:  the output array / matrix / tensor
  //   length:   the number of input / output FM elements
  //
  //   tanh(x) = sinh(x)/cosh(x)
  //          = ( e ^ x - e ^ (-x) ) / ( e ^ x + e ^ (-x) )
  for (LDATA_T i = 0; i < length; i++) {
    // FDATA_T e_x = exp(input_feature_map[i]);
    // FDATA_T e_minus_x = exp(input_feature_map[i]);
    // input_feature_map[i] = (e_x - e_minus_x) / (e_x + e_minus_x);
//    input_feature_map[i] = tanh<FXD_W_LENGTH, FXD_I_LENGTH>(input_feature_map[i]);

    // HACKING! seems in software part we can't call fixed point tanh
    // /home/esp2019/wj2285/esp-spring2019-wj2285/rnn_branch/hlsr/sdsoc/src/activation.cc:34:28: 
    // error: no matching function for call to 'tanh'
    //     input_feature_map[i] = tanh<FXD_W_LENGTH, FXD_I_LENGTH>(input_feature_map[i]);
    //                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // /opt/Xilinx-SDSoC/SDK/2019.1/gnu/aarch64/lin/aarch64-linux/aarch64-linux-gnu/include/c++/8.2.0/cmath:513:5: 
    // note: candidate template ignored: invalid explicitly-specified argument for template parameter '_Tp'
    input_feature_map[i] = FDATA_T(tanh(TOFLOAT(input_feature_map[i])));
  }
}

