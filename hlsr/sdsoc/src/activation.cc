#include "types.h"
#include "activation.h"

#include <cmath> // import exponential function: exp (val)


template <>
void act_relu(FDATA_T* input_feature_map, LDATA_T length) {
    // input_feature_map:   our input array / matrix / tensor, the total
    //            elements of which is 'length'
    // output_feature_map:  the output array / matrix / tensor, the result
    //            is did by doing x > 0? x : 0 elementwise
    // length:   the number of input / output FM elements
    for (LDATA_T i = 0; i < length; i++) {
        input_feature_map[i] = input_feature_map[i] > 0? input_feature_map[i] : 0;
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
        input_feature_map[i] = tanh(input_feature_map[i]);
    }
}

