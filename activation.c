#include <math.h>       /* import exponential function: exp (val) */

void relu(float* input_feature_map, float* output_feature_map, int length) {
	/* input_feature_map: 	our input array / matrix / tensor, the total 
							elements of which is 'length'
	   output_feature_map:	the output array / matrix / tensor, the result
	   						is did by doing x > 0? x : 0 elementwise
	   length: 	the number of input / output FM elements */
	for (int i = 0; i < length; i++) {
		output_feature_map[i] = input_feature_map[i] > 0? input_feature_map : 0;
	}
}


void tanh(float* input_feature_map, float* output_feature_map, int length) {
	/* input_feature_map: 	our input array / matrix / tensor, the total 
							elements of which is 'length'
	   output_feature_map:	the output array / matrix / tensor
	   length: 	the number of input / output FM elements 
		
	   tanh(x) = sinh(x)/cosh(x) 
		   	   = ( e ^ x - e ^ (-x) ) / ( e ^ x + e ^ (-x) ) */
	for (int i = 0; i < length; i++) {
		float e_x = exp(input_feature_map[i]);
		float e_minus_x = exp(input_feature_map[i]);
		output_feature_map[i] = (e_x - e_minus_x) / (e_x + e_minus_x)
	}
}

