#include <ap_int.h>
#include <hls_stream.h>

#define MAX_CIN  96
#define MAX_INPUT_SIZE 128

using U8 = ap_uint<8>;
using S8 = ap_int<8>;
using S32 = ap_int<32>;


void channel_feeder(const S32* biases,
                    int Cout,
                    hls::stream<int> &channel_weight_stream,
                    hls::stream<S32>& channel_accumulate_stream,
                    hls::stream<int>& channel_output_stream) 
{
#pragma HLS INLINE off

    for (int oc = 0; oc < Cout; ++oc) {
    #pragma HLS PIPELINE
        channel_weight_stream.write(oc);
        channel_output_stream.write(oc);
        channel_accumulate_stream.write(biases[oc]);
    }

}

void load_weights(const S8 *w_ptr,
                  int Cin,
                  hls::stream<int> &channel_weight_stream,
                  hls::stream<S8> &w_stream)
{
#pragma HLS INLINE off

    while (!channel_weight_stream.empty()) {
        int weight_idx = channel_weight_stream.read() * Cin;
        
        for (int ic = 0; ic < Cin; ++ic) {
        #pragma HLS PIPELINE 
            w_stream.write(w_ptr[weight_idx + ic]);
        }
    }
}


void accumulate_pixels(const U8 *input,
                       hls::stream<S8> & w_stream,
                       hls::stream<S32>& pixel_stream,
                       hls::stream<S32>& channel_accumulate_stream,
                       int input_size,    // == padded_input[0].size()
                       int output_size,
                       int Cin,
                       int stride,
                       U8  x_zero) {
#pragma HLS INLINE off


while (!channel_accumulate_stream.empty()) {
        
        // Load weights Locally
        S8 w_buf[MAX_CIN];
        #pragma HLS ARRAY_PARTITION variable=w_buf complete

        for (int ic = 0; ic < Cin; ++ic) {
        #pragma HLS PIPELINE
            w_buf[ic] = w_stream.read();
        }

        static S32 p_buffer[MAX_INPUT_SIZE][MAX_INPUT_SIZE];
        #pragma HLS ARRAY_PARTITION variable=p_buffer cyclic factor=8 dim=2

        S32 bias = channel_accumulate_stream.read();

        for (int ic = 0; ic < Cin; ++ic) {

            const U8* base = input + ic * input_size * input_size;

            S8 w = w_buf[ic];

            for (int i = 0; i < output_size; ++i) {

                int input_row = i * stride;
                const U8 * row = base + input_row*input_size;

                for (int j = 0, col = 0; j < output_size; ++j, col += stride) {
                #pragma HLS PIPELINE

                    S32 acc = (ic == 0) ? bias : p_buffer[i][j];

                    acc += (S32)w * ((S32)row[col] - x_zero);

                    p_buffer[i][j] = acc; 


                    if (ic == Cin - 1) {
                        pixel_stream.write(acc);
                    }
                }

            }


        }
    }

}


void write_output(U8 *out_ptr,
                  hls::stream<S32>& pixel_stream,
                  int output_size,
                  hls::stream<int>& channel_output_stream,
                  U8  y_zero,
                  S32 M, S32 shift)
{
#pragma HLS INLINE off

    while (!channel_output_stream.empty()) {

        int oc = channel_output_stream.read();
        
        for (int i = 0; i < output_size; ++i) {
            const int out_idx = oc * output_size * output_size + i*output_size;
            for (int j = 0; j < output_size; ++j) {
            #pragma HLS PIPELINE

                S32 acc = pixel_stream.read();
                S32 y   = ((acc * M) >> (31 - shift)) + y_zero;
                if (y < 0)   y = 0;
                if (y > 255) y = 255;
                out_ptr[out_idx + j] = (U8)y;
            }
        }
    }
}


void pointwise_conv_streamed(const U8* input,         
                        const S8* weights,       
                        const S32* biases,       
                        U8* output,              
                        int input_size, int output_size,        
                        int Cin, int Cout,
                        int stride,
                        U8 x_zero,
                        U8 y_zero,
                        S32 M,
                        S32 shift)
{
#pragma HLS INTERFACE m_axi port=input   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=biases  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=output  offset=slave bundle=gmem0

#pragma HLS INTERFACE s_axilite port=input        bundle=control
#pragma HLS INTERFACE s_axilite port=weights      bundle=control
#pragma HLS INTERFACE s_axilite port=biases       bundle=control
#pragma HLS INTERFACE s_axilite port=output       bundle=control
#pragma HLS INTERFACE s_axilite port=input_size   bundle=control
#pragma HLS INTERFACE s_axilite port=output_size  bundle=control
#pragma HLS INTERFACE s_axilite port=Cin          bundle=control
#pragma HLS INTERFACE s_axilite port=Cout         bundle=control
#pragma HLS INTERFACE s_axilite port=stride         bundle=control
#pragma HLS INTERFACE s_axilite port=x_zero       bundle=control
#pragma HLS INTERFACE s_axilite port=y_zero       bundle=control
#pragma HLS INTERFACE s_axilite port=M            bundle=control
#pragma HLS INTERFACE s_axilite port=shift        bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control


    #pragma HLS DATAFLOW 

    hls::stream<S8>  w_stream("w_stream");
    #pragma HLS STREAM variable=w_stream depth=MAX_CIN

    hls::stream<S32> pixel_stream("pixel_stream");
    #pragma HLS STREAM variable=pixel_stream depth=MAX_INPUT_SIZE    
    hls::stream<int> channel_weight_stream("channel_weight_stream");
    #pragma HLS STREAM variable=channel_weight_stream depth=MAX_CIN  
    hls::stream<S32> channel_accumulate_stream("channel_accumulate_stream");
    #pragma HLS STREAM variable=channel_accumulate_stream depth=MAX_CIN

    hls::stream<int> channel_output_stream("channel_output_stream");
    #pragma HLS STREAM variable=channel_output_stream depth=MAX_CIN
    
    channel_feeder(biases, Cout, channel_weight_stream, channel_accumulate_stream, channel_output_stream);
    load_weights(weights, Cin, channel_weight_stream, w_stream);
    accumulate_pixels(input, w_stream, pixel_stream, channel_accumulate_stream, input_size, output_size, Cin, stride, x_zero);
    write_output(output, pixel_stream, output_size, channel_output_stream, y_zero, M, shift);
  

}