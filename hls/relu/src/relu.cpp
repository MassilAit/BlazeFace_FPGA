#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<8> data_t;

void relu_load(data_t* in, hls::stream<data_t>& in_stream, int size) {
#pragma HLS INLINE off
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE 
        in_stream.write(in[i]);
    }
}

void relu_compute(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream, int size, data_t x_zero) {
#pragma HLS INLINE off
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE
        data_t val = in_stream.read();
        out_stream.write((val > x_zero) ? val : x_zero);
    }
}

void relu_store(hls::stream<data_t>& out_stream, data_t* out, int size) {
#pragma HLS INLINE off
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE 
        out[i] = out_stream.read();
    }
}

void relu(data_t* in, data_t* out, int size, data_t x_zero) {
#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem depth=1024
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem depth=1024
#pragma HLS INTERFACE s_axilite port=in    bundle=control
#pragma HLS INTERFACE s_axilite port=out   bundle=control
#pragma HLS INTERFACE s_axilite port=size  bundle=control
#pragma HLS INTERFACE s_axilite port=x_zero  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW

    hls::stream<data_t> in_stream("in_stream");
    hls::stream<data_t> out_stream("out_stream");

#pragma HLS STREAM variable=in_stream  depth=32
#pragma HLS STREAM variable=out_stream depth=32

    relu_load(in, in_stream, size);
    relu_compute(in_stream, out_stream, size, x_zero);
    relu_store(out_stream, out, size);
}
