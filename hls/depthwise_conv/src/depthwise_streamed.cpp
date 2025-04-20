#include <ap_int.h>
#include <hls_stream.h>

#define KERNEL_SIZE 3

using  U8 = ap_uint<8>;
using  S8 = ap_int <8>;
using S32 = ap_int<32>;

// ---------------------------------------------------------------------
// helper structures and constants
// ---------------------------------------------------------------------
struct Tile { U8 data[KERNEL_SIZE*KERNEL_SIZE]; };

// ---------------------------------------------------------------------
// ❶  Control‑token producer  – one token per input channel
// ---------------------------------------------------------------------
void channel_feeder(const S32          *biases,
                    int                 size,
                    int                 Cin,
                    int                 OH,
                    hls::stream<int>   &idx_stream,     // channel index   (c)
                    hls::stream<int>   &in_off_stream,  // input  offset   (c*size*size)
                    hls::stream<S32>   &bias_stream,    // bias
                    hls::stream<int>   &out_off_stream) // output offset   (c*OH*OH)
{
#pragma HLS PIPELINE II=1
    for (int c = 0; c < Cin; ++c) {
        idx_stream     .write(c);
        in_off_stream  .write(c * size * size);
        bias_stream    .write(biases[c]);
        out_off_stream .write(c * OH  * OH );
    }
}

// ---------------------------------------------------------------------
// ❷  Weight loader   (reads channel index  → streams 9 weights)
// ---------------------------------------------------------------------
void load_weights(const S8            *weights,
                  hls::stream<int>   &idx_stream,
                  hls::stream<S8>    &w_stream)
{
#pragma HLS PIPELINE II=1
    while (!idx_stream.empty()) {
        int c = idx_stream.read();
        for (int i = 0; i < KERNEL_SIZE*KERNEL_SIZE; ++i) {
#pragma HLS UNROLL
            w_stream.write(weights[c*9 + i]);
        }
    }
}

// ---------------------------------------------------------------------
// ❸  Tile loader  (reads input offset  → streams OH×OH tiles)
// ---------------------------------------------------------------------
void load_input_tile(const U8           *in,
                     hls::stream<int>  &in_off_stream,
                     hls::stream<Tile> &tile_stream,
                     int                size,
                     int                stride)
{
#pragma HLS PIPELINE II=1
    const int OH = (size - KERNEL_SIZE) / stride + 1;

    while (!in_off_stream.empty()) {
        int offset = in_off_stream.read();
        const U8 *in_c = in + offset;

        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OH; ++ow) {
                #pragma HLS PIPELINE off
                Tile t;
                for (int kh = 0; kh < KERNEL_SIZE; ++kh)
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw)
                    #pragma HLS PIPELINE                    
                        t.data[kh*KERNEL_SIZE+kw] =
                             in_c[(oh*stride+kh)*size + (ow*stride+kw)];
                tile_stream.write(t);
            }
    }
}

// ---------------------------------------------------------------------
// ❹  Convolution & quantisation (reads bias  → produces OH×OH outputs)
// ---------------------------------------------------------------------
void compute_output(hls::stream<Tile> &tile_stream,
                    hls::stream<S8>   &w_stream,
                    hls::stream<S32>  &bias_stream,
                    hls::stream<U8>   &out_stream,
                    U8                 x_zero,
                    U8                 y_zero,
                    S32                M,
                    S32                shift,
                    int                OH)
{
#pragma HLS PIPELINE II=1
    S8 w_buf[9];
#pragma HLS ARRAY_PARTITION variable=w_buf complete

    while (!bias_stream.empty()) {
        // fetch weights for this channel
        for (int i = 0; i < 9; ++i) w_buf[i] = w_stream.read();

        S32 bias = bias_stream.read();

        for (int i = 0; i < OH*OH; ++i) {
#pragma HLS PIPELINE
            Tile t = tile_stream.read();
            S32 acc = bias;
            for (int j = 0; j < 9; ++j) {
#pragma HLS UNROLL
                acc += (S32)w_buf[j] * ((S32)t.data[j] - (S32)x_zero);
            }
            S32 y = ((acc * M) >> (31 - shift)) + y_zero;
            if (y < 0)   y = 0;
            if (y > 255) y = 255;
            out_stream.write((U8)y);
        }
    }
}

// ---------------------------------------------------------------------
// ❺  Output writer  (reads output offset  → stores OH×OH results)
// ---------------------------------------------------------------------
void store_output(U8                  *out,
                  hls::stream<int>    &out_off_stream,
                  hls::stream<U8>     &out_stream,
                  int                  OH)
{
    while (!out_off_stream.empty()) {
        int offset = out_off_stream.read();
        U8 *out_c = out + offset;
        for (int i = 0; i < OH*OH; ++i) {
#pragma HLS PIPELINE
            out_c[i] = out_stream.read();
        }
    }
}

// ---------------------------------------------------------------------
// ❻  Top function
// ---------------------------------------------------------------------
void depthwise_conv_stream(const U8   *in,
                           const S8   *weights,
                           const S32  *biases,
                           U8         *out,
                           int         size,
                           int         Cin,
                           int         stride,
                           U8          x_zero,
                           U8          y_zero,
                           S32         M,
                           S32         shift)
{
#pragma HLS INTERFACE m_axi port=in      offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth=8192
#pragma HLS INTERFACE m_axi port=biases  offset=slave bundle=gmem2 depth=2048
#pragma HLS INTERFACE m_axi port=out     offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE s_axilite port=in      bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=biases  bundle=control
#pragma HLS INTERFACE s_axilite port=out     bundle=control
#pragma HLS INTERFACE s_axilite port=size    bundle=control
#pragma HLS INTERFACE s_axilite port=Cin     bundle=control
#pragma HLS INTERFACE s_axilite port=stride  bundle=control
#pragma HLS INTERFACE s_axilite port=x_zero  bundle=control
#pragma HLS INTERFACE s_axilite port=y_zero  bundle=control
#pragma HLS INTERFACE s_axilite port=M       bundle=control
#pragma HLS INTERFACE s_axilite port=shift   bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

#pragma HLS DATAFLOW

    //-------------------------------------------
    // Streams
    //-------------------------------------------
    hls::stream<int>   idx_stream("idx");
    hls::stream<int>   in_off_stream("in_off");
    hls::stream<S32>   bias_stream("bias");
    hls::stream<int>   out_off_stream("out_off");
    hls::stream<S8>    w_stream("w");
    hls::stream<Tile>  tile_stream("tile");
    hls::stream<U8>    out_stream("out");

#pragma HLS STREAM variable=idx_stream      depth=64
#pragma HLS STREAM variable=in_off_stream   depth=64
#pragma HLS STREAM variable=bias_stream     depth=64
#pragma HLS STREAM variable=out_off_stream  depth=64
#pragma HLS STREAM variable=w_stream        depth=64
#pragma HLS STREAM variable=tile_stream     depth=64
#pragma HLS STREAM variable=out_stream      depth=64

    const int OH = (size - KERNEL_SIZE) / stride + 1;

    //-------------------------------------------
    // Launch data‑flow processes
    //-------------------------------------------
    channel_feeder   (biases, size, Cin, OH,
                      idx_stream, in_off_stream, bias_stream, out_off_stream);

    load_weights     (weights,   idx_stream,    w_stream);
    load_input_tile  (in,        in_off_stream, tile_stream, size, stride);
    compute_output   (tile_stream, w_stream, bias_stream, out_stream,
                      x_zero, y_zero, M, shift, OH);
    store_output     (out, out_off_stream, out_stream, OH);
}
