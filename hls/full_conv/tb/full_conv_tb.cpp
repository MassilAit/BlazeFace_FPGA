#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <ap_int.h> // Required for U8, S8, S32 typedefs using ap_int

typedef ap_uint<8>  U8;
typedef ap_int <8>  S8;
typedef ap_int<32>  S32;

#define KERNEL_SIZE 5


void full_conv_sw(const std::vector<std::vector<std::vector<uint8_t>>>& input,
                       const std::vector<std::vector<std::vector<std::vector<int8_t>>>>& weights,
                       const std::vector<int32_t>& biases,
                       std::vector<std::vector<std::vector<uint8_t>>>& output,
                       int stride,
                       uint8_t x_zero, uint8_t y_zero,
                       float x_scale, float w_scale, float y_scale) {

        int Cin = input.size();
        int H = input[0].size();
        int W = input[0][0].size();
        int Cout = weights.size();
        int OH = (H - KERNEL_SIZE) / stride + 1;

            // ← Allocate output before writing into it
    output.assign(
      Cout,
      std::vector<std::vector<uint8_t>>(
        OH,
        std::vector<uint8_t>(OH)
      )
    );



            for (int oc = 0; oc < Cout; ++oc) {
            for (int i = 0; i < OH; ++i) {
                for (int j = 0; j < OH; ++j) {
                    // Start with the quantized bias (already in int32_t)
                    int32_t sum = biases[oc];


                    for (int ic = 0; ic < Cin; ++ic) {
                        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                               
       

                                int32_t x_int = static_cast<int32_t>(input[ic][i * stride + ki][j * stride + kj]) - static_cast<int32_t>(x_zero);

                                // Multiply by int8 weight and accumulate in int32
                                sum += static_cast<int32_t>(weights[oc][ic][ki][kj]) * x_int;


                            }
                        }
                    }



            
                    // Clamp to valid uint8 range
                    output[oc][i][j] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(std::round(sum * (x_scale * w_scale) / y_scale) + y_zero))));

                }
            }
        }

}



void full_conv_streamed(const U8* input,
                    const S8* weights,
                    const S32* biases,
                    U8* output,
                    int input_size,
                    int output_size,
                    int Cin,
                    int Cout,
                    int stride,
                    U8 x_zero,
                    U8 y_zero,
                    S32 M,
                    S32 shift);

void full_conv_wrapper(const std::vector<U8>& input,
                            const std::vector<S8>& weights,
                            const std::vector<int32_t>& biases,
                            std::vector<U8>& output,
                            int input_size, int Cin, int Cout, int stride,
                            U8 x_zero, U8 y_zero, S32 M, S32 shift) {

    int output_size = (input_size - 1) / stride + 1;
    full_conv_streamed(input.data(), weights.data(), (const S32*)biases.data(), output.data(),
                   input_size, output_size, Cin, Cout, stride, x_zero, y_zero, M, shift);
}

void QuantizeMultiplier(double real_multiplier, int32_t& multiplier, int32_t& shift) {
    if (real_multiplier == 0.0) {
        multiplier = 0;
        shift = 0;
        return;
    }

    int exp;
    double q = std::frexp(real_multiplier, &exp); // q in [0.5, 1)
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));

    multiplier = static_cast<int32_t>(q_fixed);
    shift = exp;
}

int main() {
    constexpr int Cin = 3;
    constexpr int Cout = 2;
    constexpr int size = 10;
    constexpr int stride = 1;
    constexpr int OH = (size - KERNEL_SIZE) / stride + 1;

    std::vector<std::vector<std::vector<uint8_t>>> input(Cin, std::vector<std::vector<uint8_t>>(size, std::vector<uint8_t>(size)));
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> weights(Cout, std::vector<std::vector<std::vector<int8_t>>>(Cin, std::vector<std::vector<int8_t>>(KERNEL_SIZE, std::vector<int8_t>(KERNEL_SIZE))));
    std::vector<int32_t> biases(Cout);
    std::vector<std::vector<std::vector<uint8_t>>> ref_output(Cout, std::vector<std::vector<uint8_t>>(OH, std::vector<uint8_t>(OH)));

    float x_scale = 0.11f;
    float w_scale = 0.09f;
    float y_scale = 0.07f;
    uint8_t x_zero = 17;
    uint8_t y_zero = 23;

    double real_multiplier = (x_scale * w_scale) / y_scale;
    int32_t multiplier, shift;
    QuantizeMultiplier(real_multiplier, multiplier, shift);

    // Fill in test data
    for (int c = 0; c < Cin; ++c)
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                input[c][i][j] = (c + i + j) % 256;

    for (int oc = 0; oc < Cout; ++oc) {
        biases[oc] = 1;
        for (int ic = 0; ic < Cin; ++ic)
            for (int ki = 0; ki <KERNEL_SIZE ; ++ki)
                for (int kj = 0; kj <KERNEL_SIZE ; ++kj)

                    weights[oc][ic][ki][kj] = 1 ;//(ic - oc + 3) % 5 - 2; // -2 to 2
        
    }

    // Run reference
    full_conv_sw(input, weights, biases, ref_output, stride, x_zero, y_zero, x_scale, w_scale, y_scale);

    // Flatten inputs for HLS version
    std::vector<U8> flat_input(Cin * size * size);
    std::vector<S8> flat_weights(Cout * Cin * KERNEL_SIZE * KERNEL_SIZE);
    std::vector<U8> flat_output(Cout * OH * OH);

    for (int ic = 0; ic < Cin; ++ic)
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                flat_input[ic * size * size + i * size + j] = input[ic][i][j];


    int K = KERNEL_SIZE;
    for (int oc = 0; oc < Cout; ++oc) {
        assert((int)weights[oc].size() == Cin);
        for (int ic = 0; ic < Cin; ++ic) {
            assert((int)weights[oc][ic].size() == K);
            for (int ki = 0; ki < K; ++ki) {
                assert((int)weights[oc][ic][ki].size() == K);
                for (int kj = 0; kj < K; ++kj) {
                    size_t idx = 
                        (size_t)oc * (Cin * K * K) +
                        (size_t)ic * (K   * K)     +
                        (size_t)ki *  K           +
                        (size_t)kj;
                    flat_weights[idx] = weights[oc][ic][ki][kj];
                }
            }
        }
    }

    // Run HLS
    full_conv_wrapper(flat_input, flat_weights, biases, flat_output,
                           size, Cin, Cout, stride, x_zero, y_zero, multiplier, shift);
    // Compare
    bool success = true;
    int max_diff = 0, errors = 0;

    for (int oc = 0; oc < Cout; ++oc) {
        for (int i = 0; i < OH; ++i) {
            for (int j = 0; j < OH; ++j) {
                uint8_t ref = ref_output[oc][i][j];
                uint8_t hls = flat_output[oc * OH * OH + i * OH + j];
                if (ref != hls) {
                    //std::cout << "Mismatch at oc=" << oc << " i=" << i << " j=" << j
                    //          << ": ref=" << (int)ref << " hls=" << (int)hls << "\n";
                    max_diff = std::max(max_diff, std::abs(ref - hls));
                    success = false;
                    ++errors;
                }
            }
        }
    }

    if (success) std::cout << "Test passed!\n";
    else std::cout << "Test FAILED.("<<errors<<"/"<<(Cout*OH*OH)<<") Max : "<<max_diff<< std::endl;

    return 0;
}
