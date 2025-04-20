#include <ap_int.h>
#include <hls_stream.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cstdint>

#define KERNEL_SIZE 3

typedef ap_uint<8>  U8;
typedef ap_int <8>  S8;
typedef ap_int<32>  S32;




// ------------------------ Software Reference ------------------------

template <typename T>
T clamp(T val, T min_val, T max_val) {
    return std::max(min_val, std::min(val, max_val));
}

void depthwise_conv_sw(const std::vector<std::vector<std::vector<uint8_t>>>& input,
                       const std::vector<std::vector<std::vector<int8_t>>>& weights,
                       const std::vector<int32_t>& biases,
                       std::vector<std::vector<std::vector<uint8_t>>>& output,
                       int stride,
                       uint8_t x_zero, uint8_t y_zero,
                       float x_scale, float w_scale, float y_scale) {
    int Cin = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int OH = (H - KERNEL_SIZE) / stride + 1;

    for (int c = 0; c < Cin; ++c) {
        for (int i = 0; i < OH; ++i) {
            for (int j = 0; j < OH; ++j) {
                int32_t sum = biases[c];
                for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                    for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                        int ih = i * stride + ki;
                        int iw = j * stride + kj;
                        int32_t x_int = static_cast<int32_t>(input[c][ih][iw]) - static_cast<int32_t>(x_zero);
                        int32_t w_int = static_cast<int32_t>(weights[c][ki][kj]);
                        sum += w_int * x_int;
                    }
                }
            output[c][i][j] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(std::round(sum * (x_scale * w_scale) / y_scale) + y_zero))));
            }
        }
    }


}

// ------------------------ HLS Core API ------------------------

void depthwise_conv(const U8* in,
                    const S8* weights,
                    const S32* biases,
                    U8* out,
                    int size,
                    int Cin,
                    int stride,
                    U8 x_zero,
                    U8 y_zero,
                    S32 M,
                    S32 shift);

void depthwise_conv_stream(const U8* in,
                    const S8* weights,
                    const S32* biases,
                    U8* out,
                    int size,
                    int Cin,
                    int stride,
                    U8 x_zero,
                    U8 y_zero,
                    S32 M,
                    S32 shift);

// ------------------------ HLS Test Wrapper ------------------------

void depthwise_conv_wrapper(const std::vector<U8>& input,
                            const std::vector<S8>& weights,
                            const std::vector<int32_t>& biases,
                            std::vector<U8>& output,
                            int size, int Cin, int stride,
                            U8 x_zero, U8 y_zero, S32 M, S32 shift) {
    depthwise_conv(input.data(), weights.data(), (const S32*)biases.data(), output.data(),
                   size, Cin, stride, x_zero, y_zero, M, shift);
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



// ------------------------ Testbench ------------------------

int main() {
    constexpr int Cin = 2;
    constexpr int size = 10;
    constexpr int stride = 1;
    constexpr int OH = (size - KERNEL_SIZE) / stride + 1;

    std::vector<std::vector<std::vector<uint8_t>>> input(Cin, std::vector<std::vector<uint8_t>>(size, std::vector<uint8_t>(size)));
    std::vector<std::vector<std::vector<int8_t>>> weights(Cin, std::vector<std::vector<int8_t>>(KERNEL_SIZE, std::vector<int8_t>(KERNEL_SIZE)));
    std::vector<int32_t> biases(Cin);
    std::vector<std::vector<std::vector<uint8_t>>> ref_output(Cin, std::vector<std::vector<uint8_t>>(OH, std::vector<uint8_t>(OH)));

    float x_scale = 0.12f;
    float w_scale = 0.098f;
    float y_scale = 0.076f;
    uint8_t x_zero = 22;
    uint8_t y_zero = 18;
    
    double real_multiplier = (x_scale * w_scale) / y_scale;

    int32_t multiplier;
    int32_t shift;

    QuantizeMultiplier(real_multiplier, multiplier, shift);


    for (int c = 0; c < Cin; ++c) {
        biases[c] = 0;
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                input[c][i][j] = (i + j + c) % 256;

        for (int i = 0; i < KERNEL_SIZE; ++i)
            for (int j = 0; j < KERNEL_SIZE; ++j)
                weights[c][i][j] = (i - j + c);
    }

    depthwise_conv_sw(input, weights, biases, ref_output, stride, x_zero, y_zero, x_scale, w_scale, y_scale);

    std::vector<U8> flat_input(Cin * size * size);
    std::vector<S8> flat_weights(Cin * KERNEL_SIZE * KERNEL_SIZE);
    std::vector<U8> flat_output(Cin * OH * OH);

    for (int c = 0; c < Cin; ++c) {
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                flat_input[c * size * size + i * size + j] = input[c][i][j];

        for (int i = 0; i < KERNEL_SIZE; ++i)
            for (int j = 0; j < KERNEL_SIZE; ++j)
                flat_weights[c * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j] = weights[c][i][j];
    }

    depthwise_conv_wrapper(flat_input, flat_weights, biases, flat_output,
                           size, Cin, stride, x_zero, y_zero, multiplier, shift);

    bool success = true;
    int total = 0;
    int failed =0;
    int max = 0;
    for (int c = 0; c < Cin; ++c) {
        for (int i = 0; i < OH; ++i) {
            for (int j = 0; j < OH; ++j) {
                uint8_t ref = ref_output[c][i][j];
                uint8_t hls = flat_output[c * OH * OH + i * OH + j];
                total++;
                if (ref != hls) {
                    std::cout << "Mismatch at c=" << c << " i=" << i << " j=" << j
                             << ": ref=" << (int)ref << ", hls=" << (int)hls << std::endl;
                    if (std::abs(ref-hls)>max) {
                        max = std::abs(ref-hls);
                    }
                    success = false;
                    failed++;
                }
            }
        }
    }

    if (success) std::cout << "Test passed!" << std::endl;
    else std::cout << "Test FAILED.("<<failed<<"/"<<total<<") Max : "<<max<< std::endl;

    return 0;
}
