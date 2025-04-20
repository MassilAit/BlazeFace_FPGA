#ifndef ACCEL_CONTROL
#define ACCEL_CONTROL

#include "memmap.h"
#include <vector>
#include <cstdint>

void run_relu_accel(const MemMap &mem,
                    const std::vector<std::vector<std::vector<uint8_t>>> &input,
                    std::vector<std::vector<std::vector<uint8_t>>> &output,
                    uint8_t x_zero);

void run_pw_conv_accel(const MemMap &mem,
                       const std::vector<std::vector<std::vector<uint8_t>>> &input,
                       const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &weights,
                       const std::vector<int32_t> &biases,
                       std::vector<std::vector<std::vector<uint8_t>>> &output,
                       int stride,
                       float x_scale,
                       float w_scale,
                       float y_scale,
                       uint8_t x_zero,
                       uint8_t y_zero);

void run_depthwise_conv_accel(const MemMap &mem,
                              const std::vector<std::vector<std::vector<uint8_t>>> &input,
                              const std::vector<std::vector<std::vector<int8_t>>> &weights,
                              const std::vector<int32_t> &biases,
                              std::vector<std::vector<std::vector<uint8_t>>> &output,
                              int stride,
                              float x_scale,
                              float w_scale,
                              float y_scale,
                              uint8_t x_zero,
                              uint8_t y_zero);

#endif