#ifndef TORCH_FUNCTIONS_H
#define TORCH_FUNCTIONS_H

#include <vector>
#include <memmap.h>
#include <cstdint>

std::vector<std::vector<std::vector<uint8_t>>> max_pool2d(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, int pool_size, int stride);

std::vector<std::vector<std::vector<uint8_t>>> relu(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    const uint8_t zero_point);

std::vector<std::vector<std::vector<uint8_t>>> relu(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    const uint8_t zero_point,
    MemMap &map);

std::vector<std::vector<std::vector<uint8_t>>> quantized_add_3d(
    const std::vector<std::vector<std::vector<uint8_t>>> &A,
    const std::vector<std::vector<std::vector<uint8_t>>> &B,
    float scale_a, uint8_t zp_a,
    float scale_b, uint8_t zp_b,
    float scale_out, uint8_t zp_out);

std::vector<std::vector<std::vector<float>>> dequantize_3d_vector(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    float scale,
    int32_t zero_point);

std::vector<std::vector<std::vector<float>>> pad(
    const std::vector<std::vector<std::vector<float>>> &input,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int pad_front, int pad_back, float pad_value);

std::vector<std::vector<std::vector<uint8_t>>> pad(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int pad_front = 0, int pad_back = 0, uint8_t pad_value = 0);

#endif // TORCH_FUNCTIONS_H
