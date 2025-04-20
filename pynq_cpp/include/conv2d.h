#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>
#include <cstdint>
#include <memmap.h>

class Conv2D
{
public:
    // Normal 2d convolution
    Conv2D(int in_ch, int out_ch, int k_size, int strd, int pad, float w_scale, float y_scale, uint8_t y_zero,
           const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &w,
           const std::vector<int32_t> &b);

    // Depthwise Convolution
    Conv2D(int in_ch, int k_size, int strd, int pad, float w_scale, float y_scale, uint8_t y_zero,
           const std::vector<std::vector<std::vector<int8_t>>> &w,
           const std::vector<int32_t> &b);

    std::vector<std::vector<std::vector<uint8_t>>> forward(
        const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero, float x_scale);

    std::vector<std::vector<std::vector<uint8_t>>> forward(
        const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero, float x_scale, MemMap &map);

    std::string get_layer_info(const std::string &layer_name) const;

    float y_scale;
    uint8_t y_zero;

private:
    int in_channels, out_channels, kernel_size, stride, padding;
    bool depthwise;
    float w_scale;

    std::vector<std::vector<std::vector<std::vector<int8_t>>>> weights_4d;
    std::vector<std::vector<std::vector<int8_t>>> weights_3d;
    std::vector<int32_t> biases;

    std::vector<std::vector<std::vector<uint8_t>>> apply_padding(
        const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero);
};

#endif // CONV2D_H
