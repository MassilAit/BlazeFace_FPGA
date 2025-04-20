#ifndef BLAZE_BLOCK_H
#define BLAZE_BLOCK_H

#include <vector>
#include "conv2d.h"
#include "torch_functions.h"
#include "memmap.h"
#include <string>

class BlazeBlock
{
public:
    // Constructor
    BlazeBlock(int in_ch, int out_ch, int k_size, int strd,
               const std::vector<std::vector<std::vector<int8_t>>> &w_dw,
               const std::vector<int32_t> &b_dw,
               float dw_y_scale, float dw_w_scale, uint8_t dw_y_zero,
               const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &w_pw,
               const std::vector<int32_t> &b_pw,
               float pw_y_scale, float pw_w_scale, uint8_t pw_y_zero,
               float y_scale, uint8_t y_zero);

    // Forward pass
    std::vector<std::vector<std::vector<uint8_t>>> forward(
        const std::vector<std::vector<std::vector<uint8_t>>> &input,
        uint8_t x_zero, float x_scale, MemMap &map);

    std::string get_layer_info(const std::string &layer_name) const;

    float y_scale;
    uint8_t y_zero;

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int channel_pad;
    bool downsample;

    Conv2D depthwise_conv; // First convolution (Depthwise)
    Conv2D pointwise_conv; // Second convolution (Pointwise)
};

#endif // BLAZE_BLOCK_H
