#include "blaze_block.h"
#include <sstream>

// Constructor
BlazeBlock::BlazeBlock(int in_ch, int out_ch, int k_size, int strd,
                       const std::vector<std::vector<std::vector<int8_t>>> &w_dw,
                       const std::vector<int32_t> &b_dw,
                       float dw_y_scale, float dw_w_scale, uint8_t dw_y_zero,
                       const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &w_pw,
                       const std::vector<int32_t> &b_pw,
                       float pw_y_scale, float pw_w_scale, uint8_t pw_y_zero,
                       float y_scale, uint8_t y_zero)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(strd),
      channel_pad(out_ch - in_ch), downsample(strd == 2),
      y_scale(y_scale), y_zero(y_zero),
      depthwise_conv(in_ch, k_size, strd, (strd == 1 ? (k_size - 1) / 2 : 0), dw_w_scale, dw_y_scale, dw_y_zero, w_dw, b_dw),
      pointwise_conv(in_ch, out_ch, 1, 1, 0, pw_w_scale, pw_y_scale, pw_y_zero, w_pw, b_pw) {}

// Forward pass
std::vector<std::vector<std::vector<uint8_t>>> BlazeBlock::forward(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    uint8_t x_zero, float x_scale,
    MemMap &map)
{

    std::vector<std::vector<std::vector<uint8_t>>> h = input;
    std::vector<std::vector<std::vector<uint8_t>>> x = input; // Residual Connection

    // If stride == 2, apply manual padding & max pooling
    if (downsample)
    {
        h = pad(h, 0, 2, 0, 2, 0, 0, x_zero); // Padding before pooling
        x = max_pool2d(h, 2, 2);              // Downsampling
    }

    // Depthwise Convolution (with correct padding)
    h = depthwise_conv.forward(h, x_zero, x_scale, map);

    // Pointwise Convolution
    h = pointwise_conv.forward(h, depthwise_conv.y_zero, depthwise_conv.y_scale, map);

    // If `out_channels > in_channels`, add extra zero channels to x
    if (channel_pad > 0)
    {
        x = pad(x, 0, 0, 0, 0, 0, channel_pad, x_zero);
    }

    h = quantized_add_3d(x, h, x_scale, x_zero, pointwise_conv.y_scale, pointwise_conv.y_zero, y_scale, y_zero);

    return relu(h, y_zero, map);
}

std::string BlazeBlock::get_layer_info(const std::string &layer_name) const
{
    std::ostringstream oss;
    oss << "Layer: " << layer_name << "\n";
    oss << "  Type: BlazeBlock\n";
    oss << "  Input Channels: " << in_channels << "\n";
    oss << "  Output Channels: " << out_channels << "\n";
    oss << "  Kernel Size: " << kernel_size << "x" << kernel_size << "\n";
    oss << "  Stride: " << stride << "\n";
    return oss.str();
}
