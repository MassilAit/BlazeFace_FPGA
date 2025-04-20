#include "conv2d.h"
#include "accel_control.h"
#include <cmath>
#include <sstream>

// Standard Convolution Constructor
Conv2D::Conv2D(int in_ch, int out_ch, int k_size, int strd, int pad, float w_scale, float y_scale, uint8_t y_zero,
               const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &w,
               const std::vector<int32_t> &b)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(strd),
      padding(pad), w_scale(w_scale), y_scale(y_scale), y_zero(y_zero), depthwise(false), weights_4d(w), biases(b) {}

// Depthwise Convolution Constructor
Conv2D::Conv2D(int in_ch, int k_size, int strd, int pad, float w_scale, float y_scale, uint8_t y_zero,
               const std::vector<std::vector<std::vector<int8_t>>> &w,
               const std::vector<int32_t> &b)
    : in_channels(in_ch), out_channels(in_ch), kernel_size(k_size), stride(strd),
      padding(pad), w_scale(w_scale), y_scale(y_scale), y_zero(y_zero), depthwise(true), weights_3d(w), biases(b) {}

// Padding function
std::vector<std::vector<std::vector<uint8_t>>> Conv2D::apply_padding(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero)
{

    int input_size = input[0].size();
    int new_size = input_size + 2 * padding;

    std::vector<std::vector<std::vector<uint8_t>>> padded_input(in_channels,
                                                                std::vector<std::vector<uint8_t>>(new_size, std::vector<uint8_t>(new_size, static_cast<uint8_t>(x_zero))));

    for (int ic = 0; ic < in_channels; ++ic)
    {
        for (int i = 0; i < input_size; ++i)
        {
            for (int j = 0; j < input_size; ++j)
            {
                padded_input[ic][i + padding][j + padding] = input[ic][i][j];
            }
        }
    }

    return padded_input;
}

// Forward pass (quantized Conv2D)
std::vector<std::vector<std::vector<uint8_t>>> Conv2D::forward(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero, float x_scale)
{

    // Apply padding (now outputs int32_t activations with x_zero already subtracted)
    auto padded_input = apply_padding(input, x_zero);

    int input_size = padded_input[0].size();
    int output_size = (input_size - kernel_size) / stride + 1;

    // Output tensor (uint8_t activations)
    std::vector<std::vector<std::vector<uint8_t>>> output(out_channels,
                                                          std::vector<std::vector<uint8_t>>(output_size,
                                                                                            std::vector<uint8_t>(output_size, 0)));

    if (!depthwise)
    { // Standard Convolution
        for (int oc = 0; oc < out_channels; ++oc)
        {
            for (int i = 0; i < output_size; ++i)
            {
                for (int j = 0; j < output_size; ++j)
                {
                    // Start with the quantized bias (already in int32_t)
                    int32_t sum = biases[oc];

                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int ki = 0; ki < kernel_size; ++ki)
                        {
                            for (int kj = 0; kj < kernel_size; ++kj)
                            {

                                int32_t x_int = static_cast<int32_t>(padded_input[ic][i * stride + ki][j * stride + kj]) - static_cast<int32_t>(x_zero);

                                // Multiply by int8 weight and accumulate in int32
                                sum += static_cast<int32_t>(weights_4d[oc][ic][ki][kj]) * x_int;
                            }
                        }
                    }

                    // Clamp to valid uint8 range
                    output[oc][i][j] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(std::round(sum * (x_scale * w_scale) / y_scale) + y_zero))));
                }
            }
        }
    }
    else
    { // Depthwise Convolution
        for (int c = 0; c < in_channels; ++c)
        {
            for (int i = 0; i < output_size; ++i)
            {
                for (int j = 0; j < output_size; ++j)
                {
                    // Start with the quantized bias (already in int32_t)
                    int32_t sum = biases[c];

                    for (int ki = 0; ki < kernel_size; ++ki)
                    {
                        for (int kj = 0; kj < kernel_size; ++kj)
                        {

                            int32_t x_int = static_cast<int32_t>(padded_input[c][i * stride + ki][j * stride + kj]) - static_cast<int32_t>(x_zero);

                            // Multiply by int8 weight and accumulate in int32
                            sum += static_cast<int32_t>(weights_3d[c][ki][kj]) * x_int;
                        }
                    }

                    output[c][i][j] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(std::round(sum * (x_scale * w_scale) / y_scale) + y_zero))));
                }
            }
        }
    }

    return output;
}

// Forward pass (accelerated)
std::vector<std::vector<std::vector<uint8_t>>> Conv2D::forward(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, uint8_t x_zero, float x_scale, MemMap &map)
{

    // Apply padding (now outputs int32_t activations with x_zero already subtracted)
    auto padded_input = apply_padding(input, x_zero);

    int input_size = padded_input[0].size();
    int output_size = (input_size - kernel_size) / stride + 1;

    // Output tensor (uint8_t activations)
    std::vector<std::vector<std::vector<uint8_t>>> output(out_channels,
                                                          std::vector<std::vector<uint8_t>>(output_size,
                                                                                            std::vector<uint8_t>(output_size, 0)));

    if (!depthwise)
    {
        if (kernel_size == 1)
        {
            run_pw_conv_accel(map, padded_input, weights_4d, biases, output, stride, x_scale, w_scale, y_scale, x_zero, y_zero);
        }

        else
        {
            run_pw_conv_accel(map, padded_input, weights_4d, biases, output, stride, x_scale, w_scale, y_scale, x_zero, y_zero);
        }
    }
    else
    {
        run_depthwise_conv_accel(map, padded_input, weights_3d, biases, output, stride, x_scale, w_scale, y_scale, x_zero, y_zero);
    }

    return output;
}

std::string Conv2D::get_layer_info(const std::string &layer_name) const
{
    std::ostringstream oss;
    oss << "Layer: " << layer_name << "\n";
    oss << "  Type: Conv2D\n";
    oss << "  Input Channels: " << in_channels << "\n";
    oss << "  Output Channels: " << out_channels << "\n";
    oss << "  Kernel Size: " << kernel_size << "x" << kernel_size << "\n";
    oss << "  Stride: " << stride << "\n";
    oss << "  Padding: " << padding << "\n\n";
    return oss.str();
}
