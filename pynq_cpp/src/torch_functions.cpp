#include "torch_functions.h"
#include "accel_control.h"
#include <iostream>
#include <cmath>

std::vector<std::vector<std::vector<uint8_t>>> relu(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    uint8_t zero_x)
{

    std::vector<std::vector<std::vector<uint8_t>>> output = input; // Copy input

    for (auto &channel : output)
    {
        for (auto &row : channel)
        {
            for (auto &value : row)
            {
                value = std::max(static_cast<uint8_t>(zero_x), value); // Apply ReLU
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<uint8_t>>> relu(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    uint8_t zero_x, MemMap &map)
{

    std::vector<std::vector<std::vector<uint8_t>>> output = input;

    run_relu_accel(map, input, output, zero_x);

    return output;
}

std::vector<std::vector<std::vector<float>>> pad(
    const std::vector<std::vector<std::vector<float>>> &input,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int pad_front, int pad_back, float pad_value)
{

    int in_channels = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();

    // New dimensions after padding
    int out_channels = in_channels + pad_front + pad_back;
    int out_height = in_height + pad_top + pad_bottom;
    int out_width = in_width + pad_left + pad_right;

    // Initialize output tensor filled with pad_value
    std::vector<std::vector<std::vector<float>>> padded_input(
        out_channels,
        std::vector<std::vector<float>>(out_height,
                                        std::vector<float>(out_width, pad_value)));

    // Copy original input values to the correct position
    for (int c = 0; c < in_channels; ++c)
    {
        for (int i = 0; i < in_height; ++i)
        {
            for (int j = 0; j < in_width; ++j)
            {
                padded_input[c + pad_front][i + pad_top][j + pad_left] = input[c][i][j];
            }
        }
    }

    return padded_input;
}

std::vector<std::vector<std::vector<uint8_t>>> quantized_add_3d(
    const std::vector<std::vector<std::vector<uint8_t>>> &A,
    const std::vector<std::vector<std::vector<uint8_t>>> &B,
    float scale_a, uint8_t zp_a,
    float scale_b, uint8_t zp_b,
    float scale_out, uint8_t zp_out)
{
    if (A.size() != B.size())
    {
        std::cerr << "Error: Channel size mismatch!" << std::endl;
        return {};
    }

    size_t C = A.size();
    std::vector<std::vector<std::vector<uint8_t>>> OUT(C);

    for (size_t c = 0; c < C; ++c)
    {

        size_t H = B[c].size();
        OUT[c].resize(H);

        for (size_t h = 0; h < H; ++h)
        {

            size_t W = B[c][h].size();
            OUT[c][h].resize(W);

            // Je vais devoir changer ca potentiellement
            for (size_t w = 0; w < W; ++w)
            {
                float acc = (scale_a * (static_cast<int32_t>(A[c][h][w]) - zp_a) +
                             scale_b * (static_cast<int32_t>(B[c][h][w]) - zp_b)) /
                            scale_out;

                int32_t out = static_cast<int32_t>(std::round(acc)) + zp_out;
                OUT[c][h][w] = static_cast<uint8_t>(std::min(255, std::max(0, out)));
            }
        }
    }

    return OUT;
}

std::vector<std::vector<std::vector<float>>> dequantize_3d_vector(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    float scale,
    int32_t zero_point)
{
    size_t C = input.size();
    std::vector<std::vector<std::vector<float>>> output(C);

    for (size_t c = 0; c < C; ++c)
    {
        size_t H = input[c].size();
        output[c].resize(H);

        for (size_t h = 0; h < H; ++h)
        {
            size_t W = input[c][h].size();
            output[c][h].resize(W);

            for (size_t w = 0; w < W; ++w)
            {
                output[c][h][w] = scale * (static_cast<int32_t>(input[c][h][w]) - zero_point);
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<uint8_t>>> max_pool2d(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, int pool_size, int stride)
{

    int in_channels = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();

    // Calculate output size
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;

    // Initialize output tensor
    std::vector<std::vector<std::vector<uint8_t>>> output(in_channels,
                                                          std::vector<std::vector<uint8_t>>(out_height, std::vector<uint8_t>(out_width, 0)));

    // Perform max pooling
    for (int c = 0; c < in_channels; ++c)
    { // Iterate over channels
        for (int i = 0; i < out_height; ++i)
        { // Iterate over rows
            for (int j = 0; j < out_width; ++j)
            {                        // Iterate over columns
                uint8_t max_val = 0; // Very small initial value
                for (int ki = 0; ki < pool_size; ++ki)
                {
                    for (int kj = 0; kj < pool_size; ++kj)
                    {
                        int x = i * stride + ki;
                        int y = j * stride + kj;
                        if (x < in_height && y < in_width)
                        { // Ensure within bounds
                            max_val = std::max(max_val, input[c][x][y]);
                        }
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }
    return output;
}

std::vector<std::vector<std::vector<uint8_t>>> pad(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int pad_front, int pad_back, uint8_t pad_value)
{

    int in_channels = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();

    // New dimensions after padding
    int out_channels = in_channels + pad_front + pad_back;
    int out_height = in_height + pad_top + pad_bottom;
    int out_width = in_width + pad_left + pad_right;

    // Initialize output tensor filled with pad_value
    std::vector<std::vector<std::vector<uint8_t>>> padded_input(
        out_channels,
        std::vector<std::vector<uint8_t>>(out_height,
                                          std::vector<uint8_t>(out_width, pad_value)));

    // Copy original input values to the correct position
    for (int c = 0; c < in_channels; ++c)
    {
        for (int i = 0; i < in_height; ++i)
        {
            for (int j = 0; j < in_width; ++j)
            {
                padded_input[c + pad_front][i + pad_top][j + pad_left] = input[c][i][j];
            }
        }
    }

    return padded_input;
}
