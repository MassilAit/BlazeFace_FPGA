#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<uint8_t> load_uint8_vector_from_file(const std::string &filename)
{
    std::vector<uint8_t> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }
    int value;
    while (file >> value)
    {
        data.push_back(static_cast<uint8_t>(value));
    }
    file.close();
    return data;
}

std::vector<std::vector<std::vector<uint8_t>>> load_input_uint8(const std::string &filename, int in_channels, int input_size)
{
    std::vector<uint8_t> flat_input = load_uint8_vector_from_file(filename);
    std::vector<std::vector<std::vector<uint8_t>>> input(in_channels, std::vector<std::vector<uint8_t>>(input_size, std::vector<uint8_t>(input_size)));

    int idx = 0;
    for (int ic = 0; ic < in_channels; ++ic)
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < input_size; ++j)
                input[ic][i][j] = flat_input[idx++];

    return input;
}

std::vector<float> load_vector_from_file(const std::string &filename)
{
    std::vector<float> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }
    float value;
    while (file >> value)
    {
        data.push_back(value);
    }
    file.close();
    return data;
}

std::vector<std::vector<std::vector<float>>> load_input(const std::string &filename, int in_channels, int input_size)
{
    std::vector<float> flat_input = load_vector_from_file(filename);
    std::vector<std::vector<std::vector<float>>> input(in_channels, std::vector<std::vector<float>>(input_size, std::vector<float>(input_size)));

    int idx = 0;
    for (int ic = 0; ic < in_channels; ++ic)
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < input_size; ++j)
                input[ic][i][j] = flat_input[idx++];

    return input;
}

// Function to reshape a flat vector into a 2D vector
std::vector<std::vector<float>> load_output(const std::string &filename, int num_rows, int num_cols)
{
    std::vector<float> flat_output = load_vector_from_file(filename);

    // Validate file size
    if (flat_output.size() != num_rows * num_cols)
    {
        std::cerr << "Error: File size (" << flat_output.size()
                  << ") does not match expected dimensions ("
                  << num_rows << "x" << num_cols << ")!" << std::endl;
        return {};
    }

    // Reshape into 2D vector
    std::vector<std::vector<float>> output(num_rows, std::vector<float>(num_cols));

    int idx = 0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = 0; j < num_cols; ++j)
        {
            output[i][j] = flat_output[idx++];
        }
    }

    return output;
}

// Function to get shape of a 3D vector
std::vector<size_t> get_shape(const std::vector<std::vector<std::vector<float>>> &vec)
{
    size_t dim1 = vec.size();
    size_t dim2 = dim1 > 0 ? vec[0].size() : 0;
    size_t dim3 = (dim2 > 0 && dim1 > 0) ? vec[0][0].size() : 0;

    return {dim1, dim2, dim3};
}

std::vector<std::vector<std::vector<uint8_t>>> permute(
    const std::vector<std::vector<std::vector<uint8_t>>> &input)
{

    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    // Output shape: (H, W, C)
    std::vector<std::vector<std::vector<uint8_t>>> output(H, std::vector<std::vector<uint8_t>>(W, std::vector<uint8_t>(C)));

    for (int c = 0; c < C; ++c)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                output[h][w][c] = input[c][h][w]; // Swap dimensions
            }
        }
    }
    return output;
}

std::vector<std::vector<uint8_t>> reshape(
    const std::vector<std::vector<std::vector<uint8_t>>> &input, int C_out)
{

    int H = input.size();
    int W = input[0].size();
    int C = input[0][0].size();

    if (H * W * C % C_out != 0)
    {
        throw std::runtime_error("Invalid reshape dimensions");
    }

    int new_rows = (H * W * C) / C_out;

    std::vector<std::vector<uint8_t>> output(new_rows, std::vector<uint8_t>(C_out, 0));

    int index = 0;
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            for (int c = 0; c < C; ++c)
            {
                output[index / C_out][index % C_out] = input[h][w][c];
                index++;
            }
        }
    }
    return output;
}

std::vector<std::vector<uint8_t>> concatenate(
    const std::vector<std::vector<uint8_t>> &a,
    const std::vector<std::vector<uint8_t>> &b)
{

    // Ensure both tensors have the same number of columns
    if (a[0].size() != b[0].size())
    {
        throw std::runtime_error("Cannot concatenate: Column sizes do not match.");
    }

    // Copy both matrices into a new one
    std::vector<std::vector<uint8_t>> output;
    output.reserve(a.size() + b.size()); // Preallocate memory for efficiency

    // Append all rows from a
    output.insert(output.end(), a.begin(), a.end());
    // Append all rows from b
    output.insert(output.end(), b.begin(), b.end());

    return output;
}
