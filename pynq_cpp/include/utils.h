#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <cstdint>

std::vector<float> load_vector_from_file(const std::string &filename);
std::vector<uint8_t> load_uint8_vector_from_file(const std::string &filename);
std::vector<std::vector<std::vector<uint8_t>>> load_input_uint8(const std::string &filename, int in_channels, int input_size);
std::vector<std::vector<std::vector<float>>> load_input(const std::string &filename, int in_channels, int input_size);
std::vector<size_t> get_shape(const std::vector<std::vector<std::vector<float>>> &vec);
std::vector<std::vector<std::vector<uint8_t>>> permute(const std::vector<std::vector<std::vector<uint8_t>>> &input);
std::vector<std::vector<uint8_t>> reshape(const std::vector<std::vector<std::vector<uint8_t>>> &input, int C_out);
std::vector<std::vector<uint8_t>> concatenate(const std::vector<std::vector<uint8_t>> &a, const std::vector<std::vector<uint8_t>> &b);
std::vector<std::vector<float>> load_output(const std::string &filename, int num_rows, int num_cols);

#endif // UTILS_H
