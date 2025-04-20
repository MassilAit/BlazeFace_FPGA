#include "blaze_face.h"
#include <fstream>
#include <sstream>
#include "utils.h"
#include <iostream>
#include <cmath>

// Function to extract a 1D tensor from JSON and quantize it to int32_t
std::vector<int32_t> get1DTensorAsInt32(const rapidjson::Value &json_array, float x_scale, float w_scale)
{
    std::vector<int32_t> tensor;
    for (size_t i = 0; i < json_array.Size(); ++i)
    {
        // Convert float bias to int32_t using: b_q = round(b / (x_scale * w_scale))
        int32_t quantized_bias = static_cast<int32_t>(std::round(json_array[i].GetFloat() / (x_scale * w_scale)));
        tensor.push_back(quantized_bias);
    }
    return tensor;
}

// Function to extract a 4D tensor from JSON as int8_t
std::vector<std::vector<std::vector<std::vector<int8_t>>>> get4DTensorAsInt8(const rapidjson::Value &json_array)
{
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> tensor;

    for (size_t i = 0; i < json_array.Size(); ++i)
    {
        std::vector<std::vector<std::vector<int8_t>>> sub_tensor;
        for (size_t j = 0; j < json_array[i].Size(); ++j)
        {
            std::vector<std::vector<int8_t>> matrix;
            for (size_t k = 0; k < json_array[i][j].Size(); ++k)
            {
                std::vector<int8_t> row;
                for (size_t l = 0; l < json_array[i][j][k].Size(); ++l)
                {
                    row.push_back(static_cast<int8_t>(json_array[i][j][k][l].GetInt()));
                }
                matrix.push_back(row);
            }
            sub_tensor.push_back(matrix);
        }
        tensor.push_back(sub_tensor);
    }

    return tensor;
}

// Function to extract a 3D tensor from depthwise convolution (int8_t)
std::vector<std::vector<std::vector<int8_t>>> get3DTensorForDepthwiseAsInt8(const rapidjson::Value &json_array)
{
    std::vector<std::vector<std::vector<int8_t>>> tensor;

    for (size_t i = 0; i < json_array.Size(); ++i)
    { // in_channels
        std::vector<std::vector<int8_t>> matrix;
        for (size_t k = 0; k < json_array[i][0].Size(); ++k)
        { // kernel height
            std::vector<int8_t> row;
            for (size_t l = 0; l < json_array[i][0][k].Size(); ++l)
            { // kernel width
                row.push_back(static_cast<int8_t>(json_array[i][0][k][l].GetInt()));
            }
            matrix.push_back(row);
        }
        tensor.push_back(matrix);
    }

    return tensor;
}

// Function to parse JSON and extract convolution parameters
BlazeBlock createBlazeBlock(const rapidjson::Document &doc,
                            const std::string &name,
                            int in_channel, int out_channel,
                            int k_size, int strd,
                            float x_scale, uint8_t x_zero)
{

    // Depth wise convolution layer
    float dw_w_scale = doc[(name + ".convs.0.weight").c_str()]["scale"].GetFloat();
    float dw_y_scale = doc[(name + ".convs.0.scale").c_str()].GetFloat();
    uint8_t dw_y_zero = static_cast<uint8_t>(doc[(name + ".convs.0.zero_point").c_str()].GetInt());
    const auto &w_dw = get3DTensorForDepthwiseAsInt8(doc[(name + ".convs.0.weight").c_str()]["values"]);
    const auto &b_dw = get1DTensorAsInt32(doc[(name + ".convs.0.bias").c_str()], x_scale, dw_w_scale);

    // Point wise convolution layer
    float pw_w_scale = doc[(name + ".convs.1.weight").c_str()]["scale"].GetFloat();
    float pw_y_scale = doc[(name + ".convs.1.scale").c_str()].GetFloat();
    uint8_t pw_y_zero = static_cast<uint8_t>(doc[(name + ".convs.1.zero_point").c_str()].GetInt());
    const auto &w_pw = get4DTensorAsInt8(doc[(name + ".convs.1.weight").c_str()]["values"]);
    const auto &b_pw = get1DTensorAsInt32(doc[(name + ".convs.1.bias").c_str()], dw_y_scale, pw_w_scale);

    // Skip connection
    float y_scale = doc[(name + ".skip_add.scale").c_str()].GetFloat();
    uint8_t y_zero = static_cast<uint8_t>(doc[(name + ".skip_add.zero_point").c_str()].GetInt());

    return BlazeBlock(in_channel, out_channel, k_size, strd,
                      w_dw, b_dw, dw_y_scale, dw_w_scale, dw_y_zero,
                      w_pw, b_pw, pw_y_scale, pw_w_scale, pw_y_zero,
                      y_scale, y_zero);
}

BlazeFace::BlazeFace(const std::string &weights_file)
    : input_conv(3, 24, 5, 2, 0, 0.0, 0.0, {}, {}), // Default init
      classifier_8(88, 2, 1, 1, 0, 0.0, 0.0, {}, {}),
      classifier_16(96, 6, 1, 1, 0, 0.0, 0.0, {}, {}),
      regressor_8(88, 32, 1, 1, 0, 0.0, 0.0, {}, {}),
      regressor_16(96, 96, 1, 1, 0, 0.0, 0.0, {}, {})
{
    // Open JSON file
    std::ifstream ifs(weights_file);
    if (!ifs.is_open())
    {
        std::cerr << "Error opening file: " << weights_file << std::endl;
    }

    // Read JSON content
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    // Load x_scale, x_zero

    scale_x = doc["quant.scale"][0].GetFloat();
    zero_x = static_cast<uint8_t>(doc["quant.zero_point"][0].GetInt());

    // Load input layer weights
    float layer_0_w_scale = doc["backbone1.0.weight"]["scale"].GetFloat();
    float layer_0_scale = doc["backbone1.0.scale"].GetFloat();
    uint8_t layer_0_zero = static_cast<uint8_t>(doc["backbone1.0.zero_point"].GetInt());
    const auto &layer_0_weight = get4DTensorAsInt8(doc["backbone1.0.weight"]["values"]);
    const auto &layer_0_bias = get1DTensorAsInt32(doc["backbone1.0.bias"], scale_x, layer_0_w_scale);

    input_conv = Conv2D(3, 24, 5, 2, 0, layer_0_w_scale, layer_0_scale, layer_0_zero, layer_0_weight, layer_0_bias);

    // Load backbone1 layers
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.2", 24, 24, 3, 1, input_conv.y_scale, input_conv.y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.3", 24, 28, 3, 1, backbone1[0].y_scale, backbone1[0].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.4", 28, 32, 3, 2, backbone1[1].y_scale, backbone1[1].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.5", 32, 36, 3, 1, backbone1[2].y_scale, backbone1[2].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.6", 36, 42, 3, 1, backbone1[3].y_scale, backbone1[3].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.7", 42, 48, 3, 2, backbone1[4].y_scale, backbone1[4].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.8", 48, 56, 3, 1, backbone1[5].y_scale, backbone1[5].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.9", 56, 64, 3, 1, backbone1[6].y_scale, backbone1[6].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.10", 64, 72, 3, 1, backbone1[7].y_scale, backbone1[7].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.11", 72, 80, 3, 1, backbone1[8].y_scale, backbone1[8].y_zero));
    backbone1.emplace_back(createBlazeBlock(doc, "backbone1.12", 80, 88, 3, 1, backbone1[9].y_scale, backbone1[9].y_zero));

    // Load backbone2 layers
    backbone2.emplace_back(createBlazeBlock(doc, "backbone2.0", 88, 96, 3, 2, backbone1[10].y_scale, backbone1[10].y_zero));
    backbone2.emplace_back(createBlazeBlock(doc, "backbone2.1", 96, 96, 3, 1, backbone2[0].y_scale, backbone2[0].y_zero));
    backbone2.emplace_back(createBlazeBlock(doc, "backbone2.2", 96, 96, 3, 1, backbone2[1].y_scale, backbone2[1].y_zero));
    backbone2.emplace_back(createBlazeBlock(doc, "backbone2.3", 96, 96, 3, 1, backbone2[2].y_scale, backbone2[2].y_zero));
    backbone2.emplace_back(createBlazeBlock(doc, "backbone2.4", 96, 96, 3, 1, backbone2[3].y_scale, backbone2[3].y_zero));

    // Load classifier layers
    float classifier_8_w_scale = doc["classifier_8.weight"]["scale"].GetFloat();
    float classifier_8_scale = doc["classifier_8.scale"].GetFloat();
    uint8_t classifier_8_zero = static_cast<uint8_t>(doc["classifier_8.zero_point"].GetInt());
    const auto &classifier_8_weight = get4DTensorAsInt8(doc["classifier_8.weight"]["values"]);
    const auto &classifier_8_bias = get1DTensorAsInt32(doc["classifier_8.bias"], backbone1[10].y_scale, classifier_8_w_scale);

    classifier_8 = Conv2D(88, 2, 1, 1, 0, classifier_8_w_scale, classifier_8_scale, classifier_8_zero, classifier_8_weight, classifier_8_bias);

    float classifier_16_w_scale = doc["classifier_16.weight"]["scale"].GetFloat();
    float classifier_16_scale = doc["classifier_16.scale"].GetFloat();
    uint8_t classifier_16_zero = static_cast<uint8_t>(doc["classifier_16.zero_point"].GetInt());
    const auto &classifier_16_weight = get4DTensorAsInt8(doc["classifier_16.weight"]["values"]);
    const auto &classifier_16_bias = get1DTensorAsInt32(doc["classifier_16.bias"], backbone2[4].y_scale, classifier_16_w_scale);

    classifier_16 = Conv2D(96, 6, 1, 1, 0, classifier_16_w_scale, classifier_16_scale, classifier_16_zero, classifier_16_weight, classifier_16_bias);

    // Load regressor layers
    float regressor_8_w_scale = doc["regressor_8.weight"]["scale"].GetFloat();
    float regressor_8_scale = doc["regressor_8.scale"].GetFloat();
    uint8_t regressor_8_zero = static_cast<uint8_t>(doc["regressor_8.zero_point"].GetInt());
    const auto &regressor_8_weight = get4DTensorAsInt8(doc["regressor_8.weight"]["values"]);
    const auto &regressor_8_bias = get1DTensorAsInt32(doc["regressor_8.bias"], backbone1[10].y_scale, regressor_8_w_scale);

    regressor_8 = Conv2D(88, 32, 1, 1, 0, regressor_8_w_scale, regressor_8_scale, regressor_8_zero, regressor_8_weight, regressor_8_bias);

    float regressor_16_w_scale = doc["regressor_16.weight"]["scale"].GetFloat();
    float regressor_16_scale = doc["regressor_16.scale"].GetFloat();
    uint8_t regressor_16_zero = static_cast<uint8_t>(doc["regressor_16.zero_point"].GetInt());
    const auto &regressor_16_weight = get4DTensorAsInt8(doc["regressor_16.weight"]["values"]);
    const auto &regressor_16_bias = get1DTensorAsInt32(doc["regressor_16.bias"], backbone2[4].y_scale, regressor_16_w_scale);

    regressor_16 = Conv2D(96, 96, 1, 1, 0, regressor_16_w_scale, regressor_16_scale, regressor_16_zero, regressor_16_weight, regressor_16_bias);
}

std::vector<std::vector<std::vector<uint8_t>>> BlazeFace::quantize(
    const std::vector<std::vector<std::vector<float>>> &input, float x_scale, uint8_t x_zero)
{

    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    // Output quantized tensor (same dimensions as input)
    std::vector<std::vector<std::vector<uint8_t>>> output(channels,
                                                          std::vector<std::vector<uint8_t>>(height, std::vector<uint8_t>(width, 0)));

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                // Quantization formula: x_q = round(x / x_scale) + x_zero
                int32_t quantized_value = static_cast<int32_t>(std::round(input[c][i][j] / x_scale) + x_zero);

                // Clamp to valid uint8 range (0-255)
                output[c][i][j] = static_cast<uint8_t>(std::max(0, std::min(255, quantized_value)));
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<float>>> BlazeFace::dequantize(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    float x_scale, uint8_t x_zero)
{

    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    // Output dequantized tensor (same dimensions as input)
    std::vector<std::vector<std::vector<float>>> output(channels,
                                                        std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                // Dequantization formula: x_float = (x_q - x_zero) * x_scale
                output[c][i][j] = (static_cast<float>(input[c][i][j]) - x_zero) * x_scale;
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<uint8_t>>> BlazeFace::forward(
    const std::vector<std::vector<std::vector<uint8_t>>> &input,
    MemMap &map)
{

    std::vector<std::vector<std::vector<std::vector<float>>>> output;

    // Apply manual padding
    auto x_uint = pad(input, 1, 2, 1, 2, 0, 0, 0);

    x_uint = input_conv.forward(x_uint, zero_x, scale_x, map);

    x_uint = relu(x_uint, input_conv.y_zero, map);

    float scale_backbone1 = input_conv.y_scale;
    uint8_t zero_backbone1 = input_conv.y_zero;

    // Pass through backbone1
    for (auto &layer : backbone1)
    {
        x_uint = layer.forward(x_uint, zero_backbone1, scale_backbone1, map);
        zero_backbone1 = layer.y_zero;
        scale_backbone1 = layer.y_scale;
    }

    auto h_uint = x_uint; // Store intermediate output for backbone2

    float scale_backbone2 = scale_backbone1;
    uint8_t zero_backbone2 = zero_backbone1;

    // Pass through backbone2
    for (auto &layer : backbone2)
    {
        h_uint = layer.forward(h_uint, zero_backbone2, scale_backbone2, map);
        zero_backbone2 = layer.y_zero;
        scale_backbone2 = layer.y_scale;
    }

    // Classifier outputs
    auto c1_1 = classifier_8.forward(x_uint, zero_backbone1, scale_backbone1, map); // (2, 16, 16)
    auto c1_2 = permute(c1_1);                                                      //  (16, 16, 2)
    auto c1 = reshape(c1_2, 1);                                                     //  (512, 1)

    auto c2_1 = classifier_16.forward(h_uint, zero_backbone2, scale_backbone2, map); // (6, 8, 8)
    auto c2_2 = permute(c2_1);                                                       // (8, 8, 6)
    auto c2 = reshape(c2_2, 1);                                                      //  (384, 1)

    auto c = concatenate(c1, c2); //(896, 1)

    //// Regressor outputs
    auto r1_1 = regressor_8.forward(x_uint, zero_backbone1, scale_backbone1, map); // (32, 16, 16)
    auto r1_2 = permute(r1_1);                                                     //  (16, 16, 32)
    auto r1 = reshape(r1_2, 16);                                                   //  (512, 16)

    auto r2_1 = regressor_16.forward(h_uint, zero_backbone2, scale_backbone2, map); // (96, 8, 8)
    auto r2_2 = permute(r2_1);                                                      //  (8, 8, 96)
    auto r2 = reshape(r2_2, 16);                                                    //  (384, 16)

    auto r = concatenate(r1, r2); //(896, 16)

    return {r, c};
}

std::string BlazeFace::get_model_summary() const
{
    std::ostringstream oss;
    oss << "=== BlazeFace Model Summary ===\n";

    oss << input_conv.get_layer_info("Input Conv2D");

    for (size_t i = 0; i < backbone1.size(); i++)
    {
        oss << backbone1[i].get_layer_info("Backbone1 Layer " + std::to_string(i + 2));
    }
    for (size_t i = 0; i < backbone2.size(); i++)
    {
        oss << backbone2[i].get_layer_info("Backbone2 Layer " + std::to_string(i));
    }

    oss << classifier_8.get_layer_info("Classifier 8");
    oss << classifier_16.get_layer_info("Classifier 16");
    oss << regressor_8.get_layer_info("Regressor 8");
    oss << regressor_16.get_layer_info("Regressor 16");

    oss << "================================\n";
    return oss.str();
}
