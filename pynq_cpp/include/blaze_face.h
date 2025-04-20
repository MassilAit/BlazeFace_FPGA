#ifndef BLAZEFACE_H
#define BLAZEFACE_H

#include "blaze_block.h"
#include "memmap.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <vector>
#include <string>

class BlazeFace
{
public:
    // Constructor: Load model weights from JSON file
    BlazeFace(const std::string &weights_file);

    std::string get_model_summary() const;

    // Forward pass function
    std::vector<std::vector<std::vector<uint8_t>>> forward(const std::vector<std::vector<std::vector<uint8_t>>> &input, MemMap &map);

    std::vector<std::vector<std::vector<uint8_t>>> quantize(
        const std::vector<std::vector<std::vector<float>>> &input, float x_scale, uint8_t x_zero);

    std::vector<std::vector<std::vector<float>>> dequantize(
        const std::vector<std::vector<std::vector<uint8_t>>> &input,
        float x_scale, uint8_t x_zero);

    float scale_x;
    uint8_t zero_x;

private:
    // Input convolution layer
    Conv2D input_conv;

    // Backbone layers (BlazeBlocks)
    std::vector<BlazeBlock> backbone1;
    std::vector<BlazeBlock> backbone2;

    // Output layers
    Conv2D classifier_8;
    Conv2D classifier_16;
    Conv2D regressor_8;
    Conv2D regressor_16;
};

#endif // BLAZEFACE_H
