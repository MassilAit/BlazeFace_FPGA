#include "accel_control.h"
#include "registers.h"
#include <cmath>

void QuantizeMultiplier(double real_multiplier, int32_t &multiplier, int32_t &shift)
{
    if (real_multiplier == 0.0)
    {
        multiplier = 0;
        shift = 0;
        return;
    }

    int exp;
    double q = std::frexp(real_multiplier, &exp);
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));

    multiplier = static_cast<int32_t>(q_fixed);
    shift = exp;
}

void start_pointwise_conv(const MemMap &mem, int input_size, int output_size, int Cin, int Cout, int stride, uint8_t x_zero, uint8_t y_zero, int32_t M, int32_t shift)
{
    volatile uint32_t *reg = mem.ctrl_pw;

    // Write input address
    uint64_t in_phys = IN_DDR_ADDR;
    reg[IN_LO_PW / 4] = (uint32_t)(in_phys & 0xFFFFFFFF);
    reg[IN_HI_PW / 4] = (uint32_t)(in_phys >> 32);

    // Weights
    uint64_t w_phys = WEIGHT_DDR_ADDR;
    reg[W_LO_PW / 4] = (uint32_t)(w_phys & 0xFFFFFFFF);
    reg[W_HI_PW / 4] = (uint32_t)(w_phys >> 32);

    // Biases
    uint64_t b_phys = BIAS_DDR_ADDR;
    reg[B_LO_PW / 4] = (uint32_t)(b_phys & 0xFFFFFFFF);
    reg[B_HI_PW / 4] = (uint32_t)(b_phys >> 32);

    // Output
    uint64_t out_phys = OUT_DDR_ADDR;
    reg[OUT_LO_PW / 4] = (uint32_t)(out_phys & 0xFFFFFFFF);
    reg[OUT_HI_PW / 4] = (uint32_t)(out_phys >> 32);

    // Params
    reg[IN_SIZE_PW / 4] = input_size;
    reg[OUT_SIZE_PW / 4] = output_size;
    reg[CIN_PW / 4] = Cin;
    reg[COUT_PW / 4] = Cout;
    reg[STRIDE_PW / 4] = stride;
    reg[X_ZERO_PW / 4] = x_zero;
    reg[Y_ZERO_PW / 4] = y_zero;
    reg[MUL_PW / 4] = M;
    reg[SHIFT_PW / 4] = shift;

    while ((reg[AP_CTRL_PW / 4] & 0x4) == 0) /* spin */
        ;                                    // bit‑2 = ap_idle

    // Start accelerator
    reg[AP_CTRL_PW / 4] = 0x01;

    // Wait for completion
    while ((reg[AP_CTRL_PW / 4] & 0x2) == 0)
        ;
}

void run_pw_conv_accel(const MemMap &mem,
                       const std::vector<std::vector<std::vector<uint8_t>>> &input,
                       const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &weights,
                       const std::vector<int32_t> &biases,
                       std::vector<std::vector<std::vector<uint8_t>>> &output,
                       int stride,
                       float x_scale,
                       float w_scale,
                       float y_scale,
                       uint8_t x_zero,
                       uint8_t y_zero)
{
    int Cin = input.size();
    int Cout = weights.size();
    int size = input[0].size(); // assume square input
    int kernel_size = weights[0][0].size();
    int OH = (size - kernel_size) / stride + 1;

    // Compute fixed-point multiplier and shift
    int32_t M, shift_val;
    double real_multiplier = (x_scale * w_scale) / y_scale;
    QuantizeMultiplier(real_multiplier, M, shift_val);

    // Flatten input
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                mem.in[c * size * size + i * size + j] = input[c][i][j];
            }
        }
    }

    // Zero out output
    for (int i = 0; i < Cout * OH * OH; ++i)
    {
        mem.out[i] = 10;
    }

    // Flatten weights
    for (int co = 0; co < Cout; ++co)
    {
        for (int c = 0; c < Cin; ++c)
        {
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    mem.weights[co * Cin * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j] = weights[co][c][i][j];
                }
            }
        }
    }

    // Copy biases
    for (int i = 0; i < Cout; ++i)
    {
        mem.bias[i] = biases[i];
    }

    // Launch the accelerator

    start_pointwise_conv(mem, size, OH, Cin, Cout, stride, x_zero, y_zero, M, shift_val);

    // Copy back output
    output.resize(Cout, std::vector<std::vector<uint8_t>>(OH, std::vector<uint8_t>(OH)));
    for (int c = 0; c < Cout; ++c)
    {
        for (int i = 0; i < OH; ++i)
        {
            for (int j = 0; j < OH; ++j)
            {
                output[c][i][j] = mem.out[c * OH * OH + i * OH + j];
            }
        }
    }
}

void start_full_conv(const MemMap &mem, int input_size, int output_size, int Cin, int Cout, int stride, uint8_t x_zero, uint8_t y_zero, int32_t M, int32_t shift)
{
    volatile uint32_t *reg = mem.ctrl_pw;

    // Write input address
    uint64_t in_phys = IN_DDR_ADDR;
    reg[IN_LO_F / 4] = (uint32_t)(in_phys & 0xFFFFFFFF);
    reg[IN_HI_F / 4] = (uint32_t)(in_phys >> 32);

    // Weights
    uint64_t w_phys = WEIGHT_DDR_ADDR;
    reg[W_LO_F / 4] = (uint32_t)(w_phys & 0xFFFFFFFF);
    reg[W_HI_F / 4] = (uint32_t)(w_phys >> 32);

    // Biases
    uint64_t b_phys = BIAS_DDR_ADDR;
    reg[B_LO_F / 4] = (uint32_t)(b_phys & 0xFFFFFFFF);
    reg[B_HI_F / 4] = (uint32_t)(b_phys >> 32);

    // Output
    uint64_t out_phys = OUT_DDR_ADDR;
    reg[OUT_LO_F / 4] = (uint32_t)(out_phys & 0xFFFFFFFF);
    reg[OUT_HI_F / 4] = (uint32_t)(out_phys >> 32);

    // Params
    reg[IN_SIZE_F / 4] = input_size;
    reg[OUT_SIZE_F / 4] = output_size;
    reg[CIN_F / 4] = Cin;
    reg[COUT_F / 4] = Cout;
    reg[STRIDE_F / 4] = stride;
    reg[X_ZERO_F / 4] = x_zero;
    reg[Y_ZERO_F / 4] = y_zero;
    reg[MUL_F / 4] = M;
    reg[SHIFT_F / 4] = shift;

    while ((reg[AP_CTRL_F / 4] & 0x4) == 0) /* spin */
        ;                                   // bit‑2 = ap_idle

    // Start accelerator
    reg[AP_CTRL_F / 4] = 0x01;

    // Wait for completion
    while ((reg[AP_CTRL_F / 4] & 0x2) == 0)
        ;
}

void run_full_conv_accel(const MemMap &mem,
                         const std::vector<std::vector<std::vector<uint8_t>>> &input,
                         const std::vector<std::vector<std::vector<std::vector<int8_t>>>> &weights,
                         const std::vector<int32_t> &biases,
                         std::vector<std::vector<std::vector<uint8_t>>> &output,
                         int stride,
                         float x_scale,
                         float w_scale,
                         float y_scale,
                         uint8_t x_zero,
                         uint8_t y_zero)
{
    int Cin = input.size();
    int Cout = weights.size();
    int size = input[0].size(); // assume square input
    int kernel_size = weights[0][0].size();
    int OH = (size - kernel_size) / stride + 1;

    // Compute fixed-point multiplier and shift
    int32_t M, shift_val;
    double real_multiplier = (x_scale * w_scale) / y_scale;
    QuantizeMultiplier(real_multiplier, M, shift_val);

    // Flatten input
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                mem.in[c * size * size + i * size + j] = input[c][i][j];
            }
        }
    }

    // Zero out output
    for (int i = 0; i < Cout * OH * OH; ++i)
    {
        mem.out[i] = 10;
    }

    // Flatten weights
    for (int co = 0; co < Cout; ++co)
    {
        for (int c = 0; c < Cin; ++c)
        {
            for (int i = 0; i < kernel_size; ++i)
            {
                for (int j = 0; j < kernel_size; ++j)
                {
                    mem.weights[co * Cin * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j] = weights[co][c][i][j];
                }
            }
        }
    }

    // Copy biases
    for (int i = 0; i < Cout; ++i)
    {
        mem.bias[i] = biases[i];
    }

    // Launch the accelerator

    start_full_conv(mem, size, OH, Cin, Cout, stride, x_zero, y_zero, M, shift_val);

    // Copy back output
    output.resize(Cout, std::vector<std::vector<uint8_t>>(OH, std::vector<uint8_t>(OH)));
    for (int c = 0; c < Cout; ++c)
    {
        for (int i = 0; i < OH; ++i)
        {
            for (int j = 0; j < OH; ++j)
            {
                output[c][i][j] = mem.out[c * OH * OH + i * OH + j];
            }
        }
    }
}

void start_relu_accelerator(const MemMap &mem, int size, uint8_t x_zero)
{
    volatile uint32_t *reg = mem.ctrl_relu;

    uint64_t in_phys = IN_DDR_ADDR;
    uint64_t out_phys = OUT_DDR_ADDR;

    // Set addresses
    reg[IN_LO_RELU / 4] = static_cast<uint32_t>(in_phys & 0xFFFFFFFF);
    reg[IN_HI_RELU / 4] = static_cast<uint32_t>(in_phys >> 32);
    reg[OUT_LO_RELU / 4] = static_cast<uint32_t>(out_phys & 0xFFFFFFFF);
    reg[OUT_HI_RELU / 4] = static_cast<uint32_t>(out_phys >> 32);

    // Set size and zero-point
    reg[SIZE_REG_RELU / 4] = size;
    reg[X_ZERO_RELU / 4] = x_zero;

    // Start the accelerator
    reg[AP_CTRL_RELU / 4] = 0x01;

    // Wait for completion
    while ((reg[AP_CTRL_RELU / 4] & 0x2) == 0)
        ;
}

void run_relu_accel(const MemMap &mem,
                    const std::vector<std::vector<std::vector<uint8_t>>> &input,
                    std::vector<std::vector<std::vector<uint8_t>>> &output,
                    uint8_t x_zero)
{
    int Cin = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int size = Cin * H * W;

    // Flatten input
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                mem.in[c * H * W + i * W + j] = input[c][i][j];
            }
        }
    }

    // Clear output buffer
    for (int i = 0; i < size; ++i)
    {
        mem.out[i] = 0;
    }

    // Run the accelerator
    start_relu_accelerator(mem, size, x_zero);

    // Resize and copy output
    output.resize(Cin, std::vector<std::vector<uint8_t>>(H, std::vector<uint8_t>(W)));
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                output[c][i][j] = mem.out[c * H * W + i * W + j];
            }
        }
    }
}

void start_depthwise_conv(const MemMap &mem, int size, int Cin, int stride, uint8_t x_zero, uint8_t y_zero, int32_t M, int32_t shift)
{
    volatile uint32_t *reg = mem.ctrl_dw;

    // Write input address
    uint64_t in_phys = IN_DDR_ADDR;
    reg[IN_LO_DW / 4] = (uint32_t)(in_phys & 0xFFFFFFFF);
    reg[IN_HI_DW / 4] = (uint32_t)(in_phys >> 32);

    // Weights
    uint64_t w_phys = WEIGHT_DDR_ADDR;
    reg[W_LO_DW / 4] = (uint32_t)(w_phys & 0xFFFFFFFF);
    reg[W_HI_DW / 4] = (uint32_t)(w_phys >> 32);

    // Biases
    uint64_t b_phys = BIAS_DDR_ADDR;
    reg[B_LO_DW / 4] = (uint32_t)(b_phys & 0xFFFFFFFF);
    reg[B_HI_DW / 4] = (uint32_t)(b_phys >> 32);

    // Output
    uint64_t out_phys = OUT_DDR_ADDR;
    reg[OUT_LO_DW / 4] = (uint32_t)(out_phys & 0xFFFFFFFF);
    reg[OUT_HI_DW / 4] = (uint32_t)(out_phys >> 32);

    // Params
    reg[SIZE_DW / 4] = size;
    reg[CIN_DW / 4] = Cin;
    reg[STRIDE_DW / 4] = stride;
    reg[X_ZERO_DW / 4] = x_zero;
    reg[Y_ZERO_DW / 4] = y_zero;
    reg[MUL_DW / 4] = M;
    reg[SHIFT_DW / 4] = shift;

    // Start accelerator
    reg[AP_CTRL_DW / 4] = 0x01;

    // Wait for completion
    while ((reg[AP_CTRL_DW / 4] & 0x2) == 0)
        ;
}

void run_depthwise_conv_accel(const MemMap &mem,
                              const std::vector<std::vector<std::vector<uint8_t>>> &input,
                              const std::vector<std::vector<std::vector<int8_t>>> &weights,
                              const std::vector<int32_t> &biases,
                              std::vector<std::vector<std::vector<uint8_t>>> &output,
                              int stride,
                              float x_scale,
                              float w_scale,
                              float y_scale,
                              uint8_t x_zero,
                              uint8_t y_zero)
{
    int Cin = input.size();
    int size = input[0].size();          // assume square input
    int kernel_size = weights[0].size(); // usually 3
    int OH = (size - kernel_size) / stride + 1;

    // Compute fixed-point multiplier and shift
    int32_t M, shift_val;
    double real_multiplier = (x_scale * w_scale) / y_scale;
    QuantizeMultiplier(real_multiplier, M, shift_val);

    // Flatten input
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                mem.in[c * size * size + i * size + j] = input[c][i][j];
            }
        }
    }

    // Zero out output
    for (int i = 0; i < Cin * OH * OH; ++i)
    {
        mem.out[i] = 0;
    }

    // Flatten weights
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < kernel_size; ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                mem.weights[c * kernel_size * kernel_size + i * kernel_size + j] = weights[c][i][j];
            }
        }
    }

    // Copy biases
    for (int i = 0; i < Cin; ++i)
    {
        mem.bias[i] = biases[i];
    }

    // Launch the accelerator

    start_depthwise_conv(mem, size, Cin, stride, x_zero, y_zero, M, shift_val);

    // Copy back output
    output.resize(Cin, std::vector<std::vector<uint8_t>>(OH, std::vector<uint8_t>(OH)));
    for (int c = 0; c < Cin; ++c)
    {
        for (int i = 0; i < OH; ++i)
        {
            for (int j = 0; j < OH; ++j)
            {
                output[c][i][j] = mem.out[c * OH * OH + i * OH + j];
            }
        }
    }
}