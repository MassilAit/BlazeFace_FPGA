#ifndef REGISTERS
#define REGISTERS

// ---------------------------
// AXI-Lite Control Interface
// ---------------------------

#define RELU_BASE_ADDR 0x40000000
#define DW_BASE_ADDR 0x40010000
#define PW_BASE_ADDR 0x40020000
#define FULL_BASE_ADDR 0x40030000

#define CTRL_MAP_SIZE 0x10000

// Relu Offsets
#define AP_CTRL_RELU 0x00
#define IN_LO_RELU 0x10
#define IN_HI_RELU 0x14
#define OUT_LO_RELU 0x1C
#define OUT_HI_RELU 0x20
#define SIZE_REG_RELU 0x28
#define X_ZERO_RELU 0x30

// DepthWise Offsets
#define AP_CTRL_DW 0x00
#define IN_LO_DW 0x10
#define IN_HI_DW 0x14
#define W_LO_DW 0x1C
#define W_HI_DW 0x20
#define B_LO_DW 0x28
#define B_HI_DW 0x2C
#define OUT_LO_DW 0x34
#define OUT_HI_DW 0x38
#define SIZE_DW 0x40
#define CIN_DW 0x48
#define STRIDE_DW 0x50
#define X_ZERO_DW 0x58
#define Y_ZERO_DW 0x60
#define MUL_DW 0x68
#define SHIFT_DW 0x70

// PointWise Offsets
#define AP_CTRL_PW 0x00
#define IN_LO_PW 0x10
#define IN_HI_PW 0x14
#define W_LO_PW 0x1C
#define W_HI_PW 0x20
#define B_LO_PW 0x28
#define B_HI_PW 0x2C
#define OUT_LO_PW 0x34
#define OUT_HI_PW 0x38
#define IN_SIZE_PW 0x40
#define OUT_SIZE_PW 0x48
#define CIN_PW 0x50
#define COUT_PW 0x58
#define STRIDE_PW 0x60
#define X_ZERO_PW 0x68
#define Y_ZERO_PW 0x70
#define MUL_PW 0x78
#define SHIFT_PW 0x80


// FUll Offsets
#define AP_CTRL_F 0x00
#define IN_LO_F 0x10
#define IN_HI_F 0x14
#define W_LO_F 0x1C
#define W_HI_F 0x20
#define B_LO_F 0x28
#define B_HI_F 0x2C
#define OUT_LO_F 0x34
#define OUT_HI_F 0x38
#define IN_SIZE_F 0x40
#define OUT_SIZE_F 0x48
#define CIN_F 0x50
#define COUT_F 0x58
#define STRIDE_F 0x60
#define X_ZERO_F 0x68
#define Y_ZERO_F 0x70
#define MUL_F 0x78
#define SHIFT_F 0x80


// ---------------------------
// DDR Addresses
// ---------------------------

#define IN_DDR_ADDR 0x1F000000
#define OUT_DDR_ADDR 0x1F010000
#define WEIGHT_DDR_ADDR 0x1F020000
#define BIAS_DDR_ADDR 0x1F030000
#define DDR_MAP_SIZE 0x10000

#endif