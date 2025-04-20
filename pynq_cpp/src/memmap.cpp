#include "memmap.h"
#include "registers.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

std::optional<MemMap> allocate_mem()
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0)
    {
        std::cerr << "ERROR: cannot open /dev/mem\n";
        return std::nullopt;
    }

    void *ctrl_pw = mmap(nullptr, CTRL_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, PW_BASE_ADDR);
    if (ctrl_pw == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap ctrl pw failed\n";
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // DepthWise control register
    // ---------------------------

    void *ctrl_dw = mmap(nullptr, CTRL_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DW_BASE_ADDR);
    if (ctrl_dw == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap ctrl dw failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // RELU control register
    // ---------------------------

    void *ctrl_relu = mmap(nullptr, CTRL_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RELU_BASE_ADDR);
    if (ctrl_relu == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap ctrl relu failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        munmap(ctrl_dw, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // IN DDR alocation
    // ---------------------------

    void *in_map = mmap(nullptr, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IN_DDR_ADDR);
    if (in_map == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap in buffer failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        munmap(ctrl_dw, CTRL_MAP_SIZE);
        munmap(ctrl_relu, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // OUT DDR alocation
    // ---------------------------

    void *out_map = mmap(nullptr, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, OUT_DDR_ADDR);
    if (out_map == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap out buffer failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        munmap(in_map, DDR_MAP_SIZE);
        munmap(ctrl_dw, CTRL_MAP_SIZE);
        munmap(ctrl_relu, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // WEIGHTS DDR alocation
    // ---------------------------

    void *weights_map = mmap(nullptr, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, WEIGHT_DDR_ADDR);
    if (weights_map == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap weights buffer failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        munmap(in_map, DDR_MAP_SIZE);
        munmap(out_map, DDR_MAP_SIZE);
        munmap(ctrl_dw, CTRL_MAP_SIZE);
        munmap(ctrl_relu, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    // ---------------------------
    // BIAS DDR alocation
    // ---------------------------

    void *bias_map = mmap(nullptr, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, BIAS_DDR_ADDR);
    if (bias_map == MAP_FAILED)
    {
        std::cerr << "ERROR: mmap bias buffer failed\n";
        munmap(ctrl_pw, CTRL_MAP_SIZE);
        munmap(in_map, DDR_MAP_SIZE);
        munmap(out_map, DDR_MAP_SIZE);
        munmap(weights_map, DDR_MAP_SIZE);
        munmap(ctrl_dw, CTRL_MAP_SIZE);
        munmap(ctrl_relu, CTRL_MAP_SIZE);
        close(fd);
        return std::nullopt;
    }

    return MemMap{
        .fd = fd,
        .ctrl_pw = static_cast<volatile uint32_t *>(ctrl_pw),
        .ctrl_dw = static_cast<volatile uint32_t *>(ctrl_dw),
        .ctrl_relu = static_cast<volatile uint32_t *>(ctrl_relu),
        .in = static_cast<volatile uint8_t *>(in_map),
        .out = static_cast<volatile uint8_t *>(out_map),
        .weights = static_cast<volatile int8_t *>(weights_map),
        .bias = static_cast<volatile int32_t *>(bias_map),
    };
}

void free_mem(const MemMap &m)
{
    munmap((void *)m.in, DDR_MAP_SIZE);
    munmap((void *)m.out, DDR_MAP_SIZE);
    munmap((void *)m.weights, DDR_MAP_SIZE);
    munmap((void *)m.bias, DDR_MAP_SIZE);
    munmap((void *)m.ctrl_dw, CTRL_MAP_SIZE);
    munmap((void *)m.ctrl_relu, CTRL_MAP_SIZE);
    munmap((void *)m.ctrl_pw, CTRL_MAP_SIZE);
    close(m.fd);
}