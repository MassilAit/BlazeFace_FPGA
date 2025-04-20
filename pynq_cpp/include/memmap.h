#ifndef MEMMAP
#define MEMMAP

#include <optional>
#include <cstdint>

struct MemMap
{
  int fd;
  volatile uint32_t *ctrl_pw;
  volatile uint32_t *ctrl_dw;
  volatile uint32_t *ctrl_relu;
  volatile uint8_t *in;
  volatile uint8_t *out;
  volatile int8_t *weights;
  volatile int32_t *bias;
};

std::optional<MemMap> allocate_mem();
void free_mem(const MemMap &);

#endif