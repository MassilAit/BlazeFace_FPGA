#ifndef SERIAL_COM_H
#define SERIAL_COM_H

#include <cstdint>
#include <optional>
#include <vector>

constexpr int N_IN = 3 * 128 * 128; // 49152
constexpr int N_OUT = 896 * 17;     // 15232

constexpr const char *device = "/dev/ttyPS0";
constexpr int baud = 3000000;

std::optional<int> init_serial(const char *device, int baud);

int setup_serial(const char *device, int baudrate);

std::optional<std::vector<std::vector<std::vector<uint8_t>>>> read_input(int fd);

std::optional<int> send_output(int fd, const std::vector<std::vector<std::vector<uint8_t>>> &output);

#endif
