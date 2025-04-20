#include "serial_com.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <vector>

std::optional<int> init_serial(const char *device, int baud)
{
    int fd = setup_serial(device, baud);
    if (fd < 0)
    {
        std::cerr << "Error: failed to open " << device << "\n";
        return std::nullopt;
    }
    return fd;
}

static int configure_port(int fd, int baudrate)
{
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0)
    {
        std::perror("tcgetattr");
        return -1;
    }

    speed_t speed;
    switch (baudrate)
    {
    case 230400:
        speed = B230400;
        break;
    case 3000000:
        speed = B3000000;
        break;
    default:
        speed = B230400;
        break;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8N1, no flow control
    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_lflag = 0;
    tty.c_cc[VMIN] = 1;   // block until at least 1 byte
    tty.c_cc[VTIME] = 10; // 1s read timeout

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
    {
        std::perror("tcsetattr");
        return -1;
    }
    return 0;
}

int setup_serial(const char *device, int baudrate)
{
    int fd = open(device, O_RDWR | O_NOCTTY);
    if (fd < 0)
    {
        std::perror("open");
        return -1;
    }
    if (configure_port(fd, baudrate) != 0)
    {
        close(fd);
        return -1;
    }
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    return fd;
}

int read_full(int fd, uint8_t *buf, int total)
{
    int got = 0;
    while (got < total)
    {
        int n = ::read(fd, buf + got, total - got);
        if (n <= 0)
        {
            if (n < 0)
                std::perror("read");
            break;
        }
        got += n;
    }
    return got;
}

std::optional<std::vector<std::vector<std::vector<uint8_t>>>> read_input(int fd)
{

    std::vector<uint8_t> buf(N_IN);
    int got = read_full(fd, buf.data(), N_IN);
    if (got != N_IN)
    {
        std::cerr << "read_frame: expected " << N_IN
                  << " bytes, got " << got << "\n";
        return std::nullopt;
    }

    // 2) unpack into [3][128][128]
    std::vector<std::vector<std::vector<uint8_t>>> frame(3,
                                                         std::vector<std::vector<uint8_t>>(128, std::vector<uint8_t>(128)));

    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < 128; ++i)
        {
            for (int j = 0; j < 128; ++j)
            {
                // planar layout: channel-major
                frame[c][i][j] = buf[c * (128 * 128) + i * 128 + j];
            }
        }
    }
    return frame;
}

int write_full(int fd, const uint8_t *buf, int total)
{
    int sent = 0;
    while (sent < total)
    {
        int n = ::write(fd, buf + sent, total - sent);
        if (n <= 0)
        {
            if (n < 0)
                std::perror("write");
            break;
        }
        sent += n;
    }
    return sent;
}

std::optional<int> send_output(int fd, const std::vector<std::vector<std::vector<uint8_t>>> &output)
{
    // 1) Compute total number of bytes
    size_t total = 0;
    for (const auto &mat2d : output)
        for (const auto &row : mat2d)
            total += row.size();

    // 2) Flatten into a single buffer
    std::vector<uint8_t> buf;
    buf.reserve(total);
    for (const auto &mat2d : output)
        for (const auto &row : mat2d)
            for (uint8_t v : row)
                buf.push_back(v);

    if (buf.size() != total)
    {
        std::cerr << "send_output: size mismatch ("
                  << buf.size() << " vs " << total << ")\n";
        return std::nullopt;
    }

    // 3) Send all bytes
    int written = write_full(fd, buf.data(), static_cast<int>(total));
    if (written != static_cast<int>(total))
    {
        std::cerr << "send_output: only wrote "
                  << written << " of " << total << " bytes\n";
        return std::nullopt;
    }

    return written;
}
