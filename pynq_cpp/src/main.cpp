#include <iostream>
#include <unistd.h>
#include "blaze_face.h"
#include "memmap.h"
#include "serial_com.h"

int main()
{

    // Allocates memory for PS-PL communication
    auto optional = allocate_mem();
    if (!optional)
        return 1;

    MemMap mem = *optional;

    // Oppening Serial Communication

    auto fd_opt = init_serial(device, baud);
    if (!fd_opt)
        return 1;

    int fd = *fd_opt;

    // Loading model
    BlazeFace model = BlazeFace("data/params/qblazeface_weights.json");

    std::cout << model.get_model_summary();

    while (true)
    {

        // Receive input :
        auto input = read_input(fd);
        if (input)
        {
            // Run Inference

            auto output = model.forward(*input, mem);

            // Send output
            auto res = send_output(fd, output);
            if (!res)
            {
                std::cout << "Failed to send output\n";
                break;
            }
        }
    }

    // Frees the memory
    close(fd);
    free_mem(mem);
    return 0;
}
