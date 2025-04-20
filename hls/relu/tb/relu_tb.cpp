#include <iostream>
#include <cassert>
#include <ap_int.h>

#define SIZE 8

typedef ap_uint<8> data_t;  // use signed 8-bit

void relu(data_t* in, data_t* out, int size, data_t x_zero);

int main() {
    static data_t in[SIZE]  = {0, 1, 2, 12, 18, 50, 100, 127};
    data_t x_zero = 12;
    static data_t expected[SIZE];
    static data_t out[SIZE];

    // ReLU: output = max(0, input)
    for (int i = 0; i < SIZE; i++) {
        expected[i] = (in[i] > x_zero) ? in[i] : x_zero;
    }

    relu(in, out, SIZE, x_zero);

    for (int i = 0; i < SIZE; i++) {
        assert(out[i] == expected[i] && "ReLU output mismatch!");
    }

    std::cout << "All tests passed.\n";
    return 0;
}
