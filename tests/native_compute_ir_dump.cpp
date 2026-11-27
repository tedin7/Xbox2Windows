// Print the DXIL-text IR that the direct C++ frontend builds for a compute container.
// Used by native_f16_alu_check.py. Exit 1 and the diagnostic on stderr when the program is refused.
// usage: native_compute_ir_dump CONTAINER
#include "gfx1013_compute_native.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    std::ifstream file(argv[1], std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        std::cout << x2w::translate_compute_container({bytes, {}}).dxil_ir;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
