// Print the DXIL-text IR that the direct C++ frontend builds for a pixel container.
// Used by native_pack_b32_f16_check.py to evaluate the emitted operations without a GPU.
// usage: native_pixel_ir_dump CONTAINER   (exit 1 and the diagnostic on stderr if refused)
#include "gfx1013_compute_native.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        std::cout << x2w::translate_graphics_container(bytes, true).dxil_ir;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
