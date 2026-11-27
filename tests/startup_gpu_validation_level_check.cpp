#include "startup_stubs/debug_level.h"

#include <cstdio>
#include <limits>

int main()
{
    using x2w_startup::DebugLevel;
    for (int ch = (std::numeric_limits<char>::min)(); ch <= (std::numeric_limits<char>::max)(); ++ch) {
        auto expected = DebugLevel::off;
        if (ch == '1') expected = DebugLevel::layer;
        if (ch == '2') expected = DebugLevel::gpu_validation;
        if (x2w_startup::d3d12_debug_level(static_cast<char>(ch)) != expected) {
            std::printf("FAIL debug level for character %d\n", ch);
            return 1;
        }
    }
    return 0;
}
