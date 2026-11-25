// Opt-in Tracy zones for the stand-ins (CMake option X2W_TRACY, a local measurement build).
// Without the option every macro expands to nothing and no Tracy header is included.
#pragma once

#if X2W_TRACY
#include <tracy/Tracy.hpp>
#define X2W_ZONE(name) ZoneScopedN(name)
#define X2W_FRAME_MARK FrameMark
#else
#define X2W_ZONE(name)
#define X2W_FRAME_MARK
#endif
