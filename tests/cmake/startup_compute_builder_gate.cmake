# All-or-none gate for the console BVH builder's compute pipeline set
# (src/startup_stubs/compute_builder_gate.h), without a device or a title module.
add_executable(startup_compute_builder_gate_check tests/startup_compute_builder_gate_check.cpp)
target_include_directories(startup_compute_builder_gate_check PRIVATE src)
target_compile_features(startup_compute_builder_gate_check PRIVATE cxx_std_17)
add_test(NAME startup_compute_builder_gate COMMAND startup_compute_builder_gate_check)

# A source-built module whose basename matches XboxBvhBuilder.dll, so the public slot 62 path can
# exercise the caller-module tag and the direct/indirect Dispatch gate.
add_library(startup_builder_module SHARED tests/startup_builder_module.cpp)
set_target_properties(startup_builder_module PROPERTIES OUTPUT_NAME XboxBvhBuilder)
target_compile_features(startup_builder_module PRIVATE cxx_std_17)
target_link_libraries(startup_builder_module PRIVATE d3d12)

add_executable(startup_compute_builder_gate_runtime tests/startup_compute_builder_gate_runtime.cpp)
target_compile_features(startup_compute_builder_gate_runtime PRIVATE cxx_std_17)
target_link_libraries(startup_compute_builder_gate_runtime PRIVATE d3d12_xs_shim d3d12 dxgi)
set(startup_builder_gate_header "${CMAKE_CURRENT_BINARY_DIR}/startup_builder_gate_cs.h")
add_custom_command(OUTPUT "${startup_builder_gate_header}"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_0 -E main -Vn builder_gate_cs
        -Fh "${startup_builder_gate_header}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_builder_gate.hlsl"
    DEPENDS tests/startup_builder_gate.hlsl VERBATIM)
target_sources(startup_compute_builder_gate_runtime PRIVATE "${startup_builder_gate_header}")
target_include_directories(startup_compute_builder_gate_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_compute_builder_gate_runtime
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_builder_gate_runtime_check.py"
            "$<TARGET_FILE:startup_compute_builder_gate_runtime>" "$<TARGET_FILE:startup_builder_module>")
endif()

# Slot 67 request publication must keep its normal path with the gate enabled: slot 67 never
# carries the slot 62 builder tag.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    add_test(NAME startup_compute_builder_gate_request_warp
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_request_check.py"
            "$<TARGET_FILE:startup_stream_consumer>")
    set_tests_properties(startup_compute_builder_gate_request_warp PROPERTIES
        ENVIRONMENT "X2W_COMPUTE_BUILDER_GATE=1" TIMEOUT 180)
endif()
