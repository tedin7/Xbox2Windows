# v_pack_b32_f16 in the direct C++ DXIL path: an IR check on pixel containers (no GPU) and a compute readback.
if(TARGET compile_dxil AND TARGET native_store_replay AND EXISTS "${INTAKE_PYTHON}")
    add_executable(native_pixel_ir_dump tests/native_pixel_ir_dump.cpp)
    target_include_directories(native_pixel_ir_dump PRIVATE src)
    target_link_libraries(native_pixel_ir_dump PRIVATE x2w_shader_compiler d3d12 dxgi)
    set_property(TARGET native_pixel_ir_dump PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_test(NAME native_pack_b32_f16_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_pack_b32_f16_check.py" ir "$<TARGET_FILE:native_pixel_ir_dump>")
    add_test(NAME native_pack_b32_f16_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_pack_b32_f16_check.py" gpu "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_pack_b32_f16_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
