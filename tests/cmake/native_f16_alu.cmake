# v_add_f16, v_sub_f16, v_subrev_f16 and v_mul_f16 in the direct C++ compute path.
if(TARGET compile_dxil AND TARGET native_store_replay AND EXISTS "${INTAKE_PYTHON}")
    add_executable(native_compute_ir_dump tests/native_compute_ir_dump.cpp)
    target_include_directories(native_compute_ir_dump PRIVATE src)
    target_link_libraries(native_compute_ir_dump PRIVATE x2w_shader_compiler d3d12 dxgi)
    set_property(TARGET native_compute_ir_dump PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_test(NAME native_f16_alu_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_f16_alu_check.py" ir "$<TARGET_FILE:native_compute_ir_dump>")
    add_test(NAME native_f16_alu_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_f16_alu_check.py" gpu "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_f16_alu_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
