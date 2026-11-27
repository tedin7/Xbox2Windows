if(X2W_HAVE_DXCAPI AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_frexp_f16_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_frexp_f16_check.py"
            controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_frexp_f16_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_frexp_f16_hardware
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_frexp_f16_check.py"
            hardware "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_frexp_f16_hardware PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60)
endif()
# No GPU and no DXC: evaluates the emitted IR (inline float sources). native_compute_ir_dump is
# defined in native_f16_alu.cmake, which is included before this file.
if(TARGET native_compute_ir_dump AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_frexp_f16_ir
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_frexp_f16_check.py"
            ir "$<TARGET_FILE:native_compute_ir_dump>")
    set_tests_properties(native_frexp_f16_ir PROPERTIES TIMEOUT 60)
endif()
