# Scalar sources in direct C++ binary16 SDWA arithmetic.
if(TARGET native_compute_ir_dump)
    add_test(NAME native_sdwa_f16_scalar_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sdwa_f16_scalar_check.py"
        ir "$<TARGET_FILE:native_compute_ir_dump>")
endif()
if(TARGET native_store_replay)
    add_test(NAME native_sdwa_f16_scalar_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sdwa_f16_scalar_check.py"
        gpu "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_sdwa_f16_scalar_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
