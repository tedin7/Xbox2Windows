# SDWA binary16 arithmetic in the direct C++ compute path.
if(TARGET native_compute_ir_dump)
    add_test(NAME native_sdwa_f16_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sdwa_f16_check.py" ir "$<TARGET_FILE:native_compute_ir_dump>")
    add_test(NAME native_sdwa_f16_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sdwa_f16_check.py" gpu "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_sdwa_f16_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
