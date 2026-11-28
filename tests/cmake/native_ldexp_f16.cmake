if(TARGET compile_dxil AND TARGET native_store_replay AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_ldexp_f16_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_ldexp_f16_check.py"
            controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_ldexp_f16_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_ldexp_f16_hardware
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_ldexp_f16_check.py"
            hardware "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_ldexp_f16_hardware PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60)
endif()
