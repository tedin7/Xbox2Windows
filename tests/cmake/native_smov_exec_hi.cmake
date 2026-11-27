if(X2W_HAVE_DXCAPI AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_smov_exec_hi_wave64
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_smov_exec_hi_check.py"
            wave64 "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_smov_exec_hi_wave64 PROPERTIES TIMEOUT 60)
    add_test(NAME native_smov_exec_hi_hardware
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_smov_exec_hi_check.py"
            hardware "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_smov_exec_hi_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
